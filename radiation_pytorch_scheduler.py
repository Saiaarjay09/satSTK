"""
radiation_pytorch_scheduler.py - a radiation-tolerant distributed training orchestrator for the 32-blade Alpha-Sigma orbital datacentre.

The datacentre is a 1:1 satCAD design: 2 banks x 16 server trays (Bank A = blades 1-16, Bank B = blades 17-32). This module trains a small neural
network (a compact MLP classifier) data-parallel across the 32 simulated blades while cosmic-ray upsets (seu_injector.py) flip real IEEE-754 bits
in the blades' weights and gradients, and shows how four layers of protection keep the training alive:

  1. SANITIZER   each blade's gradient gets a fast hash (CRC-32), an L2 norm and an isnan/isinf test. A blade whose gradient is non-finite, or whose
                 norm exceeds 5x the median norm, is QUARANTINED for that step (its gradient is not averaged in).
  2. TMR         when satSTK's orbit predicts the South Atlantic Anomaly (or a flare), the blades switch from 32 independent batches to Triple
                 Modular Redundancy: 10 triads (5 per bank, blades 16 and 32 are hot spares) run the SAME batch and a 2-of-3 majority vote on the
                 gradient hashes picks the answer.
  3. ECC + OBC   every blade's weights carry a per-tensor CRC-32 "parity" that is checked every step; a mismatch restores just the damaged tensor from
                 the central avionics computer (OBC), which keeps a SHA-256 verified checkpoint every N steps (mirrored in two copies) plus a journal of
                 the updates since, over a modelled 100 Gbps optical backplane. The other 31 blades keep training while one is repaired.
  4. THERMAL     on SOFT THROTTLE (core >= 75 C) Bank B hibernates and Bank A takes over the whole batch.

An UNPROTECTED twin (no sanitizer, vote, parity or checkpoint) is driven by exactly the same radiation events, so the two loss curves can be compared.

PyTorch is used when installed; otherwise a NumPy tensor engine with a hand-written backward pass gives the same maths (same weights and batch give the
same loss and gradient to ~1e-6, which the tests check).

HONESTY: the upset rates and zone multipliers are the author's simulation assumptions (seu_injector.py), the 15 s latchup recovery, the 100 Gbps
backplane and the 50 ms budget are modelled numbers, the 'thermal state' is read from satSTK's placeholder cooling-loop model (it is not fed back
by this workload), and the 'blades' are simulated replicas inside one Python process. This shows an algorithm working; it is NOT a flight-qualified
design and it says nothing about the real fault rate of any hardware. No Reed-Solomon code is used: the checkpoint is protected by SHA-256 and a mirror.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import seu_injector as si

try:                                   # PyTorch is optional: without it the NumPy engine below does the same job
    import torch
    TORCH_AVAILABLE = True
except Exception:                      # ImportError, or a broken install
    torch = None
    TORCH_AVAILABLE = False

N_BLADES = 32
BANK_A, BANK_B = tuple(range(0, 16)), tuple(range(16, 32))
TRIADS = tuple(tuple(range(s, s + 3)) for s in (0, 3, 6, 9, 12, 16, 19, 22, 25, 28))      # 5 per bank
SPARES = {"A": 15, "B": 31}                                                              # blade 16 and blade 32
SOFT_THROTTLE_C, SOFT_RELEASE_C = 75.0, 70.0
NORM_FACTOR = 5.0
BACKPLANE_GBPS, BACKPLANE_LATENCY_MS, RESTORE_BUDGET_MS = 100.0, 0.05, 50.0
FLASH_STEPS = 3

HEALTHY, SCRUBBED, QUARANTINED, TMR_STATE, HIBERNATED, RESET = "healthy", "scrubbed", "quarantined", "tmr", "hibernated", "reset"
DP, TMR = "DP", "TMR"


def bank_of(blade: int) -> str:
    return "A" if blade < 16 else "B"


def blade_name(blade: int) -> str:
    return f"{bank_of(blade)}{blade % 16 + 1:02d}"


# ====================================================================================================================================
# The workload: a small MLP classifier, with a PyTorch engine and a NumPy engine that agree
# ====================================================================================================================================
LAYER_SHAPES = (("fc1.weight", (48, 16)), ("fc1.bias", (48,)), ("fc2.weight", (32, 48)), ("fc2.bias", (32,)), ("fc3.weight", (4, 32)), ("fc3.bias", (4,)))
SLICES = []
_o = 0
for _name, _shape in LAYER_SHAPES:
    SLICES.append((_name, _shape, _o, _o + int(np.prod(_shape))))
    _o += int(np.prod(_shape))
N_PARAMS = _o                                                                            # 2,516 float32 weights (~10 KB)


def as_numpy(t) -> np.ndarray:
    """A NumPy view of a blade's tensor (shares memory, so bit flips made through either are seen by both)."""
    return t if isinstance(t, np.ndarray) else t.numpy()


def _unpack(w):
    return [w[a:b].reshape(shape) for _, shape, a, b in SLICES]


class NumpyEngine:
    """The fallback tensor engine: forward and backward pass of the MLP in plain NumPy float32."""
    name = "numpy"

    def native(self, arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=np.float32).copy()

    def loss_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray):
        with np.errstate(all="ignore"):
            W1, b1, W2, b2, W3, b3 = _unpack(w)
            z1 = X @ W1.T + b1
            h1 = np.maximum(z1, 0)
            z2 = h1 @ W2.T + b2
            h2 = np.maximum(z2, 0)
            z = h2 @ W3.T + b3
            m = z.max(axis=1, keepdims=True)
            e = np.exp(z - m)
            s = e.sum(axis=1, keepdims=True)
            p = e / s
            n = len(y)
            loss = float(np.mean(np.log(s[:, 0]) + m[:, 0] - z[np.arange(n), y]))
            dz = p.copy()
            dz[np.arange(n), y] -= 1.0
            dz /= n
            g = np.empty(N_PARAMS, dtype=np.float32)
            gW3, gb3 = dz.T @ h2, dz.sum(axis=0)
            d2 = (dz @ W3) * (z2 > 0)
            gW2, gb2 = d2.T @ h1, d2.sum(axis=0)
            d1 = (d2 @ W2) * (z1 > 0)
            gW1, gb1 = d1.T @ X, d1.sum(axis=0)
            for (_, _, a, b), part in zip(SLICES, (gW1, gb1, gW2, gb2, gW3, gb3)):
                g[a:b] = part.reshape(-1)
        return loss, g

    def logits(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            W1, b1, W2, b2, W3, b3 = _unpack(w)
            return np.maximum(np.maximum(X @ W1.T + b1, 0) @ W2.T + b2, 0) @ W3.T + b3


class TorchEngine:
    """The same MLP in PyTorch (autograd). Blades hold real torch tensors, so the bit flips hit PyTorch memory."""
    name = "torch"

    def native(self, arr: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32).copy())

    def loss_grad(self, w, X: np.ndarray, y: np.ndarray):
        wt = w.detach().clone().requires_grad_(True)
        W1, b1, W2, b2, W3, b3 = _unpack(wt)
        h = torch.relu(torch.relu(torch.from_numpy(X) @ W1.T + b1) @ W2.T + b2)
        loss = torch.nn.functional.cross_entropy(h @ W3.T + b3, torch.from_numpy(y).long())
        loss.backward()
        return float(loss.detach()), wt.grad.detach().contiguous()

    def logits(self, w, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            W1, b1, W2, b2, W3, b3 = _unpack(w)
            return (torch.relu(torch.relu(torch.from_numpy(X) @ W1.T + b1) @ W2.T + b2) @ W3.T + b3).numpy()


def get_engine(prefer: str = "auto"):
    """'auto' uses PyTorch when it is installed and NumPy otherwise; 'torch' insists on PyTorch; 'numpy' forces the fallback."""
    if prefer == "numpy" or (prefer == "auto" and not TORCH_AVAILABLE):
        return NumpyEngine()
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed; use engine='numpy' (or 'auto') for the NumPy fallback")
    return TorchEngine()


def init_weights(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.zeros(N_PARAMS, dtype=np.float32)
    for name, shape, a, b in SLICES:
        if name.endswith("weight"):
            w[a:b] = (rng.standard_normal(shape) * math.sqrt(2.0 / shape[1])).reshape(-1)
    return w


class SyntheticTask:
    """4 Gaussian blobs in 16 dimensions. `batch(step, k)` is a pure function of (seed, step, k), so every replica of a batch is identical."""

    def __init__(self, seed: int = 0, batch_size: int = 64, d_in: int = 16, n_classes: int = 4):
        self.seed, self.batch_size, self.d_in, self.n_classes = seed, batch_size, d_in, n_classes
        self.centers = np.random.default_rng([seed, 99]).standard_normal((n_classes, d_in)).astype(np.float32) * 1.12
        self._cache: dict = {}
        self.val_X, self.val_y = self._make(np.random.default_rng([seed, 7]), 512)

    def _make(self, rng, n: int):
        y = rng.integers(self.n_classes, size=n)
        X = (self.centers[y] + rng.standard_normal((n, self.d_in)).astype(np.float32)).astype(np.float32)
        return X, y.astype(np.int64)

    def batch(self, step: int, k: int):
        key = (step, k)
        if key not in self._cache:
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = self._make(np.random.default_rng([self.seed, 1, step, k]), self.batch_size)
        return self._cache[key]


# ====================================================================================================================================
# Layer 1: the gradient sanitizer; layer 2: the TMR voter
# ====================================================================================================================================
def tensor_hash(arr: np.ndarray) -> int:
    """Fast hash of a tensor's raw bytes (CRC-32)."""
    return zlib.crc32(np.ascontiguousarray(arr).tobytes()) & 0xFFFFFFFF


@dataclass
class SanitizeReport:
    accept: list          # bool per gradient (None gradients are rejected)
    norms: list           # L2 norm per gradient (nan if missing)
    hashes: list          # CRC-32 per gradient
    reasons: list         # '' if accepted, else 'missing' | 'nan/inf' | 'norm > 5x median'
    median_norm: float


def sanitize_gradients(grads: list, factor: float = NORM_FACTOR) -> SanitizeReport:
    """
    Per-blade gradient screening. A gradient is quarantined if it is missing, contains NaN/Inf, or its L2 norm is more than `factor` times the median
    of the finite norms. (With only one or two gradients the median is not a robust yardstick, so only the NaN/Inf test is meaningful.)
    """
    n = len(grads)
    norms, hashes, reasons, finite = [float("nan")] * n, [0] * n, [""] * n, [False] * n
    for i, g in enumerate(grads):
        if g is None:
            reasons[i] = "missing"
            continue
        a = as_numpy(g)
        hashes[i] = tensor_hash(a)
        with np.errstate(all="ignore"):
            nrm = float(np.linalg.norm(a.astype(np.float64)))
        norms[i] = nrm
        if not np.isfinite(a).all() or not math.isfinite(nrm):
            reasons[i] = "nan/inf"
        else:
            finite[i] = True
    good = [norms[i] for i in range(n) if finite[i]]
    med = float(np.median(good)) if good else float("nan")
    accept = [False] * n
    for i in range(n):
        if finite[i]:
            if med > 0 and norms[i] > factor * med:
                reasons[i] = f"norm > {factor:g}x median"
            else:
                accept[i] = True
    return SanitizeReport(accept, norms, hashes, reasons, med)


@dataclass
class VoteResult:
    winner: int | None    # index into the members list of a gradient from the majority, or None
    agree: list           # bool per member: same hash as the winner
    status: str           # 'unanimous' | 'majority' | 'duplex-agree' | 'no-majority' | 'single'


def tmr_vote(grads: list) -> VoteResult:
    """2-of-3 majority vote on gradient hashes. Missing (None) members abstain; with two members they must agree (duplex); one member is unvoted."""
    hashes = [None if g is None else tensor_hash(as_numpy(g)) for g in grads]
    present = [i for i, h in enumerate(hashes) if h is not None]
    if not present:
        return VoteResult(None, [False] * len(grads), "no-majority")
    if len(present) == 1:
        return VoteResult(present[0], [h is not None for h in hashes], "single")
    counts: dict = {}
    for i in present:
        counts.setdefault(hashes[i], []).append(i)
    best = max(counts.values(), key=len)
    if len(present) == 2:
        if len(best) == 2:
            return VoteResult(best[0], [h == hashes[best[0]] for h in hashes], "duplex-agree")
        return VoteResult(None, [False] * len(grads), "no-majority")
    if len(best) >= 2:
        return VoteResult(best[0], [h == hashes[best[0]] for h in hashes], "unanimous" if len(best) == 3 else "majority")
    return VoteResult(None, [False] * len(grads), "no-majority")


# ====================================================================================================================================
# Layer 3: ECC parity and the OBC checkpoint store
# ====================================================================================================================================
def tensor_crcs(w: np.ndarray) -> list:
    """The per-tensor CRC-32 'parity word' of a weight vector (one per layer tensor)."""
    return [zlib.crc32(w[a:b].tobytes()) & 0xFFFFFFFF for _, _, a, b in SLICES]


def sha256_hex(w: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()


def apply_update(w: np.ndarray, g: np.ndarray, lr: float) -> None:
    """The one SGD update used by every blade and by the OBC replay, so replaying the journal reproduces the weights bit for bit."""
    with np.errstate(all="ignore"):                      # NaN / Inf gradients are expected in the unprotected baseline
        w -= np.float32(lr) * g.astype(np.float32, copy=False)


def backplane_ms(n_bytes: int) -> float:
    """Modelled time to move `n_bytes` over the internal optical backplane: a fixed latency plus bits / 100 Gbps."""
    return BACKPLANE_LATENCY_MS + n_bytes * 8.0 / (BACKPLANE_GBPS * 1e9) * 1e3


class OBCStore:
    """
    The central avionics computer's memory: a live reference copy of the weights (SHA-256 checked every step), a SHA-256 verified checkpoint taken every
    `interval` steps in two mirrored copies, and the journal of aggregated updates since that checkpoint. If the live copy fails its SHA-256 it is rebuilt
    from the newest valid checkpoint copy plus the journal (bit-exact). Assumed radiation-hardened; the mirror and the hash are the safety net, not RS codes.
    """

    def __init__(self, w0: np.ndarray, interval: int = 10):
        self.interval = max(1, int(interval))
        self.live = w0.astype(np.float32).copy()
        self.live_sha = sha256_hex(self.live)
        self.step = 0
        self.journal: list = []
        self.ckpts = [self._snapshot(), self._snapshot()]
        self.rebuilds = 0
        self.mirror_failovers = 0
        self.checkpoints_taken = 1

    def _snapshot(self) -> dict:
        return {"step": self.step, "w": self.live.copy(), "sha": self.live_sha}

    def apply(self, g: np.ndarray, lr: float) -> None:
        self.reference()
        apply_update(self.live, g, lr)
        self.live_sha = sha256_hex(self.live)
        self.journal.append((g.astype(np.float32).copy(), lr))
        self.step += 1
        if self.step % self.interval == 0:
            self.ckpts = [self._snapshot(), self._snapshot()]
            self.journal = []
            self.checkpoints_taken += 1

    def checkpoint_now(self) -> None:
        """Take a checkpoint immediately (verified first): used when the power system warns that the battery is low."""
        self.reference()
        self.ckpts = [self._snapshot(), self._snapshot()]
        self.journal = []
        self.checkpoints_taken += 1

    def reference(self) -> np.ndarray:
        """The verified current weights. Rebuilds from checkpoint + journal if the live copy has been corrupted."""
        if sha256_hex(self.live) != self.live_sha:
            self.rebuild()
        return self.live

    def rebuild(self) -> None:
        for k, c in enumerate(self.ckpts):
            if sha256_hex(c["w"]) == c["sha"]:
                if k > 0:
                    self.mirror_failovers += 1
                w = c["w"].copy()
                for g, lr in self.journal:
                    apply_update(w, g, lr)
                self.live, self.live_sha = w, sha256_hex(w)
                self.rebuilds += 1
                return
        raise RuntimeError("both OBC checkpoint copies failed their SHA-256 check")


# ====================================================================================================================================
# The blades and the trainer
# ====================================================================================================================================
@dataclass
class Blade:
    idx: int
    w: object                                   # the blade's resident weights (torch tensor or numpy array)
    offline_until: float = 0.0                  # mission time when a latchup power-cycle ends
    hibernated: bool = False
    flash: str = ""                             # transient colour: SCRUBBED or QUARANTINED
    flash_left: int = 0
    in_tmr: bool = False
    was_down: bool = False
    pending: list = field(default_factory=list)   # bit flips this blade suffered since its weights were last checked
    hits: int = 0
    scrubs: int = 0
    quarantines: int = 0
    restores: int = 0
    latchups: int = 0
    last_event: str = ""

    def online(self, t: float) -> bool:
        return t >= self.offline_until and not self.hibernated


@dataclass
class StepEnv:
    t_s: float
    dt_s: float
    lat_deg: float
    lon_deg: float
    alt_km: float
    core_temp_c: float = 40.0
    saa_eta_s: float | None = None        # seconds until the SAA (0 inside, None if not within the look-ahead)
    flare: float = 1.0
    bank_b_powered: bool = True           # the EPS's Bank B 48 V solid-state breaker (False: Bank B has no power)
    dvfs_speed: float = 1.0               # EPS DVFS eco-clocking: relative training speed of the powered blades
    eps_mode: str | None = None           # the EPS state machine's mode (e.g. "ECLIPSE SURVIVAL")


@dataclass
class StepReport:
    step: int
    t_s: float
    mode: str
    zone: str
    flux_mult: float
    throttled: bool
    loss: float                       # mean batch loss the training loop saw (nan if poisoned)
    hits: int = 0
    latchups: int = 0
    flips: list = field(default_factory=list)          # (blade, BitFlip) for every SEU applied
    quarantined: list = field(default_factory=list)    # (blade, reason)
    scrubbed: list = field(default_factory=list)       # (blade, tensor name, restore ms)
    voted_out: list = field(default_factory=list)      # blades outvoted by their triad
    vote_failed: int = 0
    accepted_units: int = 0
    unique_batches: int = 0
    rel_throughput: float = 1.0
    updated: bool = False
    max_restore_ms: float = 0.0
    events: list = field(default_factory=list)         # human-readable lines


@dataclass
class TrainerConfig:
    lr: float = 0.1
    checkpoint_interval: int = 10
    lookahead_s: float = 300.0
    shield_flux: float = 50.0          # TMR whenever the effective flux is at least this many x baseline (SAA = 100, flare = 1000)
    soft_throttle_c: float = SOFT_THROTTLE_C
    soft_release_c: float = SOFT_RELEASE_C
    sel_offline_s: float = si.SEL_OFFLINE_S
    norm_factor: float = NORM_FACTOR


class RadiationTolerantTrainer:
    """One 32-blade training system. `protected=False` turns off the sanitizer, the TMR vote, the parity check and the OBC (the baseline)."""

    def __init__(self, engine, task: SyntheticTask, w0: np.ndarray, protected: bool = True, cfg: TrainerConfig | None = None):
        self.engine, self.task, self.protected = engine, task, protected
        self.cfg = cfg or TrainerConfig()
        self.blades = [Blade(i, engine.native(w0)) for i in range(N_BLADES)]
        self.obc = OBCStore(w0, self.cfg.checkpoint_interval)
        self.step_no = 0
        self.mode = DP
        self.throttled = False
        self.totals = dict(hits=0, mantissa=0, exponent=0, sign=0, blowups=0, latchups=0, quarantines=0, scrubs=0, restores=0, voted_out=0, vote_failed=0,
                           skipped_updates=0, tmr_steps=0, throttled_steps=0, max_restore_ms=0.0)
        self.poisoned = False
        self._t = 0.0
        self._eps_survival = False
        self._bank_b_powered = True
        self.last_loss = float("nan")

    # ---- helpers -----------------------------------------------------------------------------------------------------------------
    def weights_np(self, b: int) -> np.ndarray:
        return as_numpy(self.blades[b].w)

    def consensus_weights(self) -> np.ndarray:
        """The model a user would export: the OBC's verified weights when protected, else blade 1's (or the first online) weights."""
        if self.protected:
            return self.obc.reference().copy()
        for b in self.blades:
            if not b.hibernated:
                return self.weights_np(b.idx).copy()
        return self.weights_np(0).copy()

    def evaluate(self, w: np.ndarray) -> tuple:
        """(cross-entropy loss, accuracy) of weights `w` on the fixed validation set."""
        z = NumpyEngine().logits(w, self.task.val_X)
        with np.errstate(all="ignore"):
            m = z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z - m).sum(axis=1)) + m[:, 0]
            loss = float(np.mean(lse - z[np.arange(len(z)), self.task.val_y]))
        return loss, float(np.mean(z.argmax(axis=1) == self.task.val_y))

    def display_state(self, b: int) -> str:
        bl = self.blades[b]
        if bl.hibernated:
            return HIBERNATED
        if bl.offline_until > self._t:
            return RESET
        if bl.flash_left > 0 and bl.flash:
            return bl.flash
        return TMR_STATE if bl.in_tmr else HEALTHY

    def _restore_tensor(self, blade: Blade, ti: int, ref: np.ndarray) -> float:
        a, b = SLICES[ti][2], SLICES[ti][3]
        self.weights_np(blade.idx)[a:b] = ref[a:b]
        blade.restores += 1
        self.totals["restores"] += 1
        return backplane_ms((b - a) * 4)

    def _restore_all(self, blade: Blade) -> float:
        self.weights_np(blade.idx)[:] = self.obc.reference()
        blade.restores += 1
        self.totals["restores"] += 1
        return backplane_ms(N_PARAMS * 4)

    def _flash(self, blade: Blade, colour: str) -> None:
        if colour == QUARANTINED or blade.flash != QUARANTINED or blade.flash_left <= 0:
            blade.flash, blade.flash_left = colour, FLASH_STEPS

    def set_protected(self, on: bool) -> None:
        """Switch the protection on or off at run time. Switching ON re-seeds the OBC from the blades' per-tensor majority (it missed the updates while off)."""
        if on and not self.protected:
            live = [b.idx for b in self.blades if b.online(self._t)] or [0]
            ref = np.zeros(N_PARAMS, dtype=np.float32)
            for k, (_, _, a, b_) in enumerate(SLICES):
                crcs = [tensor_crcs(self.weights_np(i))[k] for i in live]
                best = max(set(crcs), key=crcs.count)
                ref[a:b_] = self.weights_np(live[crcs.index(best)])[a:b_]
            self.obc = OBCStore(ref, self.cfg.checkpoint_interval)
        self.protected = bool(on)
        if not on:
            self.mode = DP

    # ---- one training step -------------------------------------------------------------------------------------------------------
    def step(self, env: StepEnv, upsets: list) -> StepReport:
        cfg, t0, dt = self.cfg, env.t_s, env.dt_s
        self._t = t0 + dt
        self.step_no += 1
        zone = si.classify_zone(env.lat_deg, env.lon_deg)
        flux = si.flux_multiplier(env.lat_deg, env.lon_deg, env.alt_km, env.flare)
        rep = StepReport(self.step_no, t0, self.mode, zone, flux, self.throttled, float("nan"))
        for b in self.blades:
            b.flash_left = max(0, b.flash_left - 1)
        # -- thermal coupling (power management: applies to both systems) --
        if not self.throttled and env.core_temp_c >= cfg.soft_throttle_c:
            self.throttled = True
            rep.events.append(f"SOFT THROTTLE at {env.core_temp_c:.1f} C: Bank B hibernates, Bank A takes the whole batch")
        elif self.throttled and env.core_temp_c <= cfg.soft_release_c:
            self.throttled = False
            rep.events.append(f"core cooled to {env.core_temp_c:.1f} C: Bank B wakes")
        rep.throttled = self.throttled
        self.totals["throttled_steps"] += int(self.throttled)
        survival = env.eps_mode == "ECLIPSE SURVIVAL"
        if survival and not self._eps_survival:
            if self.protected:
                self.obc.checkpoint_now()
                rep.events.append("EPS low-battery survival: weights flushed to a fresh OBC checkpoint before Bank B loses power")
            else:
                rep.events.append("EPS low-battery survival (no checkpoint: protection is off)")
        self._eps_survival = survival
        if env.bank_b_powered != self._bank_b_powered:
            rep.events.append("EPS: Bank B 48 V breaker OPEN: blades 17-32 lose power, Bank A takes the whole batch" if not env.bank_b_powered
                              else "EPS: Bank B 48 V breaker closed: blades 17-32 power up")
            self._bank_b_powered = env.bank_b_powered
        for b in self.blades:
            b.hibernated = (self.throttled or not env.bank_b_powered) and b.idx in BANK_B
        for b in self.blades:                                                       # blades coming back from a power cycle or from hibernation
            up = b.online(t0)
            if b.was_down and up:
                if self.protected:
                    ms = self._restore_all(b)
                    rep.max_restore_ms = max(rep.max_restore_ms, ms)
                    rep.events.append(f"blade {blade_name(b.idx)} back online: weights restored from the OBC checkpoint in {ms:.3f} ms")
                else:
                    rep.events.append(f"blade {blade_name(b.idx)} back online with whatever its memory held")
            b.was_down = not up
        # -- the radiation events of this step --
        active_set = {b.idx for b in self.blades if b.online(t0)}
        late, grad_hits = [], {}
        for u in upsets:
            b = self.blades[u.blade]
            if u.blade not in active_set:
                continue                                                           # a powered-down blade cannot be upset
            if u.kind == "SEL":
                b.offline_until = t0 + u.when * dt + cfg.sel_offline_s
                b.latchups += 1
                b.last_event = "latchup: breaker power-cycle"
                self.weights_np(b.idx)[:] = 0.0                                    # a power cycle clears the blade's volatile memory
                b.was_down = True
                b.flash, b.flash_left = RESET, FLASH_STEPS
                rep.latchups += 1
                self.totals["latchups"] += 1
                rep.events.append(f"SEL latchup on blade {blade_name(b.idx)}: offline {cfg.sel_offline_s:g} s until the 48 V breaker power-cycles the tray")
                active_set.discard(u.blade)
                continue
            rep.hits += 1
            self.totals["hits"] += 1
            b.hits += 1
            if u.target == "weights":
                if u.when >= 0.5:
                    late.append(u)
                else:
                    self._hit_weights(b, u, rep)
            else:
                grad_hits.setdefault(u.blade, []).append(u)
        # -- layer 3: ECC parity check on every blade's weights, repair from the OBC --
        if self.protected:
            ref = self.obc.reference()
            ref_crc = tensor_crcs(ref)
            for i in sorted(active_set):
                b = self.blades[i]
                bad = [k for k, c in enumerate(tensor_crcs(self.weights_np(i))) if c != ref_crc[k]]
                for k in bad:
                    ms = self._restore_tensor(b, k, ref)
                    rep.scrubbed.append((i, SLICES[k][0], ms))
                    rep.max_restore_ms = max(rep.max_restore_ms, ms)
                if bad:
                    blowup = any(f.blowup for f in b.pending)
                    self._flash(b, QUARANTINED if blowup else SCRUBBED)
                    b.scrubs += 1
                    self.totals["scrubs"] += 1
                    what = ", ".join(f"{f.field} bit {f.bit}" for f in b.pending) or "parity mismatch"
                    b.last_event = f"parity mismatch ({what}) scrubbed"
                    rep.events.append(f"blade {blade_name(i)} parity check failed ({what}); restored {', '.join(SLICES[k][0] for k in bad)} from the OBC in "
                                      f"{sum(backplane_ms((SLICES[k][3] - SLICES[k][2]) * 4) for k in bad):.3f} ms; other blades kept training")
                b.pending = []
        else:
            for i in active_set:
                self.blades[i].pending = []
        for u in late:                                                              # upsets after the parity check, before the compute
            if u.blade in active_set:
                self._hit_weights(self.blades[u.blade], u, rep)
        # -- mode (layer 2) --
        eta = env.saa_eta_s
        want_tmr = self.protected and (flux >= cfg.shield_flux or (eta is not None and eta <= cfg.lookahead_s))
        new_mode = TMR if want_tmr else DP
        if new_mode != self.mode:
            rep.events.append(("entering the SAA / high flux: switching to TMR shielding (2-of-3 voting)" if new_mode == TMR
                               else "radiation eased: back to high-throughput data-parallel mode") + (f" (SAA in {eta:.0f} s)" if eta and new_mode == TMR else ""))
            self.mode = new_mode
        rep.mode = self.mode
        self.totals["tmr_steps"] += int(self.mode == TMR)
        # -- form the compute units and give them batches --
        units = self._units(sorted(active_set))
        total_batches = 10 if self.mode == TMR else N_BLADES
        for b in self.blades:
            b.in_tmr = self.mode == TMR and b.online(t0)
        ids_for = [list(range(total_batches))[u::len(units)] for u in range(len(units))] if units else []
        rep.unique_batches = total_batches if units else 0
        per_unit = max((len(x) for x in ids_for), default=1)
        rep.rel_throughput = (total_batches / per_unit / N_BLADES * env.dvfs_speed) if units else 0.0
        # -- compute the gradients (every blade computes with ITS OWN weights; gradient upsets hit the result) --
        unit_grads, unit_losses = [], []
        for members, ids in zip(units, ids_for):
            m_grads, m_losses = [], []
            for m in members:
                gs, ls = [], []
                for k in ids:
                    X, y = self.task.batch(self.step_no, k)
                    loss, g = self.engine.loss_grad(self.blades[m].w, X, y)
                    gs.append(as_numpy(g).copy())
                    ls.append(loss)
                g = gs[0] if len(gs) == 1 else np.mean(gs, axis=0, dtype=np.float32)
                for u in grad_hits.get(m, []):
                    flip = si.inject_ieee754_bitflip(g, index=int(u.u_index * N_PARAMS), bit=int(u.u_bit * 32))
                    self._count_flip(flip)
                    rep.flips.append((m, flip))
                    rep.events.append(f"blade {blade_name(m)} gradient hit: {flip.field} bit {flip.bit} -> {flip.effect}")
                    self.blades[m].last_event = f"gradient {flip.field} flip ({flip.effect})"
                    if flip.blowup:
                        self._flash(self.blades[m], QUARANTINED)
                m_grads.append(g)
                m_losses.append(float(np.mean(ls)))
            unit_grads.append((members, m_grads))
            unit_losses.append(m_losses)
        # -- aggregation --
        losses_seen = [x for ls in unit_losses for x in ls]
        agg = None
        if not units:
            rep.events.append("no blade online: step skipped")
        elif not self.protected:
            flat = [g for _, gs in unit_grads for g in gs]
            with np.errstate(all="ignore"):
                agg = np.mean(flat, axis=0, dtype=np.float32)
            rep.loss = float(np.mean(losses_seen))
            rep.accepted_units = len(flat)
        else:
            cand, cand_loss = [], []
            for (members, gs), ls in zip(unit_grads, unit_losses):
                if self.mode == TMR and len(members) >= 2:
                    v = tmr_vote(gs)
                    for m, ok in zip(members, v.agree):
                        if not ok:
                            self.blades[m].quarantines += 1
                            self.totals["voted_out"] += 1
                            rep.voted_out.append(m)
                            self._flash(self.blades[m], QUARANTINED)
                            self.blades[m].last_event = "outvoted by the other two blades"
                            rep.events.append(f"TMR: blade {blade_name(m)} outvoted 2-of-3 on its gradient hash")
                    if v.winner is None:
                        rep.vote_failed += 1
                        self.totals["vote_failed"] += 1
                        rep.events.append(f"TMR: no majority in triad of blade {blade_name(members[0])}; its batch is dropped this step")
                        continue
                    cand.append((members[v.winner], gs[v.winner]))
                    cand_loss.append(ls[v.winner])
                else:
                    for m, g, l in zip(members, gs, ls):
                        cand.append((m, g))
                        cand_loss.append(l)
            rep_s = sanitize_gradients([g for _, g in cand], cfg.norm_factor)
            kept, kept_loss = [], []
            for (m, g), ok, why, l in zip(cand, rep_s.accept, rep_s.reasons, cand_loss):
                if ok:
                    kept.append(g)
                    kept_loss.append(l)
                else:
                    rep.quarantined.append((m, why))
                    b = self.blades[m]
                    b.quarantines += 1
                    self.totals["quarantines"] += 1
                    self._flash(b, QUARANTINED)
                    b.last_event = f"gradient quarantined ({why})"
                    rep.events.append(f"blade {blade_name(m)} gradient quarantined ({why}); the others were averaged without it")
            rep.accepted_units = len(kept)
            if kept:
                rep.loss = float(np.mean(kept_loss))
                agg = np.mean(kept, axis=0, dtype=np.float32)
            else:
                rep.loss = self.last_loss                                           # nothing usable this step: hold the last good loss, skip the update

        # -- the update --
        if agg is not None:
            for i in sorted(active_set):
                apply_update(self.weights_np(i), agg, cfg.lr)
            if self.protected:
                self.obc.apply(agg, cfg.lr)
            rep.updated = True
            if not np.isfinite(agg).all():
                self.poisoned = True
        else:
            self.totals["skipped_updates"] += 1
        if not math.isnan(rep.loss) and not math.isfinite(rep.loss):
            self.poisoned = True
        self.totals["max_restore_ms"] = max(self.totals["max_restore_ms"], rep.max_restore_ms)
        if not math.isnan(rep.loss):
            self.last_loss = rep.loss
        return rep

    def _units(self, active: list) -> list:
        """The compute units for this step: single blades (data-parallel) or triads with hot-spare substitution (TMR)."""
        if self.mode == DP:
            return [[i] for i in active]
        aset, used, units = set(active), set(), []
        for tri in TRIADS:
            members = [m for m in tri if m in aset]
            if len(members) < 3:
                sp = SPARES[bank_of(tri[0])]
                if sp in aset and sp not in used and sp not in members:
                    members.append(sp)
                    used.add(sp)
            if members:
                units.append(members)
        return units

    def _count_flip(self, flip) -> None:
        self.totals[flip.field] += 1
        self.totals["blowups"] += int(flip.blowup)

    def _hit_weights(self, blade: Blade, u, rep: StepReport) -> None:
        flip = si.inject_ieee754_bitflip(blade.w, index=int(u.u_index * N_PARAMS), bit=int(u.u_bit * 32))
        self._count_flip(flip)
        blade.pending.append(flip)
        blade.last_event = f"weight {flip.field} flip ({flip.effect})"
        rep.flips.append((blade.idx, flip))
        rep.events.append(f"blade {blade_name(blade.idx)} weight hit: {flip.field} bit {flip.bit} ({flip.old:+.3g} -> {flip.new:+.3g}): {flip.effect}")


# ====================================================================================================================================
# Orbit and thermal sources, and the orchestrator that drives the protected system and its unprotected twin
# ====================================================================================================================================
class TrackOrbit:
    """Latitude / longitude / altitude from satSTK's propagated ground track, interpolated in time."""

    def __init__(self, t_s: np.ndarray, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray):
        self.t, self.lat, self.alt = np.asarray(t_s, float), np.asarray(lat, float), np.asarray(alt, float)
        self.lon_rad = np.unwrap(np.radians(np.asarray(lon, float)))

    @classmethod
    def from_result(cls, result, sat: int | None = None) -> "TrackOrbit":
        tr = result.tracks[result.target if sat is None else sat]
        return cls(result.t_s, tr.lat_deg, tr.lon_deg, tr.alt_km)

    def position(self, t: float) -> tuple:
        t = float(np.clip(t, self.t[0], self.t[-1]))
        lon = si.wrap_lon(math.degrees(float(np.interp(t, self.t, self.lon_rad))))
        return float(np.interp(t, self.t, self.lat)), lon, float(np.interp(t, self.t, self.alt))

    def saa_eta(self, t: float, lookahead_s: float) -> float | None:
        return si.predict_saa_entry_s(self.position, t, lookahead_s)


class CoreTemp:
    """Core temperature from satSTK's cooling-loop model (thermal_loop.py), interpolated in time. The workload does not feed back into it."""

    def __init__(self, result, thermal_history=None):
        if thermal_history is None:
            import thermal_loop as tl
            tr = result.tracks[result.target]
            thermal_history = tl.simulate_loop(tr.radiator_solar_load_w, result.config.step_s, result.config.thermal)
        self.t = np.asarray(result.t_s, float)
        self.temp = np.asarray(thermal_history.core_temp_c, float)

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.t, self.temp))


@dataclass
class OrchestratorConfig:
    seed: int = 1
    base_step_s: float = 20.0            # mission seconds one training step stands for
    max_steps_per_call: int = 4
    engine: str = "auto"
    protected: bool = True
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


class RadiationOrchestrator:
    """
    Drives a protected 32-blade trainer and an unprotected twin through the same orbit with the SAME radiation events.
    `advance_to(t)` runs as many training steps as the mission time that has passed calls for.
    """

    def __init__(self, orbit, thermal=None, cfg: OrchestratorConfig | None = None, eps=None):
        self.cfg = cfg or OrchestratorConfig()
        self.orbit, self.thermal = orbit, thermal
        self.eps = eps                       # an eps_power_engine.EPSHistory: Bank B breaker, DVFS and survival mode follow it
        self.engine = get_engine(self.cfg.engine)
        self.task = SyntheticTask(self.cfg.seed)
        self.injector = si.SEUInjector(self.cfg.seed + 1000)
        w0 = init_weights(self.cfg.seed)
        self.protected = RadiationTolerantTrainer(self.engine, self.task, w0, self.cfg.protected, self.cfg.trainer)
        self.baseline = RadiationTolerantTrainer(self.engine, self.task, w0, False, self.cfg.trainer)
        self.t = float(orbit.t[0]) if hasattr(orbit, "t") else 0.0
        self.flare = 1.0
        self.temp_offset_c = 0.0
        self.step_no = 0
        self.history = dict(step=[], t=[], loss_protected=[], loss_baseline=[], flux=[], zone=[], mode=[], rel_throughput=[])
        self.log: list = []                  # the last 400 event lines (mission time + what the protected system did)
        self._unread = 0
        self.last: tuple = (None, None)
        self._forced: list = []

    # ---- chaos controls ---------------------------------------------------------------------------------------------------------
    def set_flare(self, on: bool) -> None:
        self.flare = si.FLARE_MULT if on else 1.0

    def set_protection(self, on: bool) -> None:
        self.protected.set_protected(on)

    def force_latchup(self, blade: int | None = None) -> int:
        """Chaos: a latchup on one blade at the start of the next step (a random online blade if none is given)."""
        if blade is None:
            blade = int(self.injector.rng.integers(N_BLADES))
        self.inject(si.Upset(blade, "SEL", 0.0, "", 0.0, 0.0))
        return blade

    def inject(self, upset) -> None:
        """Chaos / testing: add one hand-made upset (a seu_injector.Upset) to the next training step, on top of the random ones."""
        self._forced.append(upset)

    # ---- the clock --------------------------------------------------------------------------------------------------------------
    def resync(self, t: float) -> None:
        """Jump to a new mission time without training (a seek, a loop or a pause-and-scrub)."""
        self.t = float(t)

    def advance_to(self, t: float, max_jump_s: float = 3600.0) -> list:
        owed = t - self.t
        if owed < 0 or owed > max_jump_s:
            self.resync(t)
            return []
        n = min(self.cfg.max_steps_per_call, int((owed + 1e-6) // self.cfg.base_step_s))
        if n <= 0:
            return []
        dt = owed / n
        return [self.step(dt) for _ in range(n)]

    def step(self, dt: float):
        lat, lon, alt = self.orbit.position(self.t)
        env = StepEnv(self.t, dt, lat, lon, alt, (self.thermal(self.t) if self.thermal else 40.0) + self.temp_offset_c,
                      self.orbit.saa_eta(self.t, self.cfg.trainer.lookahead_s) if hasattr(self.orbit, "saa_eta") else None, self.flare)
        if self.eps is not None:
            k = self.eps.index_at(self.t)
            env.bank_b_powered, env.eps_mode = bool(self.eps.bank_b_closed[k]), self.eps.mode[k]
            env.dvfs_speed = self.eps.cfg.dvfs_speed if bool(self.eps.dvfs[k]) else 1.0
        upsets = self.injector.draw(dt, lat, lon, alt, self.flare) + self._forced
        self._forced = []
        rp = self.protected.step(env, upsets)
        rb = self.baseline.step(env, upsets)
        self.step_no += 1
        h = self.history
        h["step"].append(self.step_no)
        h["t"].append(self.t)
        h["loss_protected"].append(rp.loss)
        h["loss_baseline"].append(rb.loss)
        h["flux"].append(rp.flux_mult)
        h["zone"].append(rp.zone)
        h["mode"].append(rp.mode)
        h["rel_throughput"].append(rp.rel_throughput)
        for line in rp.events:
            self.note(line)
        self.last = (rp, rb)
        self.t += dt
        return rp, rb

    def note(self, line: str) -> None:
        self.log.append(f"T+{self.t / 3600:6.3f} h  {line}")
        self._unread += 1
        del self.log[:-400]

    def drain_log(self) -> list:
        """The event lines added since the last call (at most the 400 kept)."""
        out = self.log[len(self.log) - min(self._unread, len(self.log)):]
        self._unread = 0
        return out

    # ---- results ----------------------------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        lp, ap = self.protected.evaluate(self.protected.consensus_weights())
        lb, ab = self.baseline.evaluate(self.baseline.consensus_weights())
        return dict(steps=self.step_no, engine=self.engine.name, protected_totals=dict(self.protected.totals), baseline_totals=dict(self.baseline.totals),
                    protected_val_loss=lp, protected_val_acc=ap, baseline_val_loss=lb, baseline_val_acc=ab,
                    baseline_poisoned=self.baseline.poisoned, first_baseline_nan_step=next((s for s, x in zip(self.history["step"], self.history["loss_baseline"])
                                                                                            if not math.isfinite(x)), None))

    def export_weights(self, path) -> Path:
        """Write the recovered (OBC-verified) weights. A PyTorch `.pt` file when torch is installed, otherwise a NumPy `.npz` with the same names."""
        w = self.protected.consensus_weights()
        path = Path(path)
        named = {name: w[a:b].reshape(shape).copy() for name, shape, a, b in SLICES}
        meta = dict(step=self.protected.step_no, sha256=sha256_hex(w), engine=self.engine.name)
        if TORCH_AVAILABLE and path.suffix == ".pt":
            torch.save({"state_dict": {k: torch.from_numpy(v) for k, v in named.items()}, "meta": meta}, path)
        else:
            path = path.with_suffix(".npz")
            np.savez(path, **named, sha256=np.array(meta["sha256"]), step=np.array(meta["step"]))
        return path

    def blade_grid(self) -> list:
        """(state, tooltip) for each of the 32 blades, protected system."""
        out = []
        for i, b in enumerate(self.protected.blades):
            out.append((self.protected.display_state(i), f"Blade {blade_name(i)}: {b.last_event or 'nominal'} | hits {b.hits}, scrubs {b.scrubs}, "
                                                          f"quarantines {b.quarantines}, restores {b.restores}, latchups {b.latchups}"))
        return out


def load_checkpoint_meta(path) -> dict:
    """Read back an exported file (for the tests and for users who want to check the SHA-256)."""
    path = Path(path)
    if path.suffix == ".pt":
        d = torch.load(path, weights_only=True)
        return {"names": list(d["state_dict"]), **d["meta"]}
    d = np.load(path)
    return {"names": [k for k in d.files if k not in ("sha256", "step")], "sha256": str(d["sha256"]), "step": int(d["step"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the radiation-tolerant 32-blade training demo against satSTK's orbit (no window).")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--engine", choices=("auto", "torch", "numpy"), default="auto")
    ap.add_argument("--flare", action="store_true", help="a 1000x solar flare for the whole run")
    ap.add_argument("--export", metavar="FILE", help="write the recovered weights (.pt with torch, .npz otherwise)")
    ap.add_argument("--start-hours", type=float, default=0.0)
    args = ap.parse_args(argv)
    from simulation import SimulationConfig, run_simulation
    cfg = SimulationConfig(duration_h=max(args.hours + args.start_hours, 1.0), step_s=30.0)
    result = run_simulation(cfg)
    orch = RadiationOrchestrator(TrackOrbit.from_result(result), CoreTemp(result), OrchestratorConfig(seed=args.seed, engine=args.engine))
    orch.set_flare(args.flare)
    t, t_end = args.start_hours * 3600.0, (args.start_hours + args.hours) * 3600.0
    orch.resync(t)
    zone_prev = None
    while t < t_end:
        t += orch.cfg.base_step_s
        for rp, rb in orch.advance_to(t, max_jump_s=1e9):
            if rp.zone != zone_prev:
                print(f"T+{rp.t_s / 3600:6.3f} h  entering {si.ZONE_LABEL[rp.zone]} (flux x{rp.flux_mult:g}, mode {rp.mode})")
                zone_prev = rp.zone
    s = orch.summary()
    pt, bt = s["protected_totals"], s["baseline_totals"]
    print(f"engine {s['engine']}; {s['steps']} training steps over {args.hours:g} h; {pt['hits']} bit flips ({pt['mantissa']} mantissa, {pt['exponent']} exponent, "
          f"{pt['sign']} sign; {pt['blowups']} blow-ups), {pt['latchups']} latchups")
    print(f"PROTECTED : val loss {s['protected_val_loss']:.4f}, accuracy {100 * s['protected_val_acc']:.1f} %; {pt['quarantines']} gradients quarantined, "
          f"{pt['voted_out']} outvoted, {pt['scrubs']} blades scrubbed, {pt['restores']} restores (slowest {pt['max_restore_ms']:.3f} ms), {pt['tmr_steps']} TMR steps")
    nan_at = s["first_baseline_nan_step"]
    print(f"UNPROTECTED: val loss {s['baseline_val_loss']:.4f}, accuracy {100 * s['baseline_val_acc']:.1f} %"
          + (f"; went NaN/Inf at step {nan_at}" if nan_at else "; survived this run"))
    if args.export:
        print(f"Wrote {orch.export_weights(args.export)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
