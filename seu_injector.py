"""
seu_injector.py - cosmic-ray Single-Event Upsets (SEUs) for the 32 server blades of the Alpha-Sigma orbital datacentre.

A proton or heavy ion crossing a memory cell can flip one bit of the GPU's HBM/VRAM. This module answers three questions:

  1. HOW OFTEN?  The rate depends on where the satellite is (`flux_multiplier`): a nominal LEO baseline, 15x over the polar horns
                 (|lat| > 60 deg) and 100x over the South Atlantic Anomaly (lat -50..0, lon -90..+40), scaled by the altitude and by
                 a solar-flare factor you choose.
  2. WHAT BIT?   `inject_ieee754_bitflip` flips ONE random bit of a float32 / bfloat16 / float16 / float64 tensor by viewing it as raw
                 integers and XOR-ing `1 << bit`, and reports whether the hit was a MANTISSA bit (tiny noise), the SIGN bit (the weight is
                 negated) or an EXPONENT bit (the value can be scaled by 2^k, up to ~1e38 or NaN/Inf).
  3. WHAT ELSE?  A rare Single-Event Latchup (SEL) takes a whole blade offline until the 48 V busbar breaker power-cycles its tray (15 s).

HONESTY: every number here is a SIMULATION ASSUMPTION chosen by the satSTK author, not a measurement. The 1e-5 SEU/blade/s baseline, the
15x and 100x zone multipliers, the zone boxes, the 300 km altitude scale, the SEL fraction and the 15 s power-cycle are all placeholders.
Real upset rates depend on the memory technology, shielding, solar cycle and orbit, and this model has not been calibrated against any
flight or beam-test data. Nothing here is a flight-qualification claim.

Works on NumPy arrays and, when PyTorch is installed, on torch tensors; no other dependency.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

# ---- the simulation assumptions (all placeholders, see the module docstring) ----------------------------------------------------------
LAMBDA_BASE = 1.0e-5          # upsets per blade per second, nominal LEO
POLAR_LAT_DEG = 60.0          # |latitude| above this is a "polar horn" (auroral zone)
POLAR_MULT = 15.0
SAA_LAT_DEG = (-50.0, 0.0)    # the South Atlantic Anomaly box
SAA_LON_DEG = (-90.0, 40.0)
SAA_MULT = 100.0
FLARE_MULT = 1000.0           # "Solar Flare / CME" chaos event, multiplies the flux
REF_ALT_KM = 600.0            # altitude at which the altitude factor is 1
ALT_SCALE_KM = 300.0          # exp((alt - 600) / 300), clipped to 0.25 .. 4 (a plain exponential, not a measured profile)
SEL_FRACTION = 2.0e-3         # latchups per upset (so a latchup is ~1 in 500 events)
SEL_OFFLINE_S = 15.0          # the busbar breaker power-cycles the tray after this long
P_WEIGHT_HIT = 0.7            # share of upsets that land in the resident weights (the rest corrupt a gradient being computed)
MAX_UPSETS_PER_BLADE_STEP = 64

NOMINAL, POLAR_N, POLAR_S, SAA = "NOMINAL", "POLAR_N", "POLAR_S", "SAA"
ZONE_MULT = {NOMINAL: 1.0, POLAR_N: POLAR_MULT, POLAR_S: POLAR_MULT, SAA: SAA_MULT}
ZONE_LABEL = {NOMINAL: "Nominal LEO", POLAR_N: "Polar horn (N)", POLAR_S: "Polar horn (S)", SAA: "South Atlantic Anomaly"}


# ---- geomagnetic zones ------------------------------------------------------------------------------------------------------------------
def wrap_lon(lon_deg: float) -> float:
    return (float(lon_deg) + 180.0) % 360.0 - 180.0


def in_saa(lat_deg: float, lon_deg: float) -> bool:
    return SAA_LAT_DEG[0] <= lat_deg <= SAA_LAT_DEG[1] and SAA_LON_DEG[0] <= wrap_lon(lon_deg) <= SAA_LON_DEG[1]


def classify_zone(lat_deg: float, lon_deg: float) -> str:
    """NOMINAL, POLAR_N, POLAR_S or SAA for a sub-satellite point."""
    if in_saa(lat_deg, lon_deg):
        return SAA
    if lat_deg > POLAR_LAT_DEG:
        return POLAR_N
    if lat_deg < -POLAR_LAT_DEG:
        return POLAR_S
    return NOMINAL


def altitude_factor(alt_km: float) -> float:
    """1.0 at 600 km; rises/falls exponentially with a 300 km scale (assumption). Clipped to 0.25 .. 4."""
    return float(np.clip(math.exp((alt_km - REF_ALT_KM) / ALT_SCALE_KM), 0.25, 4.0))


def flux_multiplier(lat_deg: float, lon_deg: float, alt_km: float = REF_ALT_KM, flare: float = 1.0) -> float:
    """Total flux relative to the nominal LEO baseline at 600 km: zone x altitude x flare."""
    return ZONE_MULT[classify_zone(lat_deg, lon_deg)] * altitude_factor(alt_km) * float(flare)


def seu_rate_per_blade_s(lat_deg: float, lon_deg: float, alt_km: float = REF_ALT_KM, flare: float = 1.0) -> float:
    return LAMBDA_BASE * flux_multiplier(lat_deg, lon_deg, alt_km, flare)


# ---- IEEE-754 bit flips ---------------------------------------------------------------------------------------------------------------
# name: (total bits, exponent bits, mantissa bits)
LAYOUTS = {"float32": (32, 8, 23), "bfloat16": (16, 8, 7), "float16": (16, 5, 10), "float64": (64, 11, 52)}
_UINT = {32: np.uint32, 16: np.uint16, 64: np.uint64}
_BLOWUP_ABS = {"float32": 1e30, "bfloat16": 1e30, "float16": 6.0e4, "float64": 1e300}     # "explosion" = non-finite or beyond this


def field_of_bit(bit: int, dtype: str = "float32") -> str:
    """'sign', 'exponent' or 'mantissa' for a bit position (0 = least significant) of the given IEEE-754-style format."""
    width, e_bits, m_bits = LAYOUTS[dtype]
    if not 0 <= bit < width:
        raise ValueError(f"bit {bit} out of range for {dtype} ({width} bits)")
    return "mantissa" if bit < m_bits else ("exponent" if bit < m_bits + e_bits else "sign")


def _is_torch(t) -> bool:
    return type(t).__module__.split(".")[0] == "torch"


def dtype_name(tensor) -> str:
    if _is_torch(tensor):
        name = str(tensor.dtype).replace("torch.", "")
    else:
        name = np.dtype(tensor.dtype).name
    if name not in LAYOUTS:
        raise TypeError(f"cannot bit-flip a {name} tensor (supported: {', '.join(LAYOUTS)})")
    return name


@dataclass
class BitFlip:
    index: int
    bit: int
    field: str            # 'mantissa' | 'exponent' | 'sign'
    dtype: str
    old: float
    new: float
    blowup: bool          # the new value is NaN/Inf or astronomically large (an "exponent explosion")

    @property
    def effect(self) -> str:
        """Plain-words result: minor noise, inverted weight, explosion/NaN, or an exponent change that stayed finite and moderate."""
        if self.blowup:
            return "explosion to NaN/Inf" if not math.isfinite(self.new) else f"explosion to {self.new:.2e}"
        return {"mantissa": "minor noise", "sign": "inverted weight", "exponent": "rescaled by a power of two"}[self.field]


def bits_of(x: float, dtype: str = "float32") -> int:
    """The raw bit pattern of one number (for tests and teaching)."""
    width = LAYOUTS[dtype][0]
    if dtype == "bfloat16":
        return int(np.float32(x).view(np.uint32)) >> 16
    return int(np.array([x], dtype=dtype).view(_UINT[width])[0])


def flip_bit_value(x: float, bit: int, dtype: str = "float32") -> float:
    """What a single number becomes when `bit` is flipped (pure function, no tensor needed)."""
    width = LAYOUTS[dtype][0]
    pattern = bits_of(x, dtype) ^ (1 << bit)
    if dtype == "bfloat16":
        return float(np.array([pattern << 16], dtype=np.uint32).view(np.float32)[0])
    return float(np.array([pattern], dtype=_UINT[width]).view(dtype)[0])


def _blowup(new: float, dtype: str) -> bool:
    return (not math.isfinite(new)) or abs(new) >= _BLOWUP_ABS[dtype]


def inject_ieee754_bitflip(tensor, index: int | None = None, bit: int | None = None, rng: np.random.Generator | None = None) -> BitFlip:
    """
    Flip ONE bit of ONE element of `tensor`, in place, and describe the hit.

    The tensor's memory is viewed as raw unsigned integers (uint32 for float32, uint16 for bfloat16 / float16, uint64 for float64) and the chosen bit
    is toggled with XOR:   raw[index] ^= (1 << bit).   Torch has no unsigned 32-bit XOR in every release, so a torch tensor is viewed as the signed
    integer of the same width instead (the same bits, a different label).

    `index` / `bit` pick the element and the bit position (0 = least significant); leave them None for uniformly random choices from `rng`.
    float32 layout: bit 31 sign, bits 30..23 exponent, bits 22..0 mantissa. Flipping bit 30 of a small weight scales it by 2^128 (~1e38);
    if the exponent becomes all ones the value is Inf or NaN.
    """
    dt = dtype_name(tensor)
    width = LAYOUTS[dt][0]
    rng = rng or np.random.default_rng()
    if _is_torch(tensor):
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")
        n = tensor.numel()
    else:
        if not tensor.flags.c_contiguous:
            raise ValueError("array must be C-contiguous")
        n = tensor.size
    if n == 0:
        raise ValueError("empty tensor")
    index = int(rng.integers(n)) if index is None else int(index)
    bit = int(rng.integers(width)) if bit is None else int(bit)
    if not 0 <= index < n:
        raise IndexError(f"element {index} out of range for {n} elements")
    field = field_of_bit(bit, dt)
    if _is_torch(tensor):
        import torch
        flat = tensor.view(-1)
        raw = flat.view({32: torch.int32, 16: torch.int16, 64: torch.int64}[width])
        old = float(flat[index])
        mask = 1 << bit
        if mask >= 1 << (width - 1):
            mask -= 1 << width                  # the same bit pattern as a negative signed integer
        raw[index] = raw[index] ^ mask
        new = float(flat[index])
    else:
        flat = tensor.reshape(-1)
        raw = flat.view(_UINT[width])
        old = float(flat[index])
        raw[index] ^= _UINT[width](1 << bit)
        new = float(flat[index])
    return BitFlip(index, bit, field, dt, old, new, _blowup(new, dt))


# ---- when and where the upsets happen -------------------------------------------------------------------------------------------------
@dataclass
class Upset:
    """One radiation event. The uniform numbers make it replayable: the same Upset flips the same bit of the same element on any system."""
    blade: int
    kind: str             # 'SEU' (bit flip) or 'SEL' (latchup)
    when: float           # 0..1, how far through the time step it happens
    target: str           # 'weights' or 'grads' (SEU only)
    u_index: float        # in [0,1): which element
    u_bit: float          # in [0,1): which bit


class SEUInjector:
    """Draws the Poisson-distributed upsets that hit each blade during a time step. Seeded, so a run can be repeated."""

    def __init__(self, seed: int = 0, n_blades: int = 32, sel_fraction: float = SEL_FRACTION, p_weight: float = P_WEIGHT_HIT):
        self.rng = np.random.default_rng(seed)
        self.n_blades = n_blades
        self.sel_fraction = sel_fraction
        self.p_weight = p_weight

    def draw(self, dt_s: float, lat_deg: float, lon_deg: float, alt_km: float = REF_ALT_KM, flare: float = 1.0, boost: float = 1.0) -> list:
        lam = seu_rate_per_blade_s(lat_deg, lon_deg, alt_km, flare) * boost * dt_s
        n_seu = np.minimum(self.rng.poisson(lam, self.n_blades), MAX_UPSETS_PER_BLADE_STEP)
        n_sel = self.rng.poisson(lam * self.sel_fraction, self.n_blades)
        out = []
        for b in range(self.n_blades):
            for _ in range(int(n_seu[b])):
                u = self.rng.random(4)
                out.append(Upset(b, "SEU", float(u[0]), "weights" if u[1] < self.p_weight else "grads", float(u[2]), float(u[3])))
            for _ in range(int(n_sel[b])):
                out.append(Upset(b, "SEL", float(self.rng.random()), "", 0.0, 0.0))
        out.sort(key=lambda e: e.when)
        return out


def predict_saa_entry_s(position_fn, t_s: float, lookahead_s: float = 300.0, sample_s: float = 30.0):
    """
    Seconds until the satellite is inside the SAA box, looking `lookahead_s` ahead (0.0 if it is inside now, None if it will not be).
    `position_fn(t)` returns (lat_deg, lon_deg) - for example from the orbit engine's ground track.
    """
    t = 0.0
    while t <= lookahead_s + 1e-9:
        lat, lon = position_fn(t_s + t)[:2]
        if in_saa(lat, lon):
            return t
        t += sample_s
    return None


if __name__ == "__main__":
    print(__doc__)
    for name, (lat, lon) in {"mid-Pacific": (0.0, -150.0), "Brazil coast (SAA)": (-25.0, -45.0), "Arctic": (75.0, 20.0)}.items():
        print(f"{name:20s} zone {classify_zone(lat, lon):8s} flux x{flux_multiplier(lat, lon):6.1f}  {seu_rate_per_blade_s(lat, lon):.1e} SEU/blade/s")
    sys.exit(0)
