"""
dsp/transmitter.py -- TRANSMISSION SOFTWARE (DATA -> LIGHT)
===========================================================
(Migrated from module1_transmitter.py of the Orbital-Datacentre optical link simulator; the command-line interface was dropped.)

Converts a user-defined payload (text string or file) into a simulated optical
field produced by an electro-optic IQ Mach-Zehnder modulator (MZM).

Signal chain
------------
    payload bytes
      -> framing        [32-bit length | payload | CRC-32]
      -> bit stream
      -> Gray-mapped PSK/QAM symbols       (BPSK, QPSK, 8PSK, 16QAM)
      -> [known preamble | data symbols]
      -> root-raised-cosine pulse shaping  (baseband complex envelope)
      -> IQ Mach-Zehnder modulator model   (sin() transfer, optional pre-distortion)
      -> optical E-field envelope scaled to the launch power  [sqrt(W)]

NOTHING about the payload is hardcoded: call transmit(payload_bytes, ...) with any bytes.

This file also hosts the DSP primitives shared by the channel and the receiver
(constellations, RRC filter, slicer, framing, container I/O).
"""
from __future__ import annotations

import dataclasses
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

# --------------------------------------------------------------------------- #
# Physical / protocol constants
# --------------------------------------------------------------------------- #
C_LIGHT = 299_792_458.0          # speed of light in vacuum        [m/s]
PLANCK = 6.62607015e-34          # Planck constant                 [J*s]

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}
HEADER_BITS = 32                 # payload length field (bytes), big-endian
CRC_BITS = 32                    # CRC-32 of the payload
PREAMBLE_SEED = 0x0B17A1         # preamble is a fixed, publicly known sequence


# --------------------------------------------------------------------------- #
# Bit / byte helpers and framing
# --------------------------------------------------------------------------- #
def bytes_to_bits(data: bytes) -> np.ndarray:
    """bytes -> array of 0/1 (MSB first)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits) -> bytes:
    """array of 0/1 -> bytes (MSB first); trailing partial byte is dropped."""
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits[: len(bits) - len(bits) % 8]).tobytes()


def build_frame_bits(payload: bytes) -> np.ndarray:
    """[32-bit length][payload][CRC-32] as a bit array."""
    header = len(payload).to_bytes(4, "big")
    crc = zlib.crc32(payload).to_bytes(4, "big")
    return bytes_to_bits(header + payload + crc)


def parse_frame_bits(bits):
    """
    Inverse of build_frame_bits.  Returns (payload | None, crc_ok, status_text).
    Robust to a corrupted header (a bogus length never causes a huge allocation).
    """
    if len(bits) < HEADER_BITS + CRC_BITS:
        return None, False, "frame too short to contain header + CRC"
    length = int.from_bytes(bits_to_bytes(bits[:HEADER_BITS]), "big")
    total = HEADER_BITS + 8 * length + CRC_BITS
    if total > len(bits):
        return None, False, (f"header declares {length} bytes but only "
                             f"{(len(bits) - HEADER_BITS - CRC_BITS) // 8} are available "
                             "(header corrupted)")
    payload = bits_to_bytes(bits[HEADER_BITS:HEADER_BITS + 8 * length])
    crc_rx = int.from_bytes(bits_to_bytes(bits[HEADER_BITS + 8 * length:total]), "big")
    ok = crc_rx == zlib.crc32(payload)
    return payload, ok, "CRC-32 OK" if ok else "CRC-32 MISMATCH (payload corrupted)"


# --------------------------------------------------------------------------- #
# Constellations (Gray mapped, unit average symbol energy)
# --------------------------------------------------------------------------- #
def _gray(n: int) -> int:
    return n ^ (n >> 1)


def constellation(name: str) -> np.ndarray:
    """Complex points indexed by the integer value of their bit label (MSB first)."""
    name = name.upper()
    if name == "BPSK":
        pts = np.array([1, -1], dtype=complex)
    elif name == "QPSK":
        # b0 -> sign of I, b1 -> sign of Q  (Gray by construction)
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    elif name == "8PSK":
        pts = np.zeros(8, dtype=complex)
        for k in range(8):
            pts[_gray(k)] = np.exp(2j * np.pi * k / 8)
    elif name == "16QAM":
        levels = np.array([-3, -1, 1, 3], dtype=float)
        pts = np.zeros(16, dtype=complex)
        for pi in range(4):
            for pq in range(4):
                pts[(_gray(pi) << 2) | _gray(pq)] = levels[pi] + 1j * levels[pq]
        pts /= np.sqrt(10.0)
    else:
        raise ValueError(f"unknown modulation '{name}'; choose from {list(BITS_PER_SYMBOL)}")
    return pts


def bits_to_symbols(bits: np.ndarray, name: str):
    """Pack bits into labels and map to constellation points. Returns (symbols, n_pad_bits)."""
    bps = BITS_PER_SYMBOL[name.upper()]
    n_pad = (-len(bits)) % bps
    padded = np.concatenate([bits, np.zeros(n_pad, dtype=np.uint8)]).reshape(-1, bps)
    labels = padded @ (1 << np.arange(bps - 1, -1, -1))
    return constellation(name)[labels], n_pad


def labels_to_bits(labels: np.ndarray, name: str) -> np.ndarray:
    bps = BITS_PER_SYMBOL[name.upper()]
    shifts = np.arange(bps - 1, -1, -1)
    return ((np.asarray(labels)[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def symbols_to_bits(symbols: np.ndarray, name: str, chunk: int = 65536) -> np.ndarray:
    """Hard-decision (minimum-distance) slicer: symbols -> bits."""
    pts = constellation(name)
    labels = np.empty(len(symbols), dtype=np.int64)
    for i in range(0, len(symbols), chunk):          # chunked to bound memory
        blk = symbols[i:i + chunk]
        labels[i:i + chunk] = np.argmin(np.abs(blk[:, None] - pts[None, :]), axis=1)
    return labels_to_bits(labels, name)


def make_preamble(n: int) -> np.ndarray:
    """Known pseudo-random QPSK preamble used by the receiver for sync + phase/gain reference."""
    rng = np.random.default_rng(PREAMBLE_SEED)
    return (rng.choice([-1.0, 1.0], n) + 1j * rng.choice([-1.0, 1.0], n)) / np.sqrt(2)


# --------------------------------------------------------------------------- #
# Pulse shaping
# --------------------------------------------------------------------------- #
def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """
    Root-raised-cosine filter, unit energy (sum h^2 = 1), odd length span*sps + 1.
    Tx and Rx each apply it, so the cascade is a Nyquist (ISI-free) raised cosine.
    """
    if not 0.0 < beta <= 1.0:
        raise ValueError("roll-off must satisfy 0 < beta <= 1")
    span += span % 2                      # even span -> integer-sample group delay
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps  # time in symbol periods
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 + beta * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = ((np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
                    / (np.pi * ti * (1 - (4 * beta * ti) ** 2)))
    return h / np.sqrt(np.sum(h ** 2))


# --------------------------------------------------------------------------- #
# Electro-optic modulator
# --------------------------------------------------------------------------- #
def iq_mach_zehnder(baseband: np.ndarray, drive: float = 0.7, predistort: bool = True) -> np.ndarray:
    """
    Dual-parallel (IQ) Mach-Zehnder modulator biased at null (push-pull).

    Each arm's optical field transfer is   E_out/E_in = sin(pi/2 * v / V_pi),
    with v/V_pi the normalised drive voltage.  The RF drive is scaled so the
    largest excursion reaches `drive` (fraction of the linear range, 0 < drive <= 1).

    predistort=True applies the inverse arcsin so the optical field is an exact
    copy of the baseband waveform; False drives the MZM linearly and exposes its
    sine compression (visible as extra BER on 8PSK/16QAM at high `drive`).
    """
    if not 0.0 < drive <= 1.0:
        raise ValueError("drive must satisfy 0 < drive <= 1")
    peak = max(np.max(np.abs(baseband.real)), np.max(np.abs(baseband.imag)))
    x = drive * baseband / peak                                   # target amplitude in [-drive, drive]
    v = (2 / np.pi) * np.arcsin(x.real) + 1j * (2 / np.pi) * np.arcsin(x.imag) if predistort else x
    return np.sin(np.pi / 2 * v.real) + 1j * np.sin(np.pi / 2 * v.imag)


# --------------------------------------------------------------------------- #
# Container I/O shared by all modules (numpy arrays + JSON metadata in one .npz)
# --------------------------------------------------------------------------- #
def save_container(path, arrays: dict, meta: dict) -> Path:
    path = Path(path)
    np.savez_compressed(path, __meta__=json.dumps(meta), **arrays)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load_container(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
    return arrays, meta


# --------------------------------------------------------------------------- #
# Transmit product
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class OpticalSignal:
    """Everything Module 2 needs to know about the transmitted beam."""
    field: np.ndarray            # complex E-field envelope, |E|^2 in Watts
    frame_bits: np.ndarray       # [header|payload|CRC] bits (ground truth, for scoring only)
    modulation: str
    symbol_rate: float           # [baud]
    sps: int                     # samples per symbol
    rolloff: float
    rrc_span: int
    wavelength_m: float
    tx_power_w: float            # mean launch power
    n_preamble: int
    n_data_symbols: int
    payload_len: int             # bytes
    modulator_error_db: float    # MZM waveform error vs ideal (more negative = more linear)

    # ---- derived quantities --------------------------------------------------
    @property
    def bits_per_symbol(self) -> int:
        return BITS_PER_SYMBOL[self.modulation]

    @property
    def sample_rate(self) -> float:
        return self.symbol_rate * self.sps

    @property
    def bit_rate(self) -> float:
        return self.symbol_rate * self.bits_per_symbol

    @property
    def n_symbols(self) -> int:
        return self.n_preamble + self.n_data_symbols

    @property
    def duration_s(self) -> float:
        return self.n_symbols / self.symbol_rate

    # ---- persistence -----------------------------------------------------------
    def meta(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if not isinstance(getattr(self, f.name), np.ndarray)}

    def save(self, path) -> Path:
        return save_container(path, {"field": self.field, "frame_bits": self.frame_bits}, self.meta())

    @classmethod
    def load(cls, path) -> "OpticalSignal":
        arrays, meta = load_container(path)
        return cls(field=arrays["field"], frame_bits=arrays["frame_bits"], **meta)


# --------------------------------------------------------------------------- #
# Main entry point of the module
# --------------------------------------------------------------------------- #
def transmit(payload: bytes,
             modulation: str = "QPSK",
             symbol_rate: float = 1e9,
             sps: int = 8,
             rolloff: float = 0.35,
             rrc_span: int = 16,
             wavelength_m: float = 1550e-9,
             tx_power_w: float = 1.0,
             n_preamble: int = 64,
             drive: float = 0.7,
             predistort: bool = True) -> OpticalSignal:
    """Convert `payload` (any bytes you supply) into a simulated modulated optical field."""
    modulation = modulation.upper()
    if not payload:
        raise ValueError("payload is empty")
    if sps < 2:
        raise ValueError("sps must be >= 2")

    # 1) framing + symbol mapping
    frame_bits = build_frame_bits(payload)
    data_syms, _ = bits_to_symbols(frame_bits, modulation)
    all_syms = np.concatenate([make_preamble(n_preamble), data_syms])

    # 2) RRC pulse shaping -> baseband complex envelope with unit mean power
    h = rrc_taps(rolloff, sps, rrc_span)
    up = np.zeros(len(all_syms) * sps, dtype=complex)
    up[::sps] = all_syms
    baseband = np.sqrt(sps) * fftconvolve(up, h, mode="full")       # mean |x|^2 ~ 1

    # 3) electro-optic modulator, then scale to the requested launch power
    mod_out = iq_mach_zehnder(baseband, drive=drive, predistort=predistort)
    field = np.sqrt(tx_power_w) * mod_out / np.sqrt(np.mean(np.abs(mod_out) ** 2))

    # modulator fidelity: waveform error after removing the best complex scale
    ref = baseband / np.sqrt(np.mean(np.abs(baseband) ** 2))
    out_n = mod_out / np.sqrt(np.mean(np.abs(mod_out) ** 2))
    k = np.vdot(ref, out_n) / np.vdot(ref, ref)
    err_db = 10 * np.log10(max(np.mean(np.abs(out_n - k * ref) ** 2), 1e-30))

    return OpticalSignal(field=field, frame_bits=frame_bits, modulation=modulation,
                         symbol_rate=float(symbol_rate), sps=int(sps), rolloff=float(rolloff),
                         rrc_span=int(rrc_span + rrc_span % 2), wavelength_m=float(wavelength_m),
                         tx_power_w=float(tx_power_w), n_preamble=int(n_preamble),
                         n_data_symbols=len(data_syms), payload_len=len(payload),
                         modulator_error_db=float(err_db))


