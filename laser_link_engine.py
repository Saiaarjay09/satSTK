"""
laser_link_engine.py - 1550 nm optical inter-satellite link (OISL) budget, bit-error rate, Doppler and turret slew rates.

Plain NumPy / SciPy; every function accepts arrays so a whole 24-hour history is computed in one call.

How the link budget works (beginner version)
--------------------------------------------
A laser transmitter sends a narrow Gaussian beam. Its width grows in proportion to distance: a beam with full divergence
`theta` (100 microradians here) is `theta * range` wide, e.g. 200 m across at 2000 km. The receiving telescope only catches the
slice of the beam that falls on its aperture, so

    received power = transmit power x optics efficiencies x (fraction of the Gaussian beam caught) x (pointing loss)

* Gaussian capture fraction (called "beam divergence loss" in the results) = 1 - exp(-2 a^2 / w^2), where `a` is the receive
  aperture radius and `w = theta*range/2` the beam radius at the receiver. It is exact at every range.
* Free-space path loss (FSPL) = (4 pi range / wavelength)^2 is also reported. It is the loss between two ideal isotropic
  antennas; with the transmit gain 32/theta^2 and receive gain (pi D / wavelength)^2 it gives the same answer as the capture
  fraction in the far field (checked in the unit tests), so the two views are consistent.
* Pointing jitter: if the beam wobbles by a random angle (RMS `sigma` on each axis) the average power loss is
  1 / (1 + 16 sigma^2 / theta^2). A fixed pointing offset is handled too.

Receiver model: shot-noise-limited coherent (homodyne) detection with an extra "implementation penalty" in dB for real-world
imperfections. Noise density N0 = h*nu/2, so BPSK needs about 9 photons per bit for a 1e-9 error rate - the textbook
quantum limit. Bit-error-rate formulas are the standard Gray-coded AWGN approximations for BPSK, QPSK, 8-PSK and 16-QAM.

Honest limits: this is a link-budget calculator, not a waveform simulator. It does not model laser phase noise, receiver
electronics, acquisition and tracking, atmospheric effects (none in space-to-space links above the buffer) or the delay
introduced by forward error correction. Absolute numbers depend on the parameters you enter and are not validated
against measured hardware.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

C_M_S = 299_792_458.0
PLANCK = 6.62607015e-34   # J s

MODULATIONS = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}   # bits per symbol


@dataclass
class LaserLinkConfig:
    """All the numbers of the link. Edit freely; the GUI's Scenario dialog exposes the main ones."""
    wavelength_nm: float = 1550.0
    tx_power_w: float = 10.0
    tx_optics_efficiency: float = 0.80       # telescope + fibre/lens losses at the transmitter (0.80 = -0.97 dB)
    rx_optics_efficiency: float = 0.80
    tx_aperture_m: float = 0.30              # reported in the budget only (the beam divergence is what sets the spreading)
    rx_aperture_m: float = 0.30
    divergence_urad: float = 100.0           # full-angle 1/e^2 beam divergence
    pointing_jitter_urad: float = 10.0       # RMS pointing error per axis (one sigma)
    static_pointing_error_urad: float = 0.0  # fixed mis-pointing (bias)
    symbol_rate_gbaud: float = 250.0         # 250 Gbaud x 4 bits (16QAM) = 1 Tbit/s raw
    detector_efficiency: float = 0.80        # quantum efficiency of the balanced photodiodes
    implementation_penalty_db: float = 10.0  # everything the ideal shot-noise-limited receiver does not have
    fec_overhead: float = 0.07               # forward error correction adds this fraction of extra bits
    ber_threshold: float = 3.8e-3            # raw BER the FEC can fix (7 % hard-decision FEC); worse = no usable data
    processing_latency_ms: float = 0.0       # fixed extra delay on top of the speed-of-light delay (FEC, framing ...)
    ber_floor: float = 1e-30                 # BER below this is shown as this value, so log plots stay finite

    @property
    def frequency_hz(self) -> float:
        return C_M_S / (self.wavelength_nm * 1e-9)

    @property
    def photon_energy_j(self) -> float:
        return PLANCK * self.frequency_hz


# ----------------------------------------------------------------------------------------------------------------------
# Single-formula building blocks (each is unit-tested against a hand calculation)
# ----------------------------------------------------------------------------------------------------------------------
def fspl_db(range_m, wavelength_m: float):
    """Free-space path loss 20*log10(4*pi*R/lambda), in dB (a positive number)."""
    return 20.0 * np.log10(4.0 * np.pi * np.asarray(range_m, dtype=float) / wavelength_m)


def beam_radius_m(range_m, divergence_rad: float):
    """1/e^2 intensity radius of the Gaussian beam at range `R` (far-field, no waist term)."""
    return 0.5 * divergence_rad * np.asarray(range_m, dtype=float)


def gaussian_capture_fraction(range_m, divergence_rad: float, rx_aperture_m: float):
    """Fraction of a centred Gaussian beam's power that falls on a circular aperture of diameter `rx_aperture_m`."""
    w = np.maximum(beam_radius_m(range_m, divergence_rad), 1e-12)
    a = 0.5 * rx_aperture_m
    return -np.expm1(-2.0 * (a / w) ** 2)


def pointing_loss_linear(divergence_rad: float, jitter_rad: float, static_rad: float = 0.0) -> float:
    """
    Mean power fraction kept after pointing error. For a Gaussian beam the on-axis fraction at off-axis angle e is
    exp(-k e^2) with k = 8/theta^2. Averaging over a 2-axis Gaussian jitter of RMS sigma and a fixed offset mu gives
        exp(-k mu^2 / (1+2 k sigma^2)) / (1 + 2 k sigma^2).
    """
    k = 8.0 / divergence_rad ** 2
    denom = 1.0 + 2.0 * k * jitter_rad ** 2
    return math.exp(-k * static_rad ** 2 / denom) / denom


def tx_gain_db(divergence_rad: float) -> float:
    """On-axis gain of a Gaussian beam of full divergence theta, 32/theta^2 (relative to isotropic)."""
    return 10.0 * math.log10(32.0 / divergence_rad ** 2)


def rx_gain_db(aperture_m: float, wavelength_m: float) -> float:
    """Gain of a circular receive aperture, (pi D / lambda)^2."""
    return 20.0 * math.log10(math.pi * aperture_m / wavelength_m)


def watts_to_dbm(p_w):
    return 10.0 * np.log10(np.maximum(np.asarray(p_w, dtype=float), 1e-300)) + 30.0


def q_function(x):
    """Gaussian tail probability Q(x) = 0.5*erfc(x/sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / math.sqrt(2.0))


def ber_from_eb_n0(modulation: str, eb_n0_linear):
    """
    Bit error rate for coherent detection in additive white Gaussian noise, Gray coding, as a function of Eb/N0 (linear).
        BPSK, QPSK : Q(sqrt(2 Eb/N0))
        8PSK       : (2/3) Q( sqrt(6 Eb/N0) sin(pi/8) )     (nearest-neighbour approximation)
        16QAM      : (3/4) Q( sqrt(0.8 Eb/N0) )              (nearest-neighbour approximation)
    """
    g = np.maximum(np.asarray(eb_n0_linear, dtype=float), 0.0)
    if modulation in ("BPSK", "QPSK"):
        return q_function(np.sqrt(2.0 * g))
    if modulation == "8PSK":
        return (2.0 / 3.0) * q_function(np.sqrt(6.0 * g) * math.sin(math.pi / 8.0))
    if modulation == "16QAM":
        return 0.75 * q_function(np.sqrt(0.8 * g))
    raise ValueError(f"unknown modulation {modulation!r}; choose from {list(MODULATIONS)}")


# ----------------------------------------------------------------------------------------------------------------------
# Geometry: range rate, Doppler, turret angles and slew rates
# ----------------------------------------------------------------------------------------------------------------------
def relative_kinematics(r1, v1, r2, v2):
    """
    Range (km) and range rate (km/s, positive = moving apart) from satellite 1 to satellite 2, both in an inertial frame.
    The range rate is the component of the relative velocity along the line of sight (a dot product, so frame-independent).
    """
    d = np.atleast_2d(r2) - np.atleast_2d(r1)
    rng = np.linalg.norm(d, axis=1)
    dv = np.atleast_2d(v2) - np.atleast_2d(v1)
    rate = np.einsum("ij,ij->i", d, dv) / np.maximum(rng, 1e-12)
    return rng, rate


def doppler_shift_hz(range_rate_km_s, frequency_hz: float):
    """First-order Doppler shift of the received carrier: positive when the satellites approach (range rate negative)."""
    return -np.asarray(range_rate_km_s, dtype=float) * 1000.0 / C_M_S * frequency_hz


@dataclass
class TurretPointing:
    """Where one terminal must point, in that satellite's own orbit frame, and how fast it must slew."""
    azimuth_deg: np.ndarray        # in the local horizontal plane, 0 = flight direction, 90 = away from the orbit normal ...
    elevation_deg: np.ndarray      # above (+) or below (-) the local horizon; the horizon is perpendicular to the nadir
    azimuth_rate_deg_s: np.ndarray
    elevation_rate_deg_s: np.ndarray
    slew_rate_deg_s: np.ndarray    # total angular rate of the pointing vector relative to the satellite


def turret_pointing(r_own, v_own, r_other, v_other) -> TurretPointing:
    """
    Pointing direction and slew rate for the laser terminal on `own` that tracks `other`.

    The satellite is assumed nadir-pointing, so its body axes are the LVLH frame: z toward the Earth, y opposite the orbit
    normal, x roughly along the velocity. In that frame:  azimuth = atan2(u_y, u_x),  elevation = asin(u_up).
    The rates are exact (no finite differencing):  d(u)/dt in the frame = (inertial line-of-sight rate) - omega x u,
    with omega = (r x v)/r^2 the rotation rate of the frame.
    """
    r_own, v_own, r_other, v_other = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (r_own, v_own, r_other, v_other))
    d = r_other - r_own
    rng = np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
    u = d / rng
    dv = v_other - v_own
    u_dot = (dv - np.einsum("ij,ij->i", dv, u)[:, None] * u) / rng           # rad/s, inertial frame
    r_n = np.linalg.norm(r_own, axis=1)[:, None]
    r_hat = r_own / r_n
    h = np.cross(r_own, v_own)
    h_hat = h / np.linalg.norm(h, axis=1)[:, None]
    x_hat = np.cross(h_hat, r_hat)
    y_hat = -h_hat
    omega = h / r_n ** 2
    u_dot_f = u_dot - np.cross(omega, u)                                      # rate seen from inside the rotating frame
    ux, uy, uz = (np.einsum("ij,ij->i", u, ax) for ax in (x_hat, y_hat, r_hat))
    dx, dy, dz = (np.einsum("ij,ij->i", u_dot_f, ax) for ax in (x_hat, y_hat, r_hat))
    horiz2 = np.maximum(ux ** 2 + uy ** 2, 1e-18)
    az = np.degrees(np.arctan2(uy, ux)) % 360.0
    el = np.degrees(np.arcsin(np.clip(uz, -1, 1)))
    az_rate = np.degrees((ux * dy - uy * dx) / horiz2)
    el_rate = np.degrees(dz / np.sqrt(np.maximum(1.0 - uz ** 2, 1e-18)))
    slew = np.degrees(np.linalg.norm(u_dot_f, axis=1))
    return TurretPointing(az, el, az_rate, el_rate, slew)


# ----------------------------------------------------------------------------------------------------------------------
# The link budget at every time step
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class LinkResult:
    """Everything the GUI plots. Quantities that need a working beam are NaN while the Earth blocks the line of sight."""
    t_s: np.ndarray
    visible: np.ndarray                 # True = line of sight clear = "locked"
    clearance_km: np.ndarray            # beam-to-Earth closest approach (negative = blocked)
    range_km: np.ndarray
    range_rate_km_s: np.ndarray
    doppler_ghz: np.ndarray
    latency_ms: np.ndarray
    fspl_db: np.ndarray
    divergence_loss_db: np.ndarray      # Gaussian beam spreading loss (-10 log10 of the capture fraction)
    pointing_loss_db: float
    rx_power_dbm: np.ndarray
    snr_db: np.ndarray                  # Es/N0 at the symbol rate (the same for every modulation)
    eb_n0_db: dict                      # per modulation
    ber: dict                           # per modulation, floored at cfg.ber_floor
    throughput_gbps: dict               # per modulation: usable data rate (0 when BER is above the FEC limit)
    adaptive_throughput_gbps: np.ndarray    # best of the four at each instant
    adaptive_modulation: np.ndarray     # index into MODULATIONS keys of the best one (-1 = none)
    pointing_a: TurretPointing          # terminal on satellite A aiming at B
    pointing_b: TurretPointing
    budget_constants: dict = field(default_factory=dict)
    config: LaserLinkConfig | None = None


def link_budget(range_km, cfg: LaserLinkConfig) -> dict:
    """Pure link-budget maths for given ranges (no geometry): returns arrays/scalars keyed by name."""
    lam = cfg.wavelength_nm * 1e-9
    rng_m = np.asarray(range_km, dtype=float) * 1000.0
    div = cfg.divergence_urad * 1e-6
    capture = gaussian_capture_fraction(rng_m, div, cfg.rx_aperture_m)
    point = pointing_loss_linear(div, cfg.pointing_jitter_urad * 1e-6, cfg.static_pointing_error_urad * 1e-6)
    p_rx = cfg.tx_power_w * cfg.tx_optics_efficiency * cfg.rx_optics_efficiency * capture * point
    penalty = 10.0 ** (cfg.implementation_penalty_db / 10.0)
    es_n0 = cfg.detector_efficiency * p_rx / (cfg.symbol_rate_gbaud * 1e9) / (0.5 * cfg.photon_energy_j) / penalty
    out = {"fspl_db": fspl_db(rng_m, lam), "divergence_loss_db": -10.0 * np.log10(np.maximum(capture, 1e-300)),
           "pointing_loss_db": -10.0 * math.log10(point), "rx_power_w": p_rx, "rx_power_dbm": watts_to_dbm(p_rx),
           "es_n0": es_n0, "snr_db": 10.0 * np.log10(np.maximum(es_n0, 1e-300)), "latency_ms": rng_m / C_M_S * 1e3 + cfg.processing_latency_ms,
           "eb_n0": {}, "ber": {}, "throughput_gbps": {}}
    for name, k in MODULATIONS.items():
        eb = es_n0 / k
        ber = np.maximum(ber_from_eb_n0(name, eb), cfg.ber_floor)
        out["eb_n0"][name] = eb
        out["ber"][name] = ber
        out["throughput_gbps"][name] = np.where(ber <= cfg.ber_threshold, k * cfg.symbol_rate_gbaud / (1.0 + cfg.fec_overhead), 0.0)
    return out


def compute_link(t_s, track_a, track_b, cfg: LaserLinkConfig, visible, clearance_km) -> LinkResult:
    """
    Link budget between two `SatelliteTrack`s (from orbital_engine) at every time step. `visible` / `clearance_km` come from
    orbital_engine.line_of_sight, so the geometry and the physics are decoupled.
    """
    rng, rate = relative_kinematics(track_a.r_eci, track_a.v_eci, track_b.r_eci, track_b.v_eci)
    b = link_budget(rng, cfg)
    nan = np.where(visible, 1.0, np.nan)
    ber = {m: v * nan for m, v in b["ber"].items()}
    eb = {m: 10.0 * np.log10(np.maximum(v, 1e-300)) * nan for m, v in b["eb_n0"].items()}
    thr = {m: np.where(visible, v, 0.0) for m, v in b["throughput_gbps"].items()}
    names = list(MODULATIONS)
    stack = np.stack([thr[m] for m in names])
    best = stack.max(axis=0)
    best_idx = np.where(best > 0, stack.argmax(axis=0), -1)
    lam = cfg.wavelength_nm * 1e-9
    consts = {"tx_power_dbm": float(watts_to_dbm(cfg.tx_power_w)), "tx_gain_db": tx_gain_db(cfg.divergence_urad * 1e-6),
              "rx_gain_db": rx_gain_db(cfg.rx_aperture_m, lam), "pointing_loss_db": b["pointing_loss_db"],
              "optics_loss_db": -10.0 * math.log10(cfg.tx_optics_efficiency * cfg.rx_optics_efficiency),
              "carrier_frequency_thz": cfg.frequency_hz / 1e12}
    return LinkResult(
        t_s=np.asarray(t_s), visible=np.asarray(visible), clearance_km=np.asarray(clearance_km), range_km=rng,
        range_rate_km_s=rate, doppler_ghz=doppler_shift_hz(rate, cfg.frequency_hz) / 1e9, latency_ms=b["latency_ms"],
        fspl_db=b["fspl_db"], divergence_loss_db=b["divergence_loss_db"], pointing_loss_db=b["pointing_loss_db"],
        rx_power_dbm=b["rx_power_dbm"] * nan, snr_db=b["snr_db"] * nan, eb_n0_db=eb, ber=ber, throughput_gbps=thr,
        adaptive_throughput_gbps=best, adaptive_modulation=best_idx,
        pointing_a=turret_pointing(track_a.r_eci, track_a.v_eci, track_b.r_eci, track_b.v_eci),
        pointing_b=turret_pointing(track_b.r_eci, track_b.v_eci, track_a.r_eci, track_a.v_eci),
        budget_constants=consts, config=cfg)
