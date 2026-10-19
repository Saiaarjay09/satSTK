"""
dsp/channel.py -- ENVIRONMENT SIMULATION (ONE VACUUM LASER HOP)
===============================================================
(Migrated from module2_channel.py of the Orbital-Datacentre optical link simulator; the dashboard and command-line interface were dropped.
satSTK's optical_dsp_bridge.py builds one LinkConfig per hop from the live orbital state and chains the hops.)

Takes the optical field from Module 1 plus user-defined orbital / terminal
variables and models what happens to the beam between two satellites.

Physics modelled
----------------
  * Orbital geometry       : altitude + range (or central-angle separation), Earth-limb
                             clearance check, orbital speed, point-ahead angle.
  * Gaussian beam optics   : waist w0 -> Rayleigh range, far-field divergence, spot radius
                             at the receiver, telescope truncation, aperture capture.
  * Spatial (pointing)
    jitter                 : per-axis Gaussian pointing error (static bias + band-limited
                             random jitter) -> time-varying fade exp(-2 theta^2 / theta_div^2).
  * Laser phase noise      : Wiener process from the laser linewidth.
  * Receiver noise         : quantum-limited noise N0 = h*nu*NF (NF = excess noise factor).
  * Vacuum                 : no turbulence/scintillation/absorption, so there is no
                             atmospheric term.  Free-space loss is purely geometric.

Outputs
-------
  * Metrics (data loss %, BER, latency) and the link budget.
  * A ReceivedBeam for the receiver: the *noise-free* degraded field and the receiver-side
    disturbance statistics. The receiver applies its own front-end noise AFTER its
    sensor/fibre coupling, because that is where the noise physically enters.

How BER / data loss are measured here
-------------------------------------
This module uses an ORACLE reference receiver (perfect timing, perfect knowledge of the
instantaneous channel gain and carrier phase). The BER it reports is therefore the
penalty from path loss + pointing fades + quantum noise + modulator distortion only.
The penalty from imperfect real-world carrier recovery / PAT residuals is measured by
the receiver's real demodulator.

Data loss % is defined as the percentage of payload BYTES containing at least one bit
error (i.e. bytes that are not delivered intact, pre-FEC).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve, lfilter
from scipy.special import erfc

from .transmitter import (C_LIGHT, PLANCK, OpticalSignal, rrc_taps, save_container,
                          load_container, symbols_to_bits)

EARTH_RADIUS_M = 6_371_000.0
MU_EARTH = 3.986004418e14            # GM of Earth [m^3/s^2]
ATMOSPHERE_TOP_M = 100_000.0         # LOS must clear this altitude to stay in vacuum


# --------------------------------------------------------------------------- #
# User-defined link variables
# --------------------------------------------------------------------------- #
@dataclass
class LinkConfig:
    # --- orbital geometry (give range OR separation; both None -> 1000 km) ---
    range_km: float | None = None            # inter-satellite distance
    separation_deg: float | None = None      # central angle between the satellites
    altitude_km: float = 550.0               # orbit altitude (both satellites)
    transverse_velocity_mps: float = 0.0     # relative velocity normal to the LOS (point-ahead)
    point_ahead_compensated: bool = True
    point_ahead_residual_urad: float = 0.5   # error left after point-ahead compensation

    # --- terminals ---
    tx_aperture_m: float = 0.10              # transmit telescope diameter
    tx_fill: float = 0.89                    # Gaussian waist / aperture radius
    rx_aperture_m: float = 0.10              # receive telescope diameter
    tx_optics_eff: float = 0.80              # linear transmission of Tx optics
    rx_optics_eff: float = 0.80              # linear transmission of Rx optics

    # --- spatial jitter (transmit pointing: affects received power) ---
    tx_bias_urad_x: float = 0.0
    tx_bias_urad_y: float = 0.0
    tx_jitter_urad: float = 1.5              # rms per axis
    tx_jitter_corner_hz: float = 500.0       # jitter bandwidth

    # --- receive angle-of-arrival disturbance (corrected by Module 3's FSM) ---
    aoa_bias_urad_x: float = 250.0
    aoa_bias_urad_y: float = -180.0
    aoa_jitter_urad: float = 15.0            # rms per axis
    aoa_jitter_corner_hz: float = 300.0

    # --- laser / receiver ---
    laser_linewidth_hz: float = 100e3
    noise_figure_db: float = 3.0             # excess noise over the quantum limit

    # --- acquisition beacon (wide, low-power beam from the transmitting terminal) ---
    beacon_power_w: float = 1.0
    beacon_divergence_urad: float = 200.0    # half-angle (1/e^2 radius)

    # --- timing / misc ---
    processing_latency_ms: float = 0.0       # modem + electronics delay
    lead_in_symbols: int = 40                # noise-only symbols before/after the frame
    seed: int | None = None


@dataclass
class OrbitGeometry:
    range_m: float
    separation_deg: float
    los_clearance_m: float
    orbital_speed_mps: float
    orbital_period_s: float
    point_ahead_urad: float
    los_ok: bool


@dataclass
class BeamBudget:
    waist_m: float
    rayleigh_range_m: float
    divergence_half_urad: float      # far-field half-angle, 1/e^2 intensity radius
    spot_radius_m: float             # beam radius at the receiver
    eta_tx_trunc: float
    eta_geometric: float             # fraction of beam power inside the Rx aperture (on-axis)
    eta_fixed: float                 # product of all static efficiencies (on-axis)
    on_axis_loss_db: float
    expected_pointing_eff: float     # E[exp(-2 theta^2/theta_div^2)] over the jitter statistics


@dataclass
class ChannelMetrics:
    # headline metrics
    data_loss_pct: float
    ber: float
    n_bit_errors: int
    n_bits: int
    latency_ms: float
    # latency breakdown
    propagation_ms: float
    serialization_ms: float
    processing_ms: float
    round_trip_ms: float
    # supporting numbers
    ber_theory: float
    esn0_db: float
    rx_power_dbm: float
    path_loss_db: float
    pointing_loss_db: float
    power_loss_pct: float
    n_bad_bytes: int
    n_payload_bytes: int
    seed: int


@dataclass
class ReceivedBeam:
    """Product handed to Module 3."""
    field: np.ndarray             # noise-free degraded field (includes lead-in/trail)
    reference_bits: np.ndarray    # transmitted frame bits (scoring only, never used to demodulate)
    tx: dict                      # transmitter metadata (modulation, rates, ...)
    noise_psd_w_per_hz: float     # receiver front-end noise PSD N0 = h*nu*NF
    aoa: dict                     # angle-of-arrival disturbance statistics
    beacon: dict                  # received beacon power etc.
    link: dict                    # geometry / budget (informational)
    metrics: dict                 # Module 2 metrics (informational)
    diagnostics: dict | None = None   # plot data (fade history, equalised symbols); not saved to .npz

    @property
    def sample_rate(self) -> float:
        return self.tx["symbol_rate"] * self.tx["sps"]

    def save(self, path):
        meta = {k: getattr(self, k) for k in ("tx", "noise_psd_w_per_hz", "aoa", "beacon", "link", "metrics")}
        return save_container(path, {"field": self.field, "reference_bits": self.reference_bits}, meta)

    @classmethod
    def load(cls, path) -> "ReceivedBeam":
        arrays, meta = load_container(path)
        return cls(field=arrays["field"], reference_bits=arrays["reference_bits"], **meta)


# --------------------------------------------------------------------------- #
# Geometry and beam optics
# --------------------------------------------------------------------------- #
def resolve_geometry(cfg: LinkConfig) -> OrbitGeometry:
    r_orb = EARTH_RADIUS_M + cfg.altitude_km * 1e3
    if cfg.range_km is not None and cfg.separation_deg is not None:
        raise ValueError("specify either range_km or separation_deg, not both")
    if cfg.separation_deg is not None:
        sep = np.radians(cfg.separation_deg)
        rng_m = 2 * r_orb * np.sin(sep / 2)
    else:
        rng_m = (cfg.range_km if cfg.range_km is not None else 1000.0) * 1e3
        if rng_m >= 2 * r_orb:
            raise ValueError("range exceeds the diameter of the orbit")
        sep = 2 * np.arcsin(rng_m / (2 * r_orb))
    clearance = r_orb * np.cos(sep / 2) - EARTH_RADIUS_M      # closest approach of the LOS to the surface
    v = np.sqrt(MU_EARTH / r_orb)
    return OrbitGeometry(range_m=rng_m, separation_deg=float(np.degrees(sep)), los_clearance_m=clearance,
                         orbital_speed_mps=v, orbital_period_s=2 * np.pi * r_orb / v,
                         point_ahead_urad=2 * cfg.transverse_velocity_mps / C_LIGHT * 1e6,
                         los_ok=clearance >= ATMOSPHERE_TOP_M)


def pointing_efficiency_expectation(bias_x, bias_y, sigma, theta_div) -> float:
    """
    Closed-form E[exp(-2*theta^2/theta_div^2)] for a 2-D Gaussian pointing error with
    mean (bias_x, bias_y) and per-axis std `sigma` (all in radians).
    """
    a = 2.0 / theta_div ** 2
    d = 1.0 + 2.0 * a * sigma ** 2
    return float(np.exp(-a * (bias_x ** 2 + bias_y ** 2) / d) / d)


def point_ahead_residual_urad(cfg: LinkConfig, geo: OrbitGeometry) -> float:
    """Static Tx pointing error left over by the point-ahead mechanism."""
    return cfg.point_ahead_residual_urad if cfg.point_ahead_compensated else geo.point_ahead_urad


def beam_budget(cfg: LinkConfig, geo: OrbitGeometry, wavelength: float) -> BeamBudget:
    a_tx = cfg.tx_aperture_m / 2
    a_rx = cfg.rx_aperture_m / 2
    w0 = cfg.tx_fill * a_tx                                    # Gaussian beam waist
    z_r = np.pi * w0 ** 2 / wavelength                         # Rayleigh range
    z = geo.range_m
    w_rx = w0 * np.sqrt(1 + (z / z_r) ** 2)                    # beam radius at the receiver
    theta_div = w_rx / z                                       # effective half-divergence
    eta_trunc = 1 - np.exp(-2 * a_tx ** 2 / w0 ** 2)           # power surviving the Tx aperture
    eta_geo = 1 - np.exp(-2 * a_rx ** 2 / w_rx ** 2)           # power captured by the Rx aperture
    eta_fixed = cfg.tx_optics_eff * eta_trunc * eta_geo * cfg.rx_optics_eff

    bx = (cfg.tx_bias_urad_x + point_ahead_residual_urad(cfg, geo)) * 1e-6
    by = cfg.tx_bias_urad_y * 1e-6
    e_point = pointing_efficiency_expectation(bx, by, cfg.tx_jitter_urad * 1e-6, theta_div)
    return BeamBudget(waist_m=w0, rayleigh_range_m=z_r, divergence_half_urad=theta_div * 1e6,
                      spot_radius_m=w_rx, eta_tx_trunc=eta_trunc, eta_geometric=eta_geo,
                      eta_fixed=eta_fixed, on_axis_loss_db=-10 * np.log10(eta_fixed),
                      expected_pointing_eff=e_point)


# --------------------------------------------------------------------------- #
# Stochastic processes
# --------------------------------------------------------------------------- #
def ar1_process(n: int, sigma: float, corner_hz: float, fs: float, rng, order: int = 1) -> np.ndarray:
    """
    Stationary band-limited Gaussian process with standard deviation `sigma`, sampled at `fs`.
    order=1 : first-order Gauss-Markov (AR(1)), -3 dB corner at `corner_hz`, 1/f^2 roll-off.
              Used for Tx pointing jitter.
    order=2 : two cascaded identical poles (1/f^4 roll-off), closer to real platform
              vibration spectra.  Used for receive angle-of-arrival jitter.
    """
    if sigma == 0 or n == 0:
        return np.zeros(n)
    a = np.exp(-2 * np.pi * corner_hz / fs)
    if order == 1:
        x0 = rng.standard_normal() * sigma                      # start in the stationary distribution
        y, _ = lfilter([sigma * np.sqrt(1 - a * a)], [1.0, -a], rng.standard_normal(n), zi=[a * x0])
        return y
    if order == 2:
        burn = int(min(12 / (1 - a), 2_000_000))                # let the filter reach steady state
        g = sigma * np.sqrt((1 - a * a) ** 3 / (1 + a * a))     # gain giving output variance sigma^2
        y = lfilter([g], np.polymul([1.0, -a], [1.0, -a]), rng.standard_normal(n + burn))
        return y[burn:]
    raise ValueError("order must be 1 or 2")


def theoretical_ber(mod: str, esn0) -> np.ndarray:
    """Gray-coded AWGN bit-error probability for coherent detection, vs Es/N0 (linear)."""
    e = np.maximum(np.asarray(esn0, dtype=float), 1e-12)
    if mod == "BPSK":
        return 0.5 * erfc(np.sqrt(e))
    if mod == "QPSK":
        return 0.5 * erfc(np.sqrt(e / 2))
    if mod == "8PSK":
        return (2 / 3) * 0.5 * erfc(np.sqrt(e) * np.sin(np.pi / 8))
    if mod == "16QAM":
        return (3 / 8) * erfc(np.sqrt(e / 10))
    raise ValueError(mod)


# --------------------------------------------------------------------------- #
# The channel
# --------------------------------------------------------------------------- #
def propagate(sig: OpticalSignal, cfg: LinkConfig, rng: np.random.Generator, seed: int = 0):
    """Send `sig` across the link.  Returns (ReceivedBeam, ChannelMetrics, BeamBudget, OrbitGeometry)."""
    geo = resolve_geometry(cfg)
    bud = beam_budget(cfg, geo, sig.wavelength_m)
    fs, sps = sig.sample_rate, sig.sps
    n_lead = cfg.lead_in_symbols * sps
    nu = C_LIGHT / sig.wavelength_m
    n0 = PLANCK * nu * 10 ** (cfg.noise_figure_db / 10)          # quantum-limited noise PSD [W/Hz]

    # --- 1) place the frame inside a noise-only window (receiver must find it) ---
    tx_field = np.concatenate([np.zeros(n_lead), sig.field, np.zeros(n_lead)])
    n = len(tx_field)

    # --- 2) transmit pointing error -> time-varying capture fade ---
    th_x = (cfg.tx_bias_urad_x + point_ahead_residual_urad(cfg, geo)) * 1e-6 + ar1_process(n, cfg.tx_jitter_urad * 1e-6, cfg.tx_jitter_corner_hz, fs, rng)
    th_y = cfg.tx_bias_urad_y * 1e-6 + ar1_process(n, cfg.tx_jitter_urad * 1e-6, cfg.tx_jitter_corner_hz, fs, rng)
    theta_div = bud.divergence_half_urad * 1e-6
    gain_point = np.exp(-2 * (th_x ** 2 + th_y ** 2) / theta_div ** 2)        # intensity fade (<=1)

    # --- 3) laser phase noise (Wiener) + unknown carrier phase ---
    var_step = 2 * np.pi * cfg.laser_linewidth_hz / fs
    phase = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.standard_normal(n) * np.sqrt(var_step))

    # --- 4) complex channel gain and received (noise-free) field ---
    g = np.sqrt(bud.eta_fixed * gain_point) * np.exp(1j * phase)
    rx_clean = tx_field * g
    frame = slice(n_lead, n_lead + len(sig.field))

    # --- 5) front-end noise and oracle reference receiver (for BER / data loss) ---
    noise = np.sqrt(n0 * fs / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    h = rrc_taps(sig.rolloff, sps, sig.rrc_span)
    mf = fftconvolve(rx_clean + noise, h, mode="full")
    k = np.arange(sig.n_symbols)
    peak_idx = n_lead + k * sps + (len(h) - 1)                                # matched-filter peak
    centre_idx = n_lead + k * sps + (len(h) - 1) // 2                         # pulse centre (for channel gain)
    z = mf[peak_idx] / (np.sqrt(sig.tx_power_w * sps) * g[centre_idx])        # oracle gain+phase equalisation
    bits_rx = symbols_to_bits(z[sig.n_preamble:], sig.modulation)[: len(sig.frame_bits)]

    bit_err = bits_rx != sig.frame_bits
    n_err = int(bit_err.sum())
    ber = n_err / len(sig.frame_bits)
    payload_err = bit_err[32:32 + 8 * sig.payload_len].reshape(-1, 8).any(axis=1)
    n_bad = int(payload_err.sum())

    # --- 6) metrics ---
    p_rx = float(np.mean(np.abs(rx_clean[frame]) ** 2))
    esn0_sym = bud.eta_fixed * gain_point[centre_idx] * sig.tx_power_w / (n0 * sig.symbol_rate)
    ber_theory = float(np.mean(theoretical_ber(sig.modulation, esn0_sym)))
    prop_ms = geo.range_m / C_LIGHT * 1e3
    ser_ms = sig.duration_s * 1e3
    metrics = ChannelMetrics(
        data_loss_pct=100.0 * n_bad / sig.payload_len, ber=ber, n_bit_errors=n_err,
        n_bits=len(sig.frame_bits), latency_ms=prop_ms + ser_ms + cfg.processing_latency_ms,
        propagation_ms=prop_ms, serialization_ms=ser_ms, processing_ms=cfg.processing_latency_ms,
        round_trip_ms=2 * (prop_ms + cfg.processing_latency_ms) + ser_ms,
        ber_theory=ber_theory, esn0_db=10 * np.log10(np.mean(esn0_sym)),
        rx_power_dbm=10 * np.log10(p_rx * 1e3), path_loss_db=10 * np.log10(sig.tx_power_w / p_rx),
        pointing_loss_db=-10 * np.log10(np.mean(gain_point[frame])),
        power_loss_pct=100.0 * (1 - p_rx / sig.tx_power_w), n_bad_bytes=n_bad,
        n_payload_bytes=sig.payload_len, seed=seed)

    # --- 7) beacon (wide beam => insensitive to the small Tx pointing error) ---
    w_b = cfg.beacon_divergence_urad * 1e-6 * geo.range_m
    a_rx = cfg.rx_aperture_m / 2
    p_beacon = cfg.beacon_power_w * cfg.tx_optics_eff * cfg.rx_optics_eff * (1 - np.exp(-2 * a_rx ** 2 / w_b ** 2))

    # plot data for the UI: pointing fade across the frame and oracle-equalised symbols (decimated)
    fi = np.linspace(n_lead, n_lead + len(sig.field) - 1, min(len(sig.field), 2000)).astype(int)
    zi = np.linspace(sig.n_preamble, len(z) - 1, min(len(z) - sig.n_preamble, 4000)).astype(int)
    diagnostics = dict(
        t_us=(fi - n_lead) / fs * 1e6, fade_db=10 * np.log10(np.maximum(gain_point[fi], 1e-30)),
        pointing_offset_urad=np.hypot(th_x[fi], th_y[fi]) * 1e6, oracle_symbols=z[zi],
        theta_div_urad=bud.divergence_half_urad)

    beam = ReceivedBeam(
        field=rx_clean, reference_bits=sig.frame_bits, tx=sig.meta(), noise_psd_w_per_hz=n0,
        aoa=dict(bias_x_urad=cfg.aoa_bias_urad_x, bias_y_urad=cfg.aoa_bias_urad_y,
                 sigma_urad=cfg.aoa_jitter_urad, corner_hz=cfg.aoa_jitter_corner_hz),
        beacon=dict(power_rx_w=float(p_beacon), spot_radius_m=float(w_b)),
        link=dict(range_m=geo.range_m, spot_radius_m=bud.spot_radius_m, lead_in_samples=n_lead),
        metrics=dataclasses.asdict(metrics), diagnostics=diagnostics)
    return beam, metrics, bud, geo


def monte_carlo(sig: OpticalSignal, cfg: LinkConfig, trials: int, seed: int) -> dict:
    """Repeat the channel with independent jitter / noise draws to get fade statistics."""
    bers, losses = [], []
    for i in range(trials):
        _, m, _, _ = propagate(sig, cfg, np.random.default_rng([seed, i + 1]), seed)
        bers.append(m.ber)
        losses.append(m.data_loss_pct)
    bers, losses = np.array(bers), np.array(losses)
    return dict(trials=trials, ber_mean=float(bers.mean()), ber_p95=float(np.percentile(bers, 95)),
                loss_mean=float(losses.mean()), p_any_loss=float(np.mean(losses > 0)),
                p_frame_error_free=float(np.mean(bers == 0)),
                bers=bers.tolist(), losses=losses.tolist())
