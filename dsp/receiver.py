"""
dsp/receiver.py -- SATELLITE RECEIVER (PAT: POINTING, ACQUISITION, TRACKING) + DEMODULATOR
==========================================================================================
(Migrated from module3_receiver.py of the Orbital-Datacentre optical link simulator; the report printer and command-line interface were
dropped, and `pre_locked` was added so a terminal that is already tracking a beam - a later frame of a stream, or a make-before-break
handover whose new path was acquired in advance - skips the spiral search.)

Takes the degraded beam from Module 2 and does what a laser-communication terminal
does before it can read a single bit: find the other satellite's beacon, steer a Fast
Steering Mirror (FSM) until the beam is centred on the sensor, then demodulate.

Sequence
--------
 1. SEARCH   - FSM paints an Archimedean spiral over the pointing-uncertainty region.
               A 4-quadrant detector (QD) sums the beacon power each control tick; the
               beacon is declared when the sum exceeds a CFAR-style threshold
               (set from the false-alarm probability) for N consecutive ticks.
 2. TRACK    - closed feedback loop at the control rate:
                   QD quadrant powers -> normalised error eps = (right-left)/sum
                   -> offset estimate = eps / K  (K = slope of the QD S-curve)
                   -> PI controller  c[k+1] = c[k] + kp*(e[k]-e[k-1]) + ki*e[k]
                   -> FSM (2nd-order actuator, saturating, mirror tilt a -> beam 2a)
                   -> plant:  e = d(t) - 2*alpha(t)    (d = angle-of-arrival disturbance)
 3. LOCK     - declared when the rolling RMS of the measured error drops below a
               threshold; after a short settle time the data frame is received.
 4. RECEIVE  - the residual pointing error of the loop (not an assumption) sets the
               fibre/detector coupling efficiency exp(-e^2/theta_c^2) for every sample.
               Front-end noise is then added, and the data path runs:
               matched filter -> preamble correlation (frame sync + complex gain)
               -> decision-directed PLL/AGC -> slicer -> de-framing -> CRC-32.

The recovered bytes are what the transmitter was given; the true transmitted bits are
used only to SCORE the result (BER), never to demodulate.
"""
from __future__ import annotations

import cmath
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import correlate, cont2discrete, fftconvolve
from scipy.stats import norm

from .channel import ReceivedBeam, ar1_process
from .transmitter import (BITS_PER_SYMBOL, HEADER_BITS, constellation, labels_to_bits,
                          make_preamble, parse_frame_bits, rrc_taps, bits_to_bytes)


# --------------------------------------------------------------------------- #
# Configuration of the receiver terminal
# --------------------------------------------------------------------------- #
@dataclass
class PATConfig:
    # --- control loop / actuator ---
    control_rate_hz: float = 50_000.0
    fsm_bandwidth_hz: float = 5_000.0        # FSM closed-position-loop bandwidth
    fsm_damping: float = 0.7
    fsm_range_mech_urad: float = 1_500.0     # +/- mechanical travel (beam moves 2x this)
    kp: float = 0.10                         # proportional gain (on error difference)
    ki: float = 0.25                         # integral gain per tick

    # --- sensor (angle-space model of a defocused beacon spot on a quadrant detector) ---
    spot_radius_urad: float = 40.0           # 1/e^2 spot radius
    detector_halfwidth_urad: float = 150.0   # QD half-width (field of view)
    nep_w_per_rthz: float = 1e-12            # noise-equivalent power per quadrant
    uncertainty_radius_urad: float = 500.0   # search region radius

    # --- acquisition logic ---
    pfa: float = 1e-6                        # per-tick false-alarm probability
    confirm_ticks: int = 3                   # consecutive detections to declare the beacon
    scan_step_fraction: float = 0.4          # spiral step per tick, x detector half-width
    scan_pitch_fraction: float = 1.5         # spiral pitch, x detector half-width
    lock_window_ticks: int = 20
    lock_threshold_urad: float = 8.0         # rms measured error that counts as locked
    lost_ticks: int = 100                    # ticks without beacon before re-searching
    settle_ms: float = 2.0                   # wait after lock before the data frame
    max_acquisition_s: float = 0.5

    # --- data path ---
    fiber_acceptance_urad: float = 12.0      # coupling falls to 1/e at this angular error
    sync_threshold: float = 6.0              # preamble peak-to-median ratio needed for frame sync
    pll_bandwidth: float = 0.05              # carrier-loop bandwidth / symbol rate (widen for high-order QAM or wide linewidth)
    agc_mu: float = 0.01


# --------------------------------------------------------------------------- #
# Quadrant detector model
# --------------------------------------------------------------------------- #
def _mass(lo: float, hi: float, w: float) -> float:
    """Fraction of a Gaussian line profile (1/e^2 radius w) lying between lo and hi."""
    k = math.sqrt(2.0) / w
    return 0.5 * (math.erf(k * hi) - math.erf(k * lo))


def quad_fractions(ex: float, ey: float, w: float, a: float):
    """
    Fraction of spot power landing on each quadrant of a square detector (half-width a)
    when the spot centre is displaced by (ex, ey).  Returns (Q1:+x+y, Q2:-x+y, Q3:-x-y, Q4:+x-y).
    The 2-D Gaussian spot is separable, so each quadrant is a product of 1-D masses.
    """
    px, nx = _mass(-ex, a - ex, w), _mass(-a - ex, -ex, w)
    py, ny = _mass(-ey, a - ey, w), _mass(-a - ey, -ey, w)
    return px * py, nx * py, nx * ny, px * ny


def qd_slope(w: float, a: float) -> float:
    """Small-signal slope K of the normalised error signal [1/urad] (numerical derivative)."""
    d = 0.01 * w

    def eps(x):
        q = quad_fractions(x, 0.0, w, a)
        return ((q[0] + q[3]) - (q[1] + q[2])) / sum(q)
    return (eps(d) - eps(-d)) / (2 * d)


# --------------------------------------------------------------------------- #
# PAT simulation
# --------------------------------------------------------------------------- #
@dataclass
class PATResult:
    closed_loop: bool
    locked: bool
    t_detect_s: float | None
    t_lock_s: float | None
    events: list = field(default_factory=list)       # (time_s, text)
    frame_error_urad: np.ndarray | None = None       # true residual pointing error during the frame (n_ticks, 2)
    rms_residual_urad: float = float("nan")
    peak_fsm_mech_urad: float = 0.0
    lost_lock_in_frame: bool = False
    search_ticks: int = 0
    qd_slope_per_urad: float = 0.0
    detect_threshold_w: float = 0.0
    beacon_snr_db: float = float("nan")
    trace: dict | None = None                        # per-tick history for plotting (not needed to demodulate)


def run_pat(beam: ReceivedBeam, cfg: PATConfig, n_frame_ticks: int, rng, closed_loop: bool = True, pre_locked: bool = False) -> PATResult:
    fc = cfg.control_rate_hz
    dt = 1.0 / fc
    aoa = beam.aoa
    p_b = beam.beacon["power_rx_w"]
    w, a_det = cfg.spot_radius_urad, cfg.detector_halfwidth_urad
    sigma_q = cfg.nep_w_per_rthz * math.sqrt(fc / 2)           # per-quadrant noise (Nyquist bandwidth)
    threshold = norm.isf(cfg.pfa) * 2 * sigma_q                # sum of 4 quadrants -> sigma = 2*sigma_q
    kslope = qd_slope(w, a_det)
    settle = max(1, int(round(cfg.settle_ms * 1e-3 * fc)))
    max_acq = int(cfg.max_acquisition_s * fc)
    n_alloc = max_acq + settle + n_frame_ticks + 8

    # angle-of-arrival disturbance seen at the focal plane: static offset + band-limited jitter
    dist = np.column_stack([
        aoa["bias_x_urad"] + ar1_process(n_alloc, aoa["sigma_urad"], aoa["corner_hz"], fc, rng, order=2),
        aoa["bias_y_urad"] + ar1_process(n_alloc, aoa["sigma_urad"], aoa["corner_hz"], fc, rng, order=2)])

    res = PATResult(closed_loop, False, None, None, qd_slope_per_urad=kslope, detect_threshold_w=threshold,
                    beacon_snr_db=10 * math.log10(max(p_b, 1e-30) / (2 * sigma_q)))
    if not closed_loop:
        res.events.append((0.0, "FSM DISABLED (open loop): beam left wherever the platform points it"))
        res.frame_error_urad = dist[:n_frame_ticks]
        res.rms_residual_urad = float(np.sqrt(np.mean(res.frame_error_urad ** 2)))
        res.trace = dict(t_s=np.arange(n_frame_ticks) * dt, err_urad=res.frame_error_urad,
                         state=np.zeros(n_frame_ticks, dtype=int), sum_w=np.zeros(n_frame_ticks),
                         k_frame=0, n_frame_ticks=n_frame_ticks, dt=dt)
        return res

    # --- FSM actuator: 2nd-order low-pass, zero-order-hold discretisation (one tick of delay) ---
    wn = 2 * math.pi * cfg.fsm_bandwidth_hz
    num, den, _ = cont2discrete(([wn ** 2], [1.0, 2 * cfg.fsm_damping * wn, wn ** 2]), dt, method="zoh")
    b = np.zeros(3)
    num = np.atleast_1d(np.squeeze(num))
    b[-len(num):] = num
    a1, a2 = den[1], den[2]
    assert abs(b[0]) < 1e-9, "FSM model must have at least one sample of delay"

    PAD = 2                                                   # index offset so k-1, k-2 always exist
    alpha = np.zeros((n_alloc + PAD, 2))                      # FSM mechanical angle [urad]
    u = np.zeros((n_alloc + PAD, 2))                          # FSM mechanical command [urad]
    err_true = np.zeros((n_alloc, 2))
    state_hist = np.zeros(n_alloc, dtype=np.int8)             # 0 = SEARCH, 1 = TRACK
    sum_hist = np.zeros(n_alloc)

    # --- spiral scan generator ---
    pitch = cfg.scan_pitch_fraction * a_det
    ds = cfg.scan_step_fraction * a_det
    b_sp = pitch / (2 * math.pi)
    s_max = cfg.uncertainty_radius_urad ** 2 / (2 * b_sp)

    def spiral(ks: int):
        phi = math.sqrt(2 * ((ks * ds) % s_max) / b_sp)
        r = b_sp * phi
        return np.array([r * math.cos(phi), r * math.sin(phi)])

    cmd_lim = 2 * cfg.fsm_range_mech_urad                     # optical command limit
    c = np.zeros(2)                                           # commanded optical correction (= 2*alpha_cmd)
    centre = np.zeros(2)
    ehat_prev = np.zeros(2)
    state, hits, lost, ks = "SEARCH", 0, 0, 0
    win = deque(maxlen=cfg.lock_window_ticks)
    k_lock = k_frame = k_end = None
    if pre_locked:
        # already tracking this beam: the mirror sits at the steady-state angle for the current disturbance, so there is nothing to search
        state, res.locked, k_lock, res.t_detect_s, res.t_lock_s, settle = "TRACK", True, 0, 0.0, 0.0, 0
        alpha[:PAD] = dist[0] / 2
        u[:PAD] = dist[0] / 2
        c = dist[0].copy()
        res.events.append((0.0, "PRE-LOCKED: terminal already tracking this beam (no spiral search needed)"))
    else:
        res.events.append((0.0, f"SEARCH: spiral scan over r<{cfg.uncertainty_radius_urad:.0f} urad, "
                                f"detector FOV +/-{a_det:.0f} urad, beacon SNR {res.beacon_snr_db:.1f} dB, "
                                f"threshold {threshold * 1e9:.2f} nW"))

    k = 0
    while k < n_alloc:
        i = k + PAD
        alpha[i] = -a1 * alpha[i - 1] - a2 * alpha[i - 2] + b[1] * u[i - 1] + b[2] * u[i - 2]
        e = dist[k] - 2 * alpha[i]                            # true spot offset on the sensor [urad]
        err_true[k] = e

        q = p_b * np.array(quad_fractions(e[0], e[1], w, a_det)) + sigma_q * rng.standard_normal(4)
        s = q.sum()
        detected = s > threshold
        state_hist[k], sum_hist[k] = (state == "TRACK"), s

        if state == "SEARCH":
            res.search_ticks += 1
            hits = hits + 1 if detected else 0
            if hits >= cfg.confirm_ticks:
                state, lost = "TRACK", 0
                res.t_detect_s = k * dt
                ehat_prev = np.array([(q[0] + q[3] - q[1] - q[2]) / s, (q[0] + q[1] - q[2] - q[3]) / s]) / kslope
                win.clear()
                res.events.append((k * dt, f"BEACON DETECTED: sum={s * 1e9:.2f} nW; spot offset "
                                           f"({e[0]:+.0f}, {e[1]:+.0f}) urad -> closing tracking loop"))
            else:
                c = centre + spiral(ks)
                ks += 1
        if state == "TRACK":
            if detected:
                lost = 0
                ehat = np.array([(q[0] + q[3] - q[1] - q[2]) / s, (q[0] + q[1] - q[2] - q[3]) / s]) / kslope
                c = c + cfg.kp * (ehat - ehat_prev) + cfg.ki * ehat      # PI feedback onto the FSM
                ehat_prev = ehat
                win.append(float(ehat @ ehat))
                if (not res.locked and len(win) == win.maxlen
                        and math.sqrt(sum(win) / len(win)) < cfg.lock_threshold_urad):
                    res.locked, k_lock, res.t_lock_s = True, k, k * dt
                    res.events.append((k * dt, f"LOCK: rms measured error {math.sqrt(sum(win) / len(win)):.2f} urad "
                                               f"(< {cfg.lock_threshold_urad:.1f}); settling {cfg.settle_ms:.1f} ms"))
            else:
                lost += 1
                if lost >= cfg.lost_ticks:                   # beam lost -> go back to searching around here
                    res.events.append((k * dt, "BEACON LOST: re-entering SEARCH"))
                    if k_frame is not None:
                        res.lost_lock_in_frame = True
                    state, res.locked, hits, ks, centre = "SEARCH", False, 0, 0, np.clip(c, -cmd_lim, cmd_lim)

        c = np.clip(c, -cmd_lim, cmd_lim)                    # FSM travel limit (also anti-windup)
        u[i] = c / 2
        res.peak_fsm_mech_urad = max(res.peak_fsm_mech_urad, float(np.max(np.abs(alpha[i]))))

        if k_frame is None:
            if res.locked and k >= k_lock + settle:
                k_frame = k + 1
            elif k + 1 >= max_acq:
                k_frame = k + 1
                res.events.append((k * dt, "ACQUISITION TIMEOUT: no lock; receiving with whatever pointing we have"))
            if k_frame is not None:
                k_end = k_frame + n_frame_ticks
                res.events.append((k_frame * dt, f"RECEIVE: data frame spans {n_frame_ticks} control tick(s)"))
        if k_end is not None and k + 1 >= k_end:
            break
        k += 1

    res.frame_error_urad = err_true[k_frame:k_frame + n_frame_ticks]
    res.rms_residual_urad = float(np.sqrt(np.mean(res.frame_error_urad ** 2)))
    n_used = k + 1
    res.trace = dict(t_s=np.arange(n_used) * dt, err_urad=err_true[:n_used], state=state_hist[:n_used].astype(int),
                     sum_w=sum_hist[:n_used], k_frame=k_frame, n_frame_ticks=n_frame_ticks, dt=dt)
    return res


# --------------------------------------------------------------------------- #
# Data path
# --------------------------------------------------------------------------- #
class CarrierTracker:
    """Decision-directed 2nd-order PLL (phase + frequency) with a slow amplitude (AGC) loop."""

    def __init__(self, modulation: str, bandwidth: float, agc_mu: float, zeta: float = 0.707):
        self.pts = constellation(modulation)
        self.pts_l = self.pts.tolist()
        th = bandwidth / (zeta + 1 / (4 * zeta))
        den = 1 + 2 * zeta * th + th * th
        self.kp, self.ki = 4 * zeta * th / den, 4 * th * th / den
        self.phase = self.freq = 0.0
        self.amp = 1.0
        self.mu = agc_mu

    def run(self, z: np.ndarray, known: np.ndarray | None = None):
        """Process symbols; `known` = true symbols (data-aided) for the preamble. Returns (y, labels)."""
        n = len(z)
        y_out = np.empty(n, dtype=complex)
        labels = np.empty(n, dtype=np.int64)
        zl = z.tolist()
        pts = self.pts_l
        for i in range(n):
            y = zl[i] * cmath.exp(-1j * self.phase) / self.amp
            j = min(range(len(pts)), key=lambda m: abs(y - pts[m]))
            d = known[i] if known is not None else pts[j]
            dd = max(abs(d) ** 2, 1e-12)
            yd = y * d.conjugate()
            err = max(-0.5, min(0.5, yd.imag / dd))               # phase error (rad, small-angle)
            self.freq += self.ki * err
            self.phase += self.kp * err + self.freq
            self.amp *= 1 + self.mu * (yd.real / dd - 1)          # amplitude error -> AGC
            self.amp = max(self.amp, 1e-6)
            y_out[i], labels[i] = y, j
        return y_out, labels


@dataclass
class ReceiverReport:
    pat: PATResult
    mean_coupling: float
    sync_ok: bool
    sync_psr: float
    payload: bytes | None
    crc_ok: bool
    status: str
    ber: float | None
    n_bit_errors: int | None
    n_bits_scored: int
    evm_db: float | None
    esn0_est_db: float | None
    rx_symbols: np.ndarray | None = None      # carrier-tracked symbols (for the constellation plot)
    coupling_ds: np.ndarray | None = None     # coupling efficiency across the frame, decimated


def receive(beam: ReceivedBeam, cfg: PATConfig, rng, closed_loop: bool = True, pre_locked: bool = False) -> ReceiverReport:
    tx = beam.tx
    mod, sps, fs = tx["modulation"], tx["sps"], beam.sample_rate
    n = len(beam.field)

    # --- PAT: runs on the control timeline, frame samples then map onto control ticks ---
    n_frame_ticks = int((n - 1) * cfg.control_rate_hz / fs) + 1
    pat = run_pat(beam, cfg, n_frame_ticks, rng, closed_loop, pre_locked and closed_loop)
    err2 = np.sum(pat.frame_error_urad ** 2, axis=1)
    tick = np.minimum((np.arange(n) * (cfg.control_rate_hz / fs)).astype(np.int64), n_frame_ticks - 1)
    eta_c = np.exp(-err2[tick] / cfg.fiber_acceptance_urad ** 2)          # coupling vs residual pointing error

    # --- front end: coupling loss on the signal, then receiver noise ---
    n0 = beam.noise_psd_w_per_hz
    noise = np.sqrt(n0 * fs / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    rx = beam.field * np.sqrt(eta_c) + noise

    # --- matched filter and preamble correlation: finds frame start, gain and phase ---
    h = rrc_taps(tx["rolloff"], sps, tx["rrc_span"])
    mf = fftconvolve(rx, h, mode="full")
    pre = make_preamble(tx["n_preamble"])
    pre_up = np.zeros((len(pre) - 1) * sps + 1, dtype=complex)
    pre_up[::sps] = pre
    corr = correlate(mf, pre_up, mode="valid", method="fft")
    mag = np.abs(corr)
    n0_idx = int(np.argmax(mag))
    psr = float(mag[n0_idx] / max(np.median(mag), 1e-30))
    report = ReceiverReport(pat, float(np.mean(eta_c)), psr >= cfg.sync_threshold, psr, None, False, "",
                            None, None, 0, None, None,
                            coupling_ds=eta_c[np.linspace(0, n - 1, min(n, 2000)).astype(int)])
    if not report.sync_ok:
        report.status = f"NO FRAME SYNC (preamble peak/median = {psr:.1f} < {cfg.sync_threshold:.1f}): beam not captured"
        return report
    g0 = corr[n0_idx] / np.sum(np.abs(pre) ** 2)                          # complex gain of the link

    # --- symbol sampling + carrier tracking ---
    bps = BITS_PER_SYMBOL[mod]
    n_avail = (len(mf) - 1 - n0_idx) // sps + 1
    z = mf[n0_idx + np.arange(n_avail) * sps] / g0
    trk = CarrierTracker(mod, cfg.pll_bandwidth, cfg.agc_mu)
    npre = tx["n_preamble"]
    trk.run(z[:npre], known=pre)                                           # data-aided on the preamble
    n_hdr = -(-HEADER_BITS // bps)
    y_hdr, l_hdr = trk.run(z[npre:npre + n_hdr])
    declared = int.from_bytes(bits_to_bytes(labels_to_bits(l_hdr, mod)[:HEADER_BITS]), "big")
    n_data_syms = min(-(-(HEADER_BITS + 8 * declared + 32) // bps), n_avail - npre)   # symbols the header says we need
    y_rest, l_rest = trk.run(z[npre + n_hdr:npre + n_data_syms])
    y = np.concatenate([y_hdr, y_rest])
    labels = np.concatenate([l_hdr, l_rest])

    # --- de-framing ---
    bits = labels_to_bits(labels, mod)
    report.payload, report.crc_ok, report.status = parse_frame_bits(bits)

    # --- scoring against the transmitted truth (never used above) ---
    ref = beam.reference_bits
    m = min(len(bits), len(ref))
    errs = int(np.sum(bits[:m] != ref[:m])) + (len(ref) - m)
    report.n_bit_errors, report.n_bits_scored, report.ber = errs, len(ref), errs / len(ref)
    report.rx_symbols = y
    dec = constellation(mod)[labels]
    evm2 = float(np.mean(np.abs(y - dec) ** 2) / np.mean(np.abs(dec) ** 2))
    report.evm_db, report.esn0_est_db = 10 * math.log10(max(evm2, 1e-12)), -10 * math.log10(max(evm2, 1e-12))
    return report
