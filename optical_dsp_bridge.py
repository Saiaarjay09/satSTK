"""
optical_dsp_bridge.py - send a real payload through the live orbital simulation, symbol by symbol.

satSTK's orbital engine knows WHERE everything is: the ground station, the satellite, the GEO relays, the range of every hop, the atmosphere over the
station, the events happening, the pointing jitter. The `dsp` package knows HOW to turn bytes into light and back: IQ Mach-Zehnder modulator, one
vacuum laser hop, spiral-search + Fast-Steering-Mirror tracking, carrier recovery and CRC-32. This module connects them.

    sim = run_simulation(SimulationConfig())
    trx = OrbitalOpticalTransceiver(sim)
    out = trx.send(b"any bytes", index=trx.index_at(3600.0), modulation="Auto-Adaptive", symbol_rate_gbaud=10)
    print(out.route_label, out.modulation, out.crc_ok, out.ber)

What happens in `send`, for the route (direct LEO, 2-hop or 3-hop GEO relay) in force at the chosen time step:

  1. the payload is framed (length + CRC-32), mapped to BPSK / QPSK / 8PSK / 16QAM, pulse-shaped and modulated (dsp.transmitter);
  2. HOP 1 leaves the ground station: the beam crosses the atmosphere (haze, cloud, turbulence, any events at that moment) and the hop's vacuum
     channel with its REAL range and pointing jitter (dsp.channel);
  3. at every GEO relay the beam is received (adding the relay's receiver noise), amplified by the EDFA back to its output power, and sent on over
     the next hop with that hop's own range and terminals - noise accumulates exactly as in a bent-pipe chain;
  4. the satellite's terminal runs PAT (spiral beacon search, quadrant detector, PI-controlled Fast Steering Mirror), then the matched filter,
     preamble sync, carrier PLL, slicer and CRC-32 check (dsp.receiver).

Everything physical (range, aperture, power, divergence, jitter, atmosphere, noise level) comes from the simulation's own configuration, so a
change in the Conditions panel changes what the DSP sees. The link budget is cross-checked against satSTK's analytic budget in the tests.

Honest limits: Doppler (up to +-4 GHz in the simulation) is assumed removed by the receiver's local-oscillator tracking, so the DSP does not
see it; there is no forward error correction here (the simulation's throughput numbers assume a 7 % FEC threshold; this chain must get every bit
right for the CRC to pass); relays are idealised amplifiers (no gain ripple or nonlinearity); the PAT model assumes the relays stay locked and
only the satellite terminal acquires; scintillation is one random fade per frame (a frame lasts microseconds, much shorter than the ~ms
turbulence time); the symbol-level run uses a few thousand to a few hundred thousand symbols, never a whole second of a 250 Gbaud link.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

import atmosphere as atm
import orbital_engine as oe
from dsp import channel as dch
from dsp import payload as pl
from dsp import receiver as drx
from dsp import transmitter as dtx
from laser_link_engine import short_name

AUTO = "Auto-Adaptive"
MODULATIONS = list(dtx.BITS_PER_SYMBOL)                 # BPSK, QPSK, 8PSK, 16QAM
ORDER_BY_RATE = ["16QAM", "8PSK", "QPSK", "BPSK"]       # tried in this order by the adaptive mode
TX_FILL = 0.89                                          # Gaussian waist / telescope radius used by dsp.channel
ETA_TRUNC = 1.0 - math.exp(-2.0 / TX_FILL ** 2)         # power surviving the transmit aperture at that fill
MAX_PAYLOAD_BYTES = 131_072                             # the demodulator's carrier loop runs symbol by symbol in Python
ADAPT_TARGET_ERRORS = 0.02                              # adaptive mode wants the expected number of bit errors in a frame below this
ADAPT_MARGIN_DB = 3.0                                   # for carrier-loop / pointing imperfections the analytic prediction ignores


class RouteUnavailable(RuntimeError):
    """The chosen route has no line of sight (or no route exists) at the chosen time."""


@dataclass
class HopSpec:
    """One optical hop with everything the DSP channel needs, taken from the simulation state."""
    src: str
    dst: str
    kind: str                       # "ground" (leaves the station, crosses the atmosphere), "relay" (GEO to GEO), "last" (GEO to the satellite) or "direct"
    range_km: float
    link: dch.LinkConfig
    tx_power_w: float
    extra_loss_db: float = 0.0      # clear air + cloud/fog/rain + events (turbulence is the random fade below)
    scint_index: float = 0.0
    elevation_deg: float | None = None
    outage_fraction: float = 0.0    # share of the time step in which an event blocks the beam entirely (applied to the first hop)
    amplified_after: bool = False   # a GEO relay receives this hop and re-amplifies it


@dataclass
class TransmissionResult:
    route_label: str
    hops: list
    modulation: str
    adaptive_table: dict | None             # {modulation: predicted bit errors in the frame} when Auto-Adaptive chose
    adaptive_reason: str
    time_s: float
    index: int
    payload: bytes
    recovered: bytes | None
    crc_ok: bool
    status: str
    ber: float | None                       # receiver BER against the transmitted bits
    ber_oracle: float | None                # BER of dsp.channel's oracle receiver; only meaningful for a single hop (None for relays: it cannot undo the earlier hops' carrier phase)
    n_bit_errors: int | None
    n_bits: int
    bytes_wrong: int
    bytes_wrong_pct: float
    frame_loss_pct: float                   # 100 when the frame (CRC) was lost, else 0
    sync_ok: bool
    evm_db: float | None
    esn0_est_db: float | None
    esn0_predicted_db: float                # analytic end-to-end Es/N0 (cascaded hops)
    rx_power_dbm: float                     # received power at the final terminal (last hop)
    path_loss_db: float                     # of the last hop (the final terminal's own link)
    blocked_by_event: bool
    symbol_rate_gbaud: float
    n_symbols: int
    latency_ms: float                       # propagation + serialisation + processing
    propagation_ms: float
    serialization_ms: float
    processing_ms: float
    pat_ms: float | None                    # time from start of search to lock (None: never locked / not applicable)
    pat_events: list
    pat_trace: dict | None
    rx_symbols: np.ndarray | None
    closed_loop: bool
    pre_locked: bool
    blocks: object | None = None            # dsp.payload.BlockReport for the recovered data
    seed: int = 0
    hop_summaries: list = field(default_factory=list)


@dataclass
class FrameRecord:
    k: int
    index: int
    time_s: float
    route_label: str
    modulation: str
    ok: bool
    ber: float | None
    n_bytes: int
    latency_ms: float
    pat_ms: float | None
    handover: bool
    hitless: bool | None
    pre_locked: bool
    reason: str


@dataclass
class StreamResult:
    frames: list
    payload: bytes
    recovered: bytes
    frames_ok: int
    frames_lost: int
    frame_loss_pct: float
    bytes_ok: int
    handovers: int
    all_handovers_hitless: bool
    blocks: object | None
    stream_crc_ok: bool
    cancelled: bool = False
    last: object | None = None                  # the TransmissionResult of the last frame sent (for the constellation / FSM plots)


def _elevation(gs: oe.GroundStation, target_ecef: np.ndarray) -> float:
    d = np.asarray(target_ecef) - gs.ecef()
    return float(np.degrees(np.arcsin(np.clip(d @ gs.enu_matrix()[2] / np.linalg.norm(d), -1, 1))))


def divergence_to_aperture(divergence_full_urad: float, wavelength_m: float) -> float:
    """Transmit telescope diameter that gives a Gaussian beam of this full-angle divergence (far field: theta_half = lambda / (pi w0), w0 = fill * a)."""
    theta_half = divergence_full_urad * 1e-6 / 2.0
    w0 = wavelength_m / (math.pi * theta_half)
    return 2.0 * w0 / TX_FILL


def make_link_config(range_km: float, divergence_full_urad: float, jitter_urad: float, rx_aperture_m: float, tx_optics_eff: float, rx_optics_eff: float,
                     wavelength_m: float, noise_figure_db: float, beacon_power_w: float, beacon_divergence_urad: float, laser_linewidth_hz: float,
                     static_bias_urad: float = 0.0, processing_ms: float = 0.0) -> dch.LinkConfig:
    """
    Translate one simulated hop into the DSP channel's parameters, keeping the total power budget identical to satSTK's:
    the beam width comes from the divergence, and the transmit optics efficiency is raised by 1/eta_trunc to cancel the Gaussian truncation loss
    that the DSP channel adds (so path loss matches the analytic budget; the cross-check test enforces it).
    """
    return dch.LinkConfig(
        range_km=range_km, altitude_km=max(550.0, range_km),                      # altitude only feeds informational geometry fields
        transverse_velocity_mps=0.0, point_ahead_compensated=True, point_ahead_residual_urad=0.0,
        tx_aperture_m=divergence_to_aperture(divergence_full_urad, wavelength_m), tx_fill=TX_FILL, rx_aperture_m=rx_aperture_m,
        tx_optics_eff=min(1.0, tx_optics_eff / ETA_TRUNC), rx_optics_eff=rx_optics_eff,
        tx_bias_urad_x=static_bias_urad, tx_bias_urad_y=0.0, tx_jitter_urad=jitter_urad,
        laser_linewidth_hz=laser_linewidth_hz, noise_figure_db=noise_figure_db,
        beacon_power_w=beacon_power_w, beacon_divergence_urad=beacon_divergence_urad, processing_latency_ms=processing_ms)


class OrbitalOpticalTransceiver:
    """Sends payloads through the route (and orbital geometry) of a finished satSTK simulation."""

    def __init__(self, result, pat: drx.PATConfig | None = None):
        self.sim = result
        self.cfg = result.config
        self.pat_cfg = pat or drx.PATConfig()

    # ------------------------------------------------------------------ the simulation state at one instant
    def index_at(self, seconds: float) -> int:
        return self.sim.index_at(seconds)

    def candidate_at(self, index: int, forced: int | None = None) -> int:
        """Candidate route in force at this step (or the forced one); raises RouteUnavailable if it has no line of sight right then."""
        rt = self.sim.route
        cand = int(rt.chosen[index]) if forced is None else int(forced)
        if cand < 0:
            raise RouteUnavailable("no route to the satellite at this time (below the mask, and no GEO relay can see it)")
        if not rt.available[cand, index]:
            raise RouteUnavailable(f"route '{rt.labels[cand]}' has no line of sight at this time")
        return cand

    def hops(self, index: int, candidate: int | None = None, tx_power_w: float | None = None, beacon_power_w: float = 1.0) -> list:
        """The chain of hops for the route at this time step, with ranges, apertures, jitter, atmosphere and events all from the simulation."""
        sim, cfg = self.sim, self.cfg
        rt = sim.route
        cand = self.candidate_at(index, candidate)
        lam = cfg.link.wavelength_nm * 1e-9
        relay = cfg.relay
        gs = cfg.ground
        tgt = sim.target
        names = rt.labels[cand].split(" -> ")                        # e.g. ['GS-INDIA', 'GEO-IND', 'NODE-ALPHA']
        relays = [int(v) for v in (rt.candidate_via1[cand], rt.candidate_via2[cand]) if v >= 0]
        pos = [sim.station_eci[index]] + [sim.relay_tracks[k].r_eci[index] for k in relays] + [sim.tracks[tgt].r_eci[index]]
        ecef = [gs.ecef()] + [sim.relay_tracks[k].r_ecef[index] for k in relays] + [sim.tracks[tgt].r_ecef[index]]
        # match satSTK's Es/N0 = eta_det x P / (Rs x (h nu / 2) x penalty): the DSP channel's noise is h nu x NF, so NF = penalty / (2 eta_det)
        nf_db = max(0.0, cfg.link.implementation_penalty_db - 10.0 * math.log10(2.0) - 10.0 * math.log10(cfg.link.detector_efficiency))
        p_ground = cfg.link.tx_power_w if tx_power_w is None else tx_power_w
        fx = sim.effects
        outage = float(fx.outage_fraction[index])
        jit_factor = float(fx.jitter_factor[index])
        hops = []
        for h in range(len(pos) - 1):
            rng_km = float(np.linalg.norm(pos[h + 1] - pos[h]))
            first, last = h == 0, h == len(pos) - 2
            direct = first and last
            to_geo_or_from_geo = bool(relays)
            if direct:
                div, jit, rx_ap, kind = cfg.link.divergence_urad, cfg.link.pointing_jitter_urad, cfg.link.rx_aperture_m, "direct"
            else:
                div, jit, rx_ap = relay.divergence_urad, relay.jitter_urad, relay.aperture_m
                kind = "ground" if first else ("last" if last else "relay")
            extra, si, el = 0.0, 0.0, None
            if first:
                el = _elevation(gs, ecef[1])
                ch = atm.channel(el, gs.alt_m, cfg.atmosphere, cfg.link.wavelength_nm)
                extra = float(ch["clear_air_loss_db"] + ch["weather_loss_db"] + fx.loss_db[index])
                si = float(ch["scintillation_index"])
                div = math.hypot(div, float(ch["beam_spread_urad"]))
                jit *= jit_factor
            link = make_link_config(
                rng_km, div, jit, rx_ap, cfg.link.tx_optics_efficiency, cfg.link.rx_optics_efficiency, lam, nf_db, beacon_power_w,
                40.0 if to_geo_or_from_geo else 200.0, 100e3 if first else 0.0, cfg.link.static_pointing_error_urad if direct else 0.0,
                relay.processing_latency_ms if (not first) else 0.0)
            hops.append(HopSpec(names[h], names[h + 1], kind, rng_km, link, p_ground if first else relay.tx_power_w, extra, si, el,
                                outage if first else 0.0, amplified_after=not last))
        return hops

    # ------------------------------------------------------------------ analytic prediction (used by the adaptive mode and the tests)
    def predict_es_n0(self, hops: list, symbol_rate_gbaud: float, margin_db: float = 0.0, scint_outage: float = 0.02) -> float:
        """Cascaded end-to-end Es/N0 (linear) from the analytic hop budgets: 1/Es/N0 = sum of 1/Es/N0 per hop (noise is amplified with the signal)."""
        lam = self.cfg.link.wavelength_nm * 1e-9
        inv = 0.0
        for hop in hops:
            bud = dch.beam_budget(hop.link, dch.resolve_geometry(hop.link), lam)
            loss = 10.0 ** (-hop.extra_loss_db / 10.0)
            if hop.scint_index > 0:
                loss *= 10.0 ** (-float(atm.scintillation_fade_db(hop.scint_index, scint_outage)) / 10.0)
            nu = dtx.C_LIGHT / lam
            n0 = dtx.PLANCK * nu * 10.0 ** (hop.link.noise_figure_db / 10.0)
            es = bud.eta_fixed * bud.expected_pointing_eff * loss * hop.tx_power_w / (n0 * symbol_rate_gbaud * 1e9)
            inv += 1.0 / max(es, 1e-300)
        return (1.0 / inv) * 10.0 ** (-margin_db / 10.0)

    def choose_modulation(self, hops: list, symbol_rate_gbaud: float, frame_bits: int):
        """
        Auto-Adaptive: the highest-order scheme whose expected number of bit errors in this frame is below ADAPT_TARGET_ERRORS (so the CRC should pass),
        judged at the analytic end-to-end Es/N0 minus a safety margin. Falls back to BPSK, flagged, if even that is predicted to fail.
        Returns (name, {modulation: expected bit errors}, reason).
        """
        es = self.predict_es_n0(hops, symbol_rate_gbaud, ADAPT_MARGIN_DB)
        table = {m: float(dch.theoretical_ber(m, es)) * frame_bits for m in MODULATIONS}
        for m in ORDER_BY_RATE:
            if table[m] <= ADAPT_TARGET_ERRORS:
                return m, table, f"Es/N0 {10 * math.log10(es):.1f} dB (after {ADAPT_MARGIN_DB:g} dB margin) -> {m} expects {table[m]:.1e} bit errors in the frame"
        return "BPSK", table, (f"Es/N0 {10 * math.log10(es):.1f} dB: even BPSK expects {table['BPSK']:.1e} bit errors per frame; "
                               "lower the symbol rate or raise the power to get a CRC lock")

    # ------------------------------------------------------------------ the transmission
    def send(self, payload: bytes, index: int, modulation: str = "QPSK", symbol_rate_gbaud: float = 10.0, tx_power_w: float | None = None,
             beacon_power_w: float = 1.0, closed_loop: bool = True, candidate: int | None = None, seed: int | None = None,
             pre_locked: bool = False, progress=None) -> TransmissionResult:
        """Fire `payload` through the route in force at time step `index`. Returns everything the interface shows."""
        if not payload:
            raise ValueError("payload is empty")
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload is {len(payload):,} bytes; the limit is {MAX_PAYLOAD_BYTES:,} bytes (the demodulator runs symbol by symbol in Python)")
        say = progress or (lambda msg, frac: None)
        seed = int(seed) if seed else int(np.random.SeedSequence().entropy % 2 ** 31) or 1
        rng = np.random.default_rng([seed, 7])
        sim, cfg = self.sim, self.cfg
        hops = self.hops(index, candidate, tx_power_w, beacon_power_w)
        route_label = " -> ".join([hops[0].src] + [h.dst for h in hops])
        frame_bits = 8 * (len(payload) + 8)
        table, reason = None, "chosen by hand"
        mod = modulation
        if modulation == AUTO:
            mod, table, reason = self.choose_modulation(hops, symbol_rate_gbaud, frame_bits)
        mod = mod.upper() if mod.upper() in dtx.BITS_PER_SYMBOL else mod
        n_sym_est = frame_bits // dtx.BITS_PER_SYMBOL[mod] + 64
        sps = 8 if n_sym_est < 100_000 else 4

        say(f"Modulating {len(payload):,} bytes as {mod} ...", 0.03)
        sig = dtx.transmit(payload, mod, symbol_rate_gbaud * 1e9, sps, 0.35, 16, cfg.link.wavelength_nm * 1e-9, hops[0].tx_power_w)
        lam = cfg.link.wavelength_nm * 1e-9
        blocked = bool(rng.random() < hops[0].outage_fraction)
        scint = 1.0
        if hops[0].scint_index > 0:                                                  # one frozen log-normal fade per frame (mean 1)
            s2 = math.log1p(hops[0].scint_index)
            scint = float(math.exp(rng.normal(-s2 / 2.0, math.sqrt(s2))))
        x = sig
        beam = metrics = bud = None
        prop_ms = proc_ms = 0.0
        for h, hop in enumerate(hops):
            say(f"Hop {h + 1}/{len(hops)}: {hop.src} -> {hop.dst}, {hop.range_km:,.0f} km ...", 0.10 + 0.55 * h / max(1, len(hops)))
            if h == 0:                                                               # atmosphere + events attenuate the beam leaving the station
                keep = 10.0 ** (-(hop.extra_loss_db + (300.0 if blocked else 0.0)) / 20.0) * math.sqrt(scint)
                x = dataclasses.replace(sig, field=sig.field * keep, tx_power_w=max(hop.tx_power_w * keep ** 2, 1e-300))     # dsp.channel takes the launch power from here
            beam, metrics, bud, _ = dch.propagate(x, hop.link, np.random.default_rng([seed, 100 + h]), seed)
            prop_ms += metrics.propagation_ms
            proc_ms += metrics.processing_ms
            if hop.amplified_after:                                                  # GEO relay: receive (noise in), EDFA gain back to its output power
                nxt = hops[h + 1]
                lead = beam.link["lead_in_samples"]
                frame = beam.field[lead:lead + len(sig.field)]
                n0 = beam.noise_psd_w_per_hz
                fs = beam.sample_rate
                noisy = frame + np.sqrt(n0 * fs / 2.0) * (rng.standard_normal(len(frame)) + 1j * rng.standard_normal(len(frame)))
                gain = math.sqrt(nxt.tx_power_w / max(float(np.mean(np.abs(noisy) ** 2)), 1e-300))
                x = dtx.OpticalSignal(field=noisy * gain, frame_bits=sig.frame_bits, modulation=sig.modulation, symbol_rate=sig.symbol_rate, sps=sig.sps,
                                      rolloff=sig.rolloff, rrc_span=sig.rrc_span, wavelength_m=sig.wavelength_m, tx_power_w=nxt.tx_power_w,
                                      n_preamble=sig.n_preamble, n_data_symbols=sig.n_data_symbols, payload_len=sig.payload_len,
                                      modulator_error_db=sig.modulator_error_db)

        say("Final terminal: beacon search, Fast Steering Mirror tracking, demodulation ...", 0.70)
        rep = drx.receive(beam, self.pat_cfg, np.random.default_rng([seed, 99]), closed_loop=closed_loop, pre_locked=pre_locked)
        say("Done", 1.0)

        wrong = pl.byte_diff(payload, rep.payload)
        recovered = rep.payload                                                            # corrupted bytes are kept for the diff viewer; crc_ok says whether to trust them
        ok = bool(rep.crc_ok and not wrong.any())
        pat_ms = None if rep.pat.t_lock_s is None else float(rep.pat.t_lock_s * 1e3)
        ser_ms = sig.duration_s * 1e3
        es_pred = self.predict_es_n0(hops, symbol_rate_gbaud, 0.0, 1.0 - 1e-9)
        blocks = pl.verify_blocks(recovered, len(payload) // pl.BLOCK_BYTES) if len(payload) % pl.BLOCK_BYTES == 0 and pl.verify_blocks(payload).is_test_payload else None
        return TransmissionResult(
            route_label=route_label, hops=hops, modulation=mod, adaptive_table=table, adaptive_reason=reason, time_s=float(sim.t_s[index]), index=index,
            payload=payload, recovered=recovered, crc_ok=ok, status=rep.status, ber=rep.ber, ber_oracle=metrics.ber if len(hops) == 1 else None, n_bit_errors=rep.n_bit_errors,
            n_bits=len(sig.frame_bits), bytes_wrong=int(wrong.sum()), bytes_wrong_pct=100.0 * float(wrong.mean()), frame_loss_pct=0.0 if ok else 100.0,
            sync_ok=rep.sync_ok, evm_db=rep.evm_db, esn0_est_db=rep.esn0_est_db, esn0_predicted_db=10.0 * math.log10(es_pred),
            rx_power_dbm=metrics.rx_power_dbm, path_loss_db=metrics.path_loss_db, blocked_by_event=blocked, symbol_rate_gbaud=symbol_rate_gbaud,
            n_symbols=sig.n_symbols, latency_ms=prop_ms + ser_ms + proc_ms, propagation_ms=prop_ms, serialization_ms=ser_ms, processing_ms=proc_ms,
            pat_ms=pat_ms, pat_events=list(rep.pat.events), pat_trace=rep.pat.trace, rx_symbols=rep.rx_symbols, closed_loop=closed_loop,
            pre_locked=pre_locked, blocks=blocks, seed=seed,
            hop_summaries=[dict(src=h.src, dst=h.dst, kind=h.kind, range_km=h.range_km, elevation_deg=h.elevation_deg, extra_loss_db=h.extra_loss_db,
                                tx_power_w=h.tx_power_w) for h in hops])

    # ------------------------------------------------------------------ continuous streaming
    def stream(self, payload: bytes, start_index: int, duration_s: float, frame_bytes: int = 1024, modulation: str = AUTO, symbol_rate_gbaud: float = 10.0,
               tx_power_w: float | None = None, beacon_power_w: float = 1.0, closed_loop: bool = True, seed: int | None = None, progress=None,
               should_cancel=None) -> StreamResult:
        """
        Slice `payload` into frames and send them one after another while the satellite moves along its orbit: frame k leaves at
        start + (k + 0.5) / n * duration of simulated time, over whatever route is in force THEN, with modulation re-chosen per frame in Auto mode.
        A route change that the simulation marks hitless (new path lit before the old one dropped) hands over with the terminal already tracking;
        any other change - or the first frame, or a frame after a lost lock - starts with a full beacon search.
        """
        if frame_bytes < 16:
            raise ValueError("frame size must be at least 16 bytes")
        sim = self.sim
        rt = sim.route
        step = sim.config.step_s
        frames = [payload[i:i + frame_bytes] for i in range(0, len(payload), frame_bytes)]
        n = len(frames)
        seed = int(seed) if seed else int(np.random.SeedSequence().entropy % 2 ** 31) or 1
        records, recovered_parts = [], []
        prev_idx, prev_cand, prev_locked = None, None, False
        t0 = float(sim.t_s[start_index])
        ho_by_index = {h.index: h for h in rt.handovers}
        n_ho, all_hitless, cancelled, last_out = 0, True, False, None
        for k, fb in enumerate(frames):
            if should_cancel and should_cancel():
                cancelled = True
                break
            t_k = t0 + (k + 0.5) / n * duration_s
            idx = min(sim.index_at(t_k), sim.n - 1)
            if progress:
                progress(f"Frame {k + 1}/{n} at T+{sim.t_s[idx] / 60:.1f} min ...", k / n)
            try:
                cand = self.candidate_at(idx)
            except RouteUnavailable as exc:
                records.append(FrameRecord(k, idx, float(sim.t_s[idx]), "NO ROUTE", "-", False, None, len(fb), float("nan"), None, False, None, False, str(exc)))
                recovered_parts.append(bytes(len(fb)))
                prev_locked = False
                continue
            handover, hitless = False, None
            if prev_cand is not None and cand != prev_cand:
                handover = True
                between = [h for i2, h in ho_by_index.items() if prev_idx < i2 <= idx]
                hitless = all(h.hitless for h in between) if between else True
                n_ho += 1
                all_hitless &= bool(hitless)
            pre = bool(prev_locked and closed_loop and (not handover or hitless))
            try:
                out = self.send(fb, idx, modulation, symbol_rate_gbaud, tx_power_w, beacon_power_w, closed_loop, cand, seed + k * 7919, pre_locked=pre)
            except ValueError as exc:
                records.append(FrameRecord(k, idx, float(sim.t_s[idx]), rt.labels[cand], "-", False, None, len(fb), float("nan"), None, handover, hitless, pre, str(exc)))
                recovered_parts.append(bytes(len(fb)))
                prev_locked = False
                continue
            locked_now = out.sync_ok and (out.pat_ms is not None or pre)
            records.append(FrameRecord(k, idx, float(sim.t_s[idx]), out.route_label, out.modulation, out.crc_ok, out.ber, len(fb), out.latency_ms, out.pat_ms,
                                       handover, hitless, pre, "CRC-32 OK" if out.crc_ok else out.status))
            recovered_parts.append(out.recovered if (out.crc_ok and out.recovered is not None) else bytes(len(fb)))
            prev_idx, prev_cand, prev_locked = idx, cand, bool(locked_now)
            last_out = out
        recovered = b"".join(recovered_parts)
        ok = sum(r.ok for r in records)
        done = len(records)
        blocks = pl.verify_blocks(recovered, len(payload) // pl.BLOCK_BYTES) if len(payload) % pl.BLOCK_BYTES == 0 and pl.verify_blocks(payload).is_test_payload else None
        import zlib
        return StreamResult(records, payload, recovered, ok, done - ok, 100.0 * (done - ok) / max(done, 1), sum(r.n_bytes for r in records if r.ok), n_ho, all_hitless,
                            blocks, zlib.crc32(recovered) == zlib.crc32(payload) and len(recovered) == len(payload), cancelled, last_out)
