"""OrbitalOpticalTransceiver: live orbital state -> hop chain -> symbol-level DSP -> recovered payload."""
import math

import numpy as np
import pytest

import events as ev
import optical_dsp_bridge as ob
import simulation as sim
from dsp import channel as dch
from dsp import payload as pl
from helpers import direct_cfg
from laser_link_engine import link_budget

LAM = 1550e-9


@pytest.fixture(scope="module")
def world():
    r = sim.run_simulation(sim.SimulationConfig())
    return r, ob.OrbitalOpticalTransceiver(r)


def at(world, kind, frac=0.5):
    r, _ = world
    w = np.flatnonzero(r.route.kind == kind)
    return int(w[int(len(w) * frac)])


# ---- hop chain from the simulation state -------------------------------------------------------------------------------------
def test_hop_chain_matches_the_route_ranges_and_terminals(world):
    r, trx = world
    for kind, n_hops in ((1, 1), (2, 2), (3, 3)):
        i = at(world, kind)
        hops = trx.hops(i)
        assert len(hops) == n_hops and " -> ".join([hops[0].src] + [h.dst for h in hops]) == r.route.label(i)
        assert sum(h.range_km for h in hops) == pytest.approx(r.primary_link.range_km[i], rel=1e-9)         # same geometry as the simulation's path length
        assert hops[0].src == "GS-INDIA" and hops[-1].dst == "NODE-ALPHA"
    h1 = trx.hops(at(world, 1))[0]
    assert h1.kind == "direct" and 600 < h1.range_km < 2500 and h1.tx_power_w == 10.0
    h2 = trx.hops(at(world, 2))
    assert h2[0].kind == "ground" and h2[0].range_km == pytest.approx(35976, abs=15) and h2[0].elevation_deg == pytest.approx(74.7, abs=0.3)
    assert h2[1].kind == "last" and 35000 < h2[1].range_km < 44500 and h2[1].tx_power_w == 10.0              # the EDFA on the relay
    h3 = trx.hops(at(world, 3))
    assert [h.kind for h in h3] == ["ground", "relay", "last"] and 60000 < h3[1].range_km < 80000


def test_dsp_hops_carry_the_relay_terminals_300mm_10w_10urad(world):
    r, trx = world
    hop = trx.hops(at(world, 2))[1]
    assert hop.link.rx_aperture_m == 0.30 and hop.link.tx_jitter_urad == 1.0
    geo = dch.resolve_geometry(hop.link)
    bud = dch.beam_budget(hop.link, geo, LAM)
    assert 2 * bud.divergence_half_urad == pytest.approx(10.0, rel=0.01)                                      # a 10 microradian full-angle beam
    assert hop.link.tx_aperture_m == pytest.approx(ob.divergence_to_aperture(10.0, LAM))
    assert ob.divergence_to_aperture(10.0, LAM) < 0.30                                                       # it fits inside a 300 mm telescope


def test_dsp_power_budget_equals_the_simulations_analytic_budget_hop_by_hop(world):
    """Fraction of the launched power that arrives, hop by hop: the DSP channel's Gaussian-optics maths vs satSTK's link_budget."""
    import laser_link_engine as ll
    r, trx = world
    cfg = r.config
    for kind in (1, 2, 3):
        for h in trx.hops(at(world, kind)):
            bud = dch.beam_budget(h.link, dch.resolve_geometry(h.link), LAM)
            dsp_fraction = bud.eta_fixed * bud.expected_pointing_eff
            sat_cfg = cfg.link if h.kind == "direct" else ll.hop_config(cfg.link, cfg.relay)
            sat = link_budget(h.range_km, sat_cfg, divergence_urad=2 * bud.divergence_half_urad, jitter_urad=h.link.tx_jitter_urad)
            sat_fraction = float(sat["rx_power_w"]) / sat_cfg.tx_power_w
            assert 10 * math.log10(dsp_fraction / sat_fraction) == pytest.approx(0.0, abs=0.02)


def test_predicted_end_to_end_es_n0_equals_the_simulations_snr_on_every_path_type(world):
    r, trx = world
    link = r.primary_link
    for kind in (1, 2, 3):
        for i in np.flatnonzero(r.route.kind == kind)[::700]:
            es = trx.predict_es_n0(trx.hops(int(i)), 250.0, 0.0, 1e-3)          # same 0.1 % scintillation margin the simulation uses
            assert 10 * math.log10(es) == pytest.approx(float(link.snr_db[i]), abs=1e-3)


def test_unavailable_or_missing_routes_raise(world):
    r, trx = world
    i = at(world, 2)
    label = r.route.labels
    atl = label.index("GS-INDIA -> GEO-ATL -> NODE-ALPHA")                      # GEO-ATL is below the horizon from India: that candidate never exists
    assert not r.route.available[atl].any()
    with pytest.raises(ob.RouteUnavailable):
        trx.hops(i, candidate=atl)
    with pytest.raises(ValueError):
        trx.send(b"", i)
    with pytest.raises(ValueError):
        trx.send(bytes(ob.MAX_PAYLOAD_BYTES + 1), i)


# ---- sending ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", [1, 2, 3])
@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM"])
def test_every_route_and_modulation_delivers_a_text_payload_at_10_gbaud(world, kind, mod):
    r, trx = world
    msg = ("Hello from GS-INDIA via the " + {1: "direct beam", 2: "GEO relay", 3: "two GEO relays"}[kind] + "!").encode()
    out = trx.send(msg, at(world, kind), mod, 10.0, seed=3)
    assert out.crc_ok and out.recovered == msg and out.ber == 0.0 and out.frame_loss_pct == 0.0 and out.bytes_wrong == 0
    assert out.modulation == mod and out.sync_ok and out.pat_ms is not None and out.pat_ms < 5.0


def test_latency_is_light_time_over_every_hop_plus_serialisation(world):
    r, trx = world
    for kind in (1, 2, 3):
        i = at(world, kind)
        out = trx.send(b"timing " * 20, i, "QPSK", 10.0, seed=4)
        assert out.propagation_ms == pytest.approx(sum(h.range_km for h in out.hops) * 1000 / 299792458.0 * 1e3, rel=1e-9)
        assert out.latency_ms == pytest.approx(out.propagation_ms + out.serialization_ms + out.processing_ms)
        assert out.serialization_ms == pytest.approx(out.n_symbols / 10e9 * 1e3, rel=1e-9)
    assert out.latency_ms > 470                                              # a three-hop route is half a second of light time


def test_the_63kb_payload_goes_through_every_route_with_every_64_byte_block_intact(world):
    r, trx = world
    data = pl.generate_test_payload()
    for kind in (1, 2, 3):
        out = trx.send(data, at(world, kind), ob.AUTO, 10.0, seed=7)
        assert out.crc_ok and out.recovered == data and out.blocks.n_bad == 0 and out.blocks.n_blocks == 1008


def test_a_too_dense_modulation_over_the_longest_route_flips_a_bit_and_the_blocks_localise_it(world):
    r, trx = world
    data = pl.generate_test_payload()
    out = trx.send(data, at(world, 3, 0.5), "16QAM", 10.0, seed=1)
    assert not out.crc_ok and 0 < out.bytes_wrong < 50                                   # a handful of flipped bits is enough to fail the CRC
    assert 1 <= out.blocks.n_bad <= out.bytes_wrong                                      # ... and the per-block CRCs say which 64-byte lines they hit
    assert out.frame_loss_pct == 100.0 and out.recovered is not None and len(out.recovered) == len(data)


def test_at_250_gbaud_the_geo_paths_cannot_pass_a_crc_without_fec(world):
    """The simulation's 250 Gbaud budget relies on FEC (raw BER ~1e-3); this DSP has none, so a 63 KB frame cannot survive it."""
    r, trx = world
    data = pl.generate_test_payload()
    out = trx.send(data, at(world, 3), "BPSK", 250.0, seed=2)
    assert not out.crc_ok and out.ber is not None and 1e-5 < out.ber < 1e-1
    small = trx.send(b"tiny", at(world, 2), "QPSK", 250.0, seed=2)           # a tiny frame often squeaks through; a big one never does
    assert small.n_bits < out.n_bits


def test_real_receiver_pays_a_few_db_over_textbook_theory(world):
    """The measured (PAT residual + carrier loop) BER sits well above the analytic BER at the same Es/N0, which is the point of simulating it."""
    r, trx = world
    i = at(world, 2)
    data = bytes(np.random.default_rng(0).integers(0, 256, 4000, dtype=np.uint8))
    es = trx.predict_es_n0(trx.hops(i), 250.0, 0.0, 1 - 1e-9)
    theory = float(dch.theoretical_ber("QPSK", es))
    meas = np.mean([trx.send(data, i, "QPSK", 250.0, seed=s).ber for s in range(1, 6)])
    assert meas > 5 * theory and meas < 0.05


def test_open_loop_fsm_cannot_receive_but_closed_loop_can(world):
    r, trx = world
    i = at(world, 2)
    on = trx.send(b"steer the mirror " * 4, i, "QPSK", 10.0, closed_loop=True, seed=5)
    off = trx.send(b"steer the mirror " * 4, i, "QPSK", 10.0, closed_loop=False, seed=5)
    assert on.crc_ok and not off.crc_ok and not off.sync_ok and off.pat_ms is None
    assert on.pat_trace is not None and np.any(on.pat_trace["state"] == 0) and np.any(on.pat_trace["state"] == 1)


def test_a_weak_beacon_means_no_lock_over_a_geo_hop(world):
    r, trx = world
    i = at(world, 2)
    out = trx.send(b"no beacon, no lock " * 4, i, "QPSK", 10.0, beacon_power_w=1e-6, seed=6)
    assert out.pat_ms is None and not out.crc_ok and any("TIMEOUT" in t for _, t in out.pat_events)


def test_transmit_power_changes_the_received_power_by_the_same_number_of_db(world):
    r, trx = world
    i = at(world, 1)
    a = trx.send(b"power " * 10, i, "QPSK", 10.0, tx_power_w=10.0, seed=8)
    b = trx.send(b"power " * 10, i, "QPSK", 10.0, tx_power_w=1.0, seed=8)
    assert a.rx_power_dbm - b.rx_power_dbm == pytest.approx(10.0, abs=0.01)
    assert a.esn0_predicted_db - b.esn0_predicted_db == pytest.approx(10.0, abs=0.01)


def test_cloud_and_events_in_the_simulation_reach_the_dsp():
    import atmosphere as at_
    cfg = sim.SimulationConfig()
    cfg.atmosphere = at_.WEATHER_PRESETS["Overcast"]
    r = sim.run_simulation(cfg)
    trx = ob.OrbitalOpticalTransceiver(r)
    i = int(np.flatnonzero(r.route.kind == 2)[100])
    hop0 = trx.hops(i)[0]
    assert hop0.extra_loss_db > 25.0                                           # 25 dB of cloud, a bit more at 74.7 degrees elevation
    out = trx.send(b"through the clouds " * 8, i, "BPSK", 10.0, seed=9)
    assert not out.crc_ok                                                       # ~28 dB less signal at 10 Gbaud: no frame
    clear = sim.run_simulation(sim.SimulationConfig())
    assert ob.OrbitalOpticalTransceiver(clear).send(b"through the clouds " * 8, i, "BPSK", 10.0, seed=9).crc_ok


def test_an_event_that_blocks_the_beam_blocks_the_frame():
    base = sim.run_simulation(sim.SimulationConfig())
    i = int(np.flatnonzero(base.route.kind == 2)[200])
    t = float(base.t_s[i])
    r = sim.run_simulation(sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("flock", t - 60.0, 120.0, 0.5)])))
    trx = ob.OrbitalOpticalTransceiver(r)
    assert r.effects.outage_fraction[i] == 1.0
    out = trx.send(b"a flock of birds " * 5, i, "QPSK", 10.0, seed=10)
    assert out.blocked_by_event and not out.crc_ok and not out.sync_ok
    free = ob.OrbitalOpticalTransceiver(base).send(b"a flock of birds " * 5, i, "QPSK", 10.0, seed=10)
    assert free.crc_ok and not free.blocked_by_event


def test_earthquake_shaking_raises_the_pointing_jitter_the_dsp_sees():
    base = sim.run_simulation(sim.SimulationConfig())
    i = int(np.flatnonzero(base.route.kind == 2)[150])
    t = float(base.t_s[i])
    r = sim.run_simulation(sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("earthquake", t - 100.0, 200.0, 1.0)])))
    j_quake = ob.OrbitalOpticalTransceiver(r).hops(i)[0].link.tx_jitter_urad
    j_calm = ob.OrbitalOpticalTransceiver(base).hops(i)[0].link.tx_jitter_urad
    assert j_quake == pytest.approx(j_calm * 40.0)


# ---- adaptive modulation ------------------------------------------------------------------------------------------------------
def test_adaptive_mode_picks_a_denser_scheme_on_better_links_and_never_worse_than_needed(world):
    r, trx = world
    order = {m: k for k, m in enumerate(["BPSK", "QPSK", "8PSK", "16QAM"])}
    picks = {}
    for kind in (1, 2, 3):
        hops = trx.hops(at(world, kind))
        for rate in (10.0, 40.0, 100.0):
            picks[(kind, rate)] = trx.choose_modulation(hops, rate, 8 * 1032)[0]
    for rate in (10.0, 40.0, 100.0):
        assert order[picks[(1, rate)]] >= order[picks[(2, rate)]] >= order[picks[(3, rate)]]            # fewer hops, denser modulation
    for kind in (1, 2, 3):
        assert order[picks[(kind, 10.0)]] >= order[picks[(kind, 40.0)]] >= order[picks[(kind, 100.0)]]    # faster symbols, sparser modulation
    assert picks[(1, 100.0)] != picks[(3, 100.0)] and picks[(3, 40.0)] == "BPSK"


def test_adaptive_mode_always_gets_a_crc_lock_where_a_lock_is_possible(world):
    r, trx = world
    rng = np.random.default_rng(1)
    for rate in (10.0, 40.0, 60.0):
        for kind in (1, 2, 3):
            data = bytes(rng.integers(0, 256, 8000, dtype=np.uint8))
            out = trx.send(data, at(world, kind, rng.random()), ob.AUTO, rate, seed=int(rng.integers(1, 10 ** 6)))
            assert out.crc_ok, f"{kind} {rate} {out.modulation}"
            assert out.adaptive_table is not None and set(out.adaptive_table) == {"BPSK", "QPSK", "8PSK", "16QAM"}


def test_adaptive_mode_reports_when_even_bpsk_cannot_work(world):
    r, trx = world
    hops = trx.hops(at(world, 3))
    mod, table, reason = trx.choose_modulation(hops, 500.0, 8 * 64520)
    assert mod == "BPSK" and table["BPSK"] > ob.ADAPT_TARGET_ERRORS and "lower the symbol rate" in reason


# ---- continuous streaming --------------------------------------------------------------------------------------------------------
def test_stream_across_a_direct_pass_hands_over_without_dropping_a_frame(world):
    r, trx = world
    data = pl.generate_test_payload()
    direct_start = float(r.direct_windows[0][0][0])
    start = r.index_at(direct_start - 1800.0)
    s = trx.stream(data, start, 3600.0, 1024, ob.AUTO, 40.0, seed=3)
    labels = {f.route_label for f in s.frames}
    assert "GS-INDIA -> NODE-ALPHA" in labels and len(labels) >= 2                       # started on a GEO route, crossed onto the direct pass, and back
    assert s.handovers >= 2 and s.all_handovers_hitless
    assert s.frames_lost == 0 and s.frame_loss_pct == 0.0 and s.stream_crc_ok and s.recovered == data
    assert s.blocks.n_bad == 0 and len(s.frames) == 63
    assert len({f.modulation for f in s.frames}) >= 2                                    # the modulation followed the route
    assert not s.frames[0].pre_locked and all(f.pre_locked for f in s.frames[1:] if not f.handover or f.hitless)


def test_stream_log_records_each_frames_route_time_and_modulation(world):
    r, trx = world
    data = pl.generate_test_payload()[:8 * 1024]
    s = trx.stream(data, 0, 1800.0, 1024, "QPSK", 10.0, seed=5)
    assert len(s.frames) == 8 and [f.k for f in s.frames] == list(range(8))
    assert all(f.modulation == "QPSK" and f.ok for f in s.frames)
    assert all(a.time_s < b.time_s for a, b in zip(s.frames, s.frames[1:])) and s.frames[-1].time_s < 1800.0
    assert s.bytes_ok == len(data)


def test_a_frame_with_no_route_is_lost_and_filled_with_zeros():
    r = sim.run_simulation(direct_cfg())                                                  # no relays: most of the day there is no route
    trx = ob.OrbitalOpticalTransceiver(r)
    data = b"x" * 3000
    s = trx.stream(data, 0, 600.0, 1000, "QPSK", 10.0, seed=2)
    assert s.frames_lost == 3 and all(f.route_label == "NO ROUTE" for f in s.frames) and s.recovered == bytes(3000) and not s.stream_crc_ok


def test_stream_can_be_cancelled(world):
    r, trx = world
    calls = []
    s = trx.stream(pl.generate_test_payload(), 0, 3600.0, 1024, "QPSK", 10.0, seed=1, should_cancel=lambda: len(calls.append(1) or calls) > 5)
    assert s.cancelled and len(s.frames) == 5
