"""The random / rare events engine."""
import numpy as np
import pytest

import events as ev
import simulation as sim

DAY = 86400.0


def test_same_seed_same_events_different_seed_different_events():
    cfg = ev.EventsConfig(enabled=True, seed=7, rate_scale=5.0)
    a, b = ev.generate_events(cfg, DAY), ev.generate_events(cfg, DAY)
    assert [(e.key, e.start_s, e.end_s) for e in a] == [(e.key, e.start_s, e.end_s) for e in b]
    c = ev.generate_events(ev.EventsConfig(enabled=True, seed=8, rate_scale=5.0), DAY)
    assert [(e.key, e.start_s) for e in a] != [(e.key, e.start_s) for e in c]


def test_disabled_means_no_random_events_but_manual_ones_still_apply():
    assert ev.generate_events(ev.EventsConfig(enabled=False), DAY) == []
    cfg = ev.EventsConfig(enabled=False, manual=[ev.ManualEvent("ash", 3600.0, 7200.0, 1.0)])
    out = ev.generate_events(cfg, DAY)
    assert len(out) == 1 and out[0].source == "manual" and out[0].loss_db == pytest.approx(60.0) and out[0].outage


def test_event_counts_follow_the_poisson_rates():
    """Average over many seeds: bird crossings arrive at about 6 per day, and the rate scale multiplies that."""
    counts = [sum(e.key == "bird" for e in ev.generate_events(ev.EventsConfig(enabled=True, seed=s), DAY)) for s in range(400)]
    assert np.mean(counts) == pytest.approx(6.0, rel=0.08)
    scaled = [sum(e.key == "bird" for e in ev.generate_events(ev.EventsConfig(enabled=True, seed=s, rate_scale=3.0), DAY)) for s in range(200)]
    assert np.mean(scaled) == pytest.approx(18.0, rel=0.08)
    twelve_h = [sum(e.key == "bird" for e in ev.generate_events(ev.EventsConfig(enabled=True, seed=s), DAY / 2)) for s in range(400)]
    assert np.mean(twelve_h) == pytest.approx(3.0, rel=0.1)


def test_durations_and_losses_respect_the_catalogue_ranges():
    out = ev.generate_events(ev.EventsConfig(enabled=True, seed=3, rate_scale=60.0), DAY)
    assert len(out) > 200
    for e in out:
        et = ev.EVENT_TYPES[e.key]
        assert et.duration_s[0] - 1e-9 <= e.duration_s <= et.duration_s[1] + 1e-9
        if et.loss_db:
            assert et.loss_db[0] - 1e-9 <= e.loss_db <= et.loss_db[1] + 1e-9
        assert 0.0 <= e.start_s <= DAY


def test_disabled_types_are_left_out():
    out = ev.generate_events(ev.EventsConfig(enabled=True, seed=2, rate_scale=30.0, disabled_types=["bird", "aircraft"]), DAY)
    assert out and not any(e.key in ("bird", "aircraft") for e in out)


def test_short_event_is_applied_as_the_fraction_of_the_step_it_covers():
    t = np.arange(0, 100.0, 10.0)
    e = ev.Event("bird", "Bird", ev.ANIMALS, 41.0, 44.0, 0.0, True, 1.0)        # 3 s inside the step centred on t=40 (35..45)
    fx = ev.apply_events([e], t, 10.0)
    assert fx.outage_fraction[4] == pytest.approx(0.3)
    assert fx.outage_fraction.sum() == pytest.approx(0.3)
    assert fx.loss_db.max() == 0.0 and fx.jitter_factor.max() == 1.0


def test_event_spanning_steps_and_boundaries():
    t = np.arange(0, 100.0, 10.0)
    e = ev.Event("storm", "Storm", ev.NATURE, 22.0, 58.0, 20.0, False, 1.0)
    fx = ev.apply_events([e], t, 10.0)
    assert fx.loss_db[3] == pytest.approx(20.0) and fx.loss_db[4] == pytest.approx(20.0)      # fully covered steps (25..35, 35..45)
    assert fx.loss_db[0] == 0.0 and fx.loss_db[8] == 0.0
    assert 0.0 < fx.loss_db[2] < 20.0 and 0.0 < fx.loss_db[6] < 20.0                         # partly covered
    assert fx.outage_fraction.max() == 0.0


def test_overlapping_losses_add_in_decibels_and_outages_union():
    t = np.arange(0, 60.0, 10.0)
    a = ev.Event("storm", "A", ev.NATURE, 0.0, 60.0, 10.0, False, 1.0)
    b = ev.Event("fog", "B", ev.NATURE, 0.0, 60.0, 5.0, False, 1.0)
    c = ev.Event("bird", "C", ev.ANIMALS, -5.0, 60.0, 0.0, True, 1.0)
    d = ev.Event("flock", "D", ev.ANIMALS, -5.0, 30.0, 0.0, True, 1.0)
    fx = ev.apply_events([a, b], t, 10.0)
    assert fx.loss_db[2] == pytest.approx(15.0)
    fx = ev.apply_events([c, d], t, 10.0)
    assert np.all(fx.outage_fraction == pytest.approx(1.0))


def test_earthquake_multiplies_the_jitter_and_active_events_lookup():
    t = np.arange(0, 200.0, 10.0)
    e = ev.Event("earthquake", "Quake", ev.NATURE, 50.0, 90.0, 10.0, True, 20.0)
    fx = ev.apply_events([e], t, 10.0)
    assert fx.jitter_factor[6] == 20.0 and fx.jitter_factor[0] == 1.0 and fx.jitter_factor[15] == 1.0
    assert ev.active_events([e], 60.0, 10.0) == [e] and ev.active_events([e], 150.0, 10.0) == []


# ---- events inside the full simulation -----------------------------------------------------------------------------------
def _first_pass_mid(result):
    """Middle of the longest pass, snapped to a sample time (so event timing relative to the 10 s steps is exact)."""
    s, e = result.primary_windows[int(np.argmax(result.primary_windows[:, 1] - result.primary_windows[:, 0]))]
    return float(result.t_s[result.index_at(0.5 * (s + e))])


def test_a_manual_outage_during_a_pass_blocks_the_beam_and_removes_throughput():
    base = sim.run_simulation(sim.SimulationConfig())
    mid = _first_pass_mid(base)
    cfg = sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("flock", mid - 30.0, 60.0, 0.5)]))
    r = sim.run_simulation(cfg)
    i = r.index_at(mid)
    assert base.primary_link.locked[i] and not r.primary_link.locked[i]
    assert r.beam_state(i)[0] == "blocked" and "Flock" in r.beam_state(i)[1]
    assert r.primary_link.adaptive_throughput_gbps[i] == 0.0 and base.primary_link.adaptive_throughput_gbps[i] > 0
    assert len(r.primary_windows) == len(base.primary_windows) + 1          # the pass is split in two


def test_a_brief_bird_only_trims_throughput_instead_of_deleting_the_step():
    base = sim.run_simulation(sim.SimulationConfig())
    mid = _first_pass_mid(base)
    r = sim.run_simulation(sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("bird", mid - 1.0, 2.0, 0.5)])))
    i = r.index_at(mid)
    assert r.primary_link.locked[i]
    assert r.primary_link.adaptive_throughput_gbps[i] == pytest.approx(0.8 * base.primary_link.adaptive_throughput_gbps[i], rel=1e-6)   # 2 s of a 10 s step lost


def test_a_loss_event_reduces_power_by_exactly_its_decibels():
    base = sim.run_simulation(sim.SimulationConfig())
    mid = _first_pass_mid(base)
    ev_cfg = ev.EventsConfig(manual=[ev.ManualEvent("cirrus", mid - 600.0, 1200.0, 1.0)])      # severity 1 = 4 dB
    r = sim.run_simulation(sim.SimulationConfig(events=ev_cfg))
    i = r.index_at(mid)
    assert base.primary_link.rx_power_dbm[i] - r.primary_link.rx_power_dbm[i] == pytest.approx(4.0, abs=1e-6)


def test_earthquake_degrades_pointing_and_blocks():
    base = sim.run_simulation(sim.SimulationConfig())
    mid = _first_pass_mid(base)
    r = sim.run_simulation(sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("earthquake", mid - 100.0, 200.0, 1.0)])))
    i = r.index_at(mid)
    assert r.primary_link.pointing_loss_db[i] > base.primary_link.pointing_loss_db[i] + 3.0
    assert r.beam_state(i)[0] == "blocked"


def test_events_far_from_any_pass_change_nothing_about_the_link():
    base = sim.run_simulation(sim.SimulationConfig())
    gap_t = base.t_s[np.flatnonzero(~base.primary_link.visible)[10]]
    r = sim.run_simulation(sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("storm", gap_t, 10.0, 1.0)])))
    assert np.array_equal(r.primary_link.locked, base.primary_link.locked)
