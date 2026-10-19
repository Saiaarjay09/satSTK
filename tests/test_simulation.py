"""Whole-scenario checks: the engines wired together, including cross-checks between independent code paths."""
import json

import numpy as np
import pytest

import orbital_engine as oe
import simulation as sim
from helpers import direct_cfg


@pytest.fixture(scope="module")
def result():
    return sim.run_simulation(direct_cfg(satellites=sim.two_satellites(), step_s=10.0))


def test_default_scenario_is_the_south_india_station_one_satellite_and_three_geo_relays():
    cfg = sim.SimulationConfig()
    assert [s.name for s in cfg.satellites] == ["Orbital-Node-Alpha"]
    assert (cfg.ground.name, cfg.ground.lat_deg, cfg.ground.lon_deg, cfg.ground.min_elevation_deg) == ("GS-INDIA", 13.0, 77.5, 10.0)
    assert cfg.relay.enabled and [r.short for r in cfg.relay.relays] == ["GEO-IND", "GEO-ATL", "GEO-PAC"]
    r = sim.run_simulation(cfg)
    assert len(r.tracks) == 1 and r.n == 8641 and list(r.ground_links) == [0] and r.crosslinks == {} and len(r.relay_tracks) == 3


def test_two_satellite_scenario_shape(result):
    assert [t.name for t in result.tracks] == ["Orbital-Node-Alpha", "Orbital-Node-Beta"]
    assert sorted(result.ground_links) == [0, 1] and sorted(result.crosslinks) == [(0, 1)]
    assert result.t_s[-1] == 86400.0


def test_satellites_are_600km_sun_synchronous(result):
    for tr in result.tracks:
        a, e, i, *_ = tr.elements0
        assert a - oe.R_EARTH == pytest.approx(600.0)
        assert np.degrees(i) == pytest.approx(97.79, abs=0.02)


def test_station_never_moves_in_the_earth_frame_and_goes_round_in_the_inertial_frame(result):
    r_ecef = oe.eci_to_ecef(result.station_eci, result.gmst)
    assert np.ptp(r_ecef, axis=0) == pytest.approx([0, 0, 0], abs=1e-6)
    assert np.linalg.norm(result.station_eci, axis=1) == pytest.approx(np.linalg.norm(oe.GroundStation().ecef()))
    # after one sidereal day the station is back where it started in the inertial frame
    lon = np.unwrap(np.arctan2(result.station_eci[:, 1], result.station_eci[:, 0]))
    assert (lon[-1] - lon[0]) / (2 * np.pi) == pytest.approx(86400.0 / 86164.1, abs=2e-4)


def test_passes_agree_with_the_mask_and_have_sane_durations(result):
    link = result.primary_link
    win = result.access_windows[0]
    assert 1 <= len(win) <= 8
    dur = (win[:, 1] - win[:, 0]) / 60.0
    assert np.all(dur < 12.0) and dur.max() > 2.0                      # a 600 km pass above 20 degrees lasts a few minutes, never more than ~12
    el = link.channel["elevation_deg"]
    assert np.all(el[link.locked] >= result.config.ground.min_elevation_deg - 1e-9)
    assert np.all(~link.locked[el < result.config.ground.min_elevation_deg])
    assert link.range_km[link.locked].min() >= 600.0 - 1e-6              # never closer than the altitude


def test_range_matches_the_inertial_geometry_and_range_rate_matches_its_derivative(result):
    link = result.primary_link
    d = np.linalg.norm(result.tracks[0].r_eci - result.station_eci, axis=1)
    assert link.range_km == pytest.approx(d)
    assert np.gradient(link.range_km, result.t_s)[5:-5] == pytest.approx(link.range_rate_km_s[5:-5], abs=0.05)


def test_beam_state_colours(result):
    link = result.primary_link
    i_up = int(np.argmax(link.locked))
    assert result.beam_state(i_up)[0] in ("locked", "degraded")
    assert result.beam_state(int(np.argmax(~link.visible)))[0] == "blocked"


def test_one_second_and_ten_second_results_agree():
    fine = sim.run_simulation(direct_cfg(step_s=1.0, duration_h=24))
    coarse = sim.run_simulation(direct_cfg(step_s=10.0, duration_h=24))
    wf, wc = fine.primary_windows, coarse.primary_windows
    assert wf.shape == wc.shape
    assert wf == pytest.approx(wc, abs=12.0)                           # pass edges within one 10 s step


def test_orbital_period_visible_in_the_simulated_ground_track(result):
    lat = result.tracks[0].lat_deg
    up = np.flatnonzero((lat[:-1] < 0) & (lat[1:] >= 0))
    assert np.diff(result.t_s[up]).mean() == pytest.approx(96.69 * 60, rel=0.002)


def test_eclipses_repeat_every_orbit(result):
    umbra = result.eclipse_windows[0]["umbra"]
    assert len(umbra) in (14, 15)
    assert np.diff(umbra[:, 0]).mean() == pytest.approx(96.69 * 60, rel=0.002)
    assert 33.0 < ((umbra[:, 1] - umbra[:, 0]) / 60.0).mean() < 36.0


def test_a_station_at_the_pole_sees_the_sun_synchronous_satellite_every_orbit():
    """An orbit inclined 97.79 degrees reaches 82.2 degrees latitude, so a station at 82 N sees nearly every revolution."""
    r = sim.run_simulation(direct_cfg(ground=oe.GroundStation("Pole", 80.0, 0.0, 0.0, 10.0), duration_h=24))
    assert len(r.primary_windows) >= 12


def test_moving_the_station_moves_the_passes():
    a = sim.run_simulation(direct_cfg(ground=oe.GroundStation("A", 28.3, -16.5, 0.0, 20.0)))
    b = sim.run_simulation(direct_cfg(ground=oe.GroundStation("B", 28.3, 100.0, 0.0, 20.0)))
    assert not np.allclose(a.primary_windows[:1], b.primary_windows[:1])


def test_higher_minimum_elevation_means_fewer_and_shorter_passes():
    lo = sim.run_simulation(direct_cfg(ground=oe.GroundStation("S", 28.3, -16.5, 0.0, 10.0)))
    hi = sim.run_simulation(direct_cfg(ground=oe.GroundStation("S", 28.3, -16.5, 0.0, 50.0)))
    assert np.sum(lo.primary_link.locked) > np.sum(hi.primary_link.locked)


def test_energy_stays_bounded_with_j2(result):
    tr = result.tracks[0]
    e = 0.5 * np.sum(tr.v_eci ** 2, axis=1) - oe.MU_EARTH / np.linalg.norm(tr.r_eci, axis=1)
    assert np.ptp(e) / abs(e.mean()) < 0.01


def test_config_round_trips_through_json_including_events():
    from events import EventsConfig, ManualEvent
    cfg = direct_cfg(step_s=1.0, duration_h=12, satellites=sim.two_satellites(), target_index=1,
                               events=EventsConfig(enabled=True, seed=9, rate_scale=4.0, disabled_types=["bird"], manual=[ManualEvent("ash", 3600.0, 7200.0, 0.8)]))
    cfg.ground = oe.GroundStation("Somewhere", -33.9, 18.4, 25.0, 30.0)
    cfg.atmosphere.rain_mm_h = 12.0
    again = sim.SimulationConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.step_s == 1.0 and again.duration_h == 12 and again.target_index == 1 and again.start == cfg.start
    assert again.ground == cfg.ground and again.atmosphere.rain_mm_h == 12.0
    assert again.events.manual[0].key == "ash" and again.events.seed == 9 and again.events.disabled_types == ["bird"]
    assert [s.name for s in again.satellites] == [s.name for s in cfg.satellites]


def test_second_satellite_can_be_the_target():
    r = sim.run_simulation(direct_cfg(satellites=sim.two_satellites(), target_index=1, duration_h=6))
    assert r.primary_link is r.ground_links[1] and r.target == 1


def test_three_satellites_give_three_crosslink_pairs():
    sats = sim.two_satellites() + [oe.SatelliteConfig("Orbital-Node-Gamma", ltan_hours=12.0, true_anomaly_deg=90.0)]
    r = sim.run_simulation(direct_cfg(satellites=sats, duration_h=2, step_s=10))
    assert sorted(r.crosslinks) == [(0, 1), (0, 2), (1, 2)]


def test_validation_rejects_bad_input():
    for bad in (direct_cfg(satellites=[]), direct_cfg(step_s=0.0), direct_cfg(target_index=3),
                direct_cfg(ground=oe.GroundStation("x", 95.0, 0.0, 0.0, 20.0)),
                direct_cfg(ground=oe.GroundStation("x", 0.0, 0.0, 0.0, 95.0))):
        with pytest.raises(ValueError):
            bad.validate()


def test_csv_export_and_summary(tmp_path, result):
    p = tmp_path / "out.csv"
    result.write_csv(p)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result.n + 1 and lines[0].startswith("t_s,utc,elevation_deg")
    text = result.summary_text()
    assert "Beam origin" in text and "passes above the mask" in text and "Orbital-Node-Alpha" in text


def test_no_proprietary_stk_or_windows_com_imports_anywhere():
    """The point of the rewrite: no Ansys STK, no COM. Scan every source file."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    banned = ("comtypes", "win32com", "pythoncom", "agi.stk", "ansys.stk", "import agi")
    for f in list(root.glob("*.py")) + list(root.glob("ui/*.py")) + [root / "requirements.txt"]:
        text = f.read_text(encoding="utf-8")
        for word in banned:
            assert word not in text, f"{f.name} mentions {word}"
