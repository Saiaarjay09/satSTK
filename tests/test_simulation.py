"""Whole-scenario checks: the engines wired together, including cross-checks between independent code paths."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

import orbital_engine as oe
import simulation as sim


@pytest.fixture(scope="module")
def result():
    return sim.run_simulation(sim.SimulationConfig(step_s=10.0))


def test_default_scenario_shape(result):
    assert len(result.tracks) == 2 and result.n == 8641
    assert [t.name for t in result.tracks] == ["Orbital-Node-Alpha", "Orbital-Node-Beta"]
    assert result.t_s[-1] == 86400.0


def test_both_satellites_are_600km_sun_synchronous(result):
    for tr in result.tracks:
        a, e, i, *_ = tr.elements0
        assert a - oe.R_EARTH == pytest.approx(600.0)
        assert np.degrees(i) == pytest.approx(97.79, abs=0.02)


def test_access_windows_agree_with_visibility_mask(result):
    link = result.primary_link
    win = result.access_windows[(0, 1)]
    inside = np.zeros(result.n, dtype=bool)
    for s, e in win:
        inside |= (result.t_s >= s - 1e-9) & (result.t_s <= e + 1e-9)
    # every sample marked visible lies inside a window (samples within 10 s of an edge may differ by interpolation)
    assert np.mean(inside == link.visible) > 0.995
    assert np.all(np.diff(win.ravel()) > 0)


def test_one_second_and_ten_second_results_agree():
    fine = sim.run_simulation(sim.SimulationConfig(step_s=1.0, duration_h=6))
    coarse = sim.run_simulation(sim.SimulationConfig(step_s=10.0, duration_h=6))
    wf, wc = fine.access_windows[(0, 1)], coarse.access_windows[(0, 1)]
    assert wf.shape == wc.shape
    assert wf == pytest.approx(wc, abs=5.0)                       # window edges within 5 s (the interpolation helps a lot)
    uf, uc = fine.eclipse_windows[0]["umbra"], coarse.eclipse_windows[0]["umbra"]
    assert uf == pytest.approx(uc, abs=10.0)


def test_orbital_period_visible_in_the_simulated_ground_track(result):
    """The satellite returns to the same latitude/phase once per period: the ascending-node crossings are ~96.7 min apart."""
    lat = result.tracks[0].lat_deg
    up = np.flatnonzero((lat[:-1] < 0) & (lat[1:] >= 0))
    spacing = np.diff(result.t_s[up])
    assert spacing.mean() == pytest.approx(96.69 * 60, rel=0.002)


def test_eclipses_repeat_every_orbit(result):
    umbra = result.eclipse_windows[0]["umbra"]
    assert len(umbra) in (14, 15)
    assert np.diff(umbra[:, 0]).mean() == pytest.approx(96.69 * 60, rel=0.002)
    dur = (umbra[:, 1] - umbra[:, 0]) / 60.0
    assert 33.0 < dur.mean() < 36.0                                # beta ~ -17 deg: a little under the 35.5 min maximum


def test_los_is_symmetric_and_range_is_geometric(result):
    link = result.primary_link
    d = np.linalg.norm(result.tracks[1].r_eci - result.tracks[0].r_eci, axis=1)
    assert link.range_km == pytest.approx(d)
    # inside every access window the beam clears the buffered Earth, outside it does not
    assert np.all(link.clearance_km[link.visible] >= 0) and np.all(link.clearance_km[~link.visible] < 0)


def test_doppler_integrates_to_range_change(result):
    link = result.primary_link
    # range rate from the velocity dot product must match the derivative of range
    numeric = np.gradient(link.range_km, result.t_s)
    assert numeric[5:-5] == pytest.approx(link.range_rate_km_s[5:-5], abs=2e-3)


def test_energy_is_not_conserved_but_stays_bounded_with_j2(result):
    """J2 exchanges energy with the oblate-Earth potential; specific orbital energy wobbles by a small fraction only."""
    tr = result.tracks[0]
    e = 0.5 * np.sum(tr.v_eci ** 2, axis=1) - oe.MU_EARTH / np.linalg.norm(tr.r_eci, axis=1)
    assert np.ptp(e) / abs(e.mean()) < 0.01


def test_config_round_trips_through_json():
    cfg = sim.SimulationConfig(step_s=1.0, duration_h=12)
    again = sim.SimulationConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.step_s == 1.0 and again.duration_h == 12
    assert [s.name for s in again.satellites] == [s.name for s in cfg.satellites]
    assert again.link.tx_power_w == cfg.link.tx_power_w and again.start == cfg.start


def test_three_satellites_give_three_pairs():
    sats = sim.default_satellites() + [oe.SatelliteConfig("Orbital-Node-Gamma", ltan_hours=12.0, true_anomaly_deg=90.0)]
    r = sim.run_simulation(sim.SimulationConfig(satellites=sats, duration_h=2, step_s=10))
    assert sorted(r.links) == [(0, 1), (0, 2), (1, 2)]


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        sim.SimulationConfig(satellites=sim.default_satellites()[:1]).validate()
    with pytest.raises(ValueError):
        sim.SimulationConfig(step_s=0.0).validate()


def test_csv_export_and_summary(tmp_path, result):
    p = tmp_path / "out.csv"
    result.write_csv(p)
    lines = p.read_text().splitlines()
    assert len(lines) == result.n + 1 and lines[0].startswith("t_s,utc,los,range_km")
    text = result.summary_text()
    assert "Orbital-Node-Alpha" in text and "access windows" in text


def test_no_proprietary_stk_or_windows_com_imports_anywhere():
    """The point of this rewrite: no Ansys STK, no COM. Scan every source file."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    banned = ("comtypes", "win32com", "pythoncom", "agi.stk", "ansys.stk", "import agi")
    for f in list(root.glob("*.py")) + list(root.glob("ui/*.py")) + [root / "requirements.txt"]:
        text = f.read_text()
        for word in banned:
            assert word not in text, f"{f.name} mentions {word}"
