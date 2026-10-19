"""Tests for satstk.py: the orbit maths and the STK plumbing, using a stand-in for STK."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import satstk as m  # noqa: E402


class FakeResult:
    def __init__(self, lines):
        self.lines = lines
        self.Count = len(lines)

    def Item(self, i):
        return self.lines[i]


class FakeRoot:
    """Answers Connect commands from a table; anything unknown is rejected like STK rejects a bad command."""
    def __init__(self, answers=None, accept=()):
        self.answers, self.accept, self.sent = answers or {}, accept, []

    def ExecuteCommand(self, cmd):
        self.sent.append(cmd)
        for key, lines in self.answers.items():
            if cmd.startswith(key):
                return FakeResult(lines)
        if any(cmd.startswith(a) for a in self.accept):
            return FakeResult([])
        raise RuntimeError("NACK")


def test_sun_synchronous_inclination_matches_the_textbook_value():
    assert m.sso_inclination_deg(600) == pytest.approx(97.79, abs=0.02)
    assert m.sso_inclination_deg(800) == pytest.approx(98.60, abs=0.03)
    with pytest.raises(ValueError):
        m.sso_inclination_deg(20000)


def test_sun_position_and_local_time_of_node():
    t = m.parse_utcg("21 Dec 2026 00:00:00.000")
    assert m.sun_right_ascension_deg(t) == pytest.approx(268.9, abs=1.5)         # near the solstice the Sun's RA is about 270 deg
    ra = m.sun_right_ascension_deg(t)
    assert m.raan_for_ltan_deg(12.0, t) == pytest.approx(ra)                    # noon node = under the Sun
    assert m.raan_for_ltan_deg(10.5, t) == pytest.approx((ra - 22.5) % 360)     # each hour is 15 degrees


def test_utcg_round_trip():
    t = m.parse_utcg("21 Dec 2026 06:30:15.250")
    assert m.format_utcg(t) == "21 Dec 2026 06:30:15.250"
    assert m.parse_utcg("1 Jan 2027 00:00:00") == m.parse_utcg("1 Jan 2027 00:00:00.000")


def test_offline_orbit_is_circular_with_the_right_period():
    import numpy as np
    start = m.parse_utcg(m.SCENARIO_START)
    el = m.orbit_elements(10.5, 0.0, start)
    period = 2 * math.pi * math.sqrt(el["a_km"] ** 3 / m.MU_KM3_S2)
    t, r = m.offline_propagate(el, start, np.linspace(0, period, 400))
    radii = np.linalg.norm(r, axis=1)
    assert radii.min() == pytest.approx(radii.max()) and radii[0] == pytest.approx(6978.137)
    assert np.linalg.norm(r[-1] - r[0]) < 80                                      # back near the start after one period (J2 drift only)


def test_preview_eclipse_and_intermittent_line_of_sight():
    r = m.offline_preview()
    assert r["period_min"] == pytest.approx(96.7, abs=0.2)
    assert 0.33 < r["eclipse_fraction"] < 0.38                                    # a 600 km orbit is in shadow about 36% of the time near beta ~ 0
    assert 12 <= len(r["eclipse"]) <= 16                                          # one pass per orbit
    assert all(30 < (b - a) / 60 < 37 for a, b in r["eclipse"][1:-1])
    assert 0.05 < r["los_fraction"] < 0.95 and len(r["access"]) > 5               # intermittent: sometimes visible, sometimes not
    assert 1000 < r["range_km"][0] < r["range_km"][1] < 7000


def test_line_of_sight_is_blocked_by_the_earth():
    import numpy as np
    start = m.parse_utcg(m.SCENARIO_START)
    a = m.orbit_elements(10.5, 0.0, start)
    b = m.orbit_elements(10.5, 180.0, start)                                      # opposite side of the same orbit
    _, ra = m.offline_propagate(a, start, np.array([0.0]))
    _, rb = m.offline_propagate(b, start, np.array([0.0]))
    d = rb - ra
    closest = np.linalg.norm(ra + np.clip(-(ra @ d.T) / (d @ d.T), 0, 1) * d)
    assert closest < 1.0                                                          # the straight line passes through the Earth's centre


def test_parse_lighting_report_finds_the_three_kinds():
    lines = ["Satellite-Orbital-Node-Alpha: Lighting Times", "Sunlight", "Start Time (UTCG)  Stop Time (UTCG)  Duration (sec)",
             "21 Dec 2026 00:00:00.000   21 Dec 2026 00:23:40.100   1420.100",
             "Penumbra", "21 Dec 2026 00:23:40.100   21 Dec 2026 00:24:00.200   20.100",
             "Umbra", "21 Dec 2026 00:24:00.200   21 Dec 2026 00:58:10.000   2049.800"]
    rows = m.parse_lighting_report(lines)
    assert [r["kind"] for r in rows] == ["Sunlight", "Penumbra", "Umbra"]
    assert rows[2]["duration_s"] == pytest.approx(2049.8) and rows[2]["stop"] == "21 Dec 2026 00:58:10.000"
    assert m.parse_lighting_report(["nothing useful", "21 Dec 2026 00:00:00.000 21 Dec 2026 00:01:00.000 60"]) == []   # no heading, no kind


def test_connect_wraps_errors_with_the_command():
    with pytest.raises(m.StkError, match="Transmitter nope"):
        m.connect(FakeRoot(), "Transmitter nope")
    assert m.connect(FakeRoot({"Report_RM": ["a", "b"]}), "Report_RM x") == ["a", "b"]


def test_set_first_matching_finds_an_attribute_by_name_pattern():
    answers = {"Transmitter_RM p GetValue Model.AntennaControl": ["Antenna = Optical_Gaussian", "Antenna.Beamwidth = 0.001 rad",
                                                                   "Antenna.Efficiency = 0.55", "Antenna.DesignFrequency = 1 Hz (Read-only)"]}
    root = FakeRoot(answers, accept=["Transmitter p SetValue Model.AntennaControl.Antenna.Beamwidth", "Transmitter p SetValue Antenna.Beamwidth"])
    hit = m.set_first_matching(root, "Transmitter", "p", r"divergence|beamwidth", "1e-4")
    assert hit and hit.endswith("Beamwidth") and any("SetValue" in c and "Beamwidth 1e-4" in c for c in root.sent)
    assert m.set_first_matching(FakeRoot(answers), "Transmitter", "p", r"nothing_like_this", "1") is None
    assert m.set_first_matching(FakeRoot(answers), "Transmitter", "p", r"designfrequency", "1") is None     # read-only is never written


def test_summarise_link_ignores_the_no_link_flag():
    rows = [dict(**{"Rcvd. Iso. Power": -9999.99, "BER": 0.5, "Propagation Delay": 0.016, "Range": 4800}),
            dict(**{"Rcvd. Iso. Power": -52.0, "BER": 1e-12, "Propagation Delay": 0.0145, "Range": 4350}),
            dict(**{"Rcvd. Iso. Power": -50.0, "BER": 1e-30, "Propagation Delay": 0.0150, "Range": 4500})]
    s = m.summarise_link(rows)
    assert s["samples"] == 3 and s["valid_samples"] == 2
    assert s["power_dbw"] == (-52.0, -51.0, -50.0)
    assert s["latency_ms"] == (pytest.approx(14.5), pytest.approx(16.0))
    assert m.summarise_link([dict(Range=3000.0)])["latency_ms"][0] == pytest.approx(3000 / m.C_KM_S * 1000)       # falls back to range / c
    assert m.summarise_link([])["power_dbw"] is None


def test_laser_frequency_for_1550_nm():
    assert m.C_KM_S * 1000 / (1550e-9) == pytest.approx(1.934e14, rel=1e-3)


def test_start_stk_explains_a_missing_python_api():
    with pytest.raises(m.StkError, match="agi.stk12"):
        m.start_stk()
