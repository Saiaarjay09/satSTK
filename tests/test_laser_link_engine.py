"""Tests for laser_link_engine.py - every one compares against a closed-form value worked out by hand."""
import math

import numpy as np
import pytest

import laser_link_engine as ll
import orbital_engine as oe

LAM = 1550e-9


# ---- free-space path loss ---------------------------------------------------------------------------------------------
def test_fspl_matches_formula_at_1000km():
    # 20 log10(4 pi * 1e6 m / 1.55e-6 m) = 20 log10(8.107e12) = 258.18 dB
    assert 4 * math.pi * 1.0e6 / LAM == pytest.approx(8.1073e12, rel=1e-4)
    assert float(ll.fspl_db(1.0e6, LAM)) == pytest.approx(258.18, abs=0.01)


def test_fspl_is_6db_per_doubling_of_distance():
    assert float(ll.fspl_db(2.0e6, LAM) - ll.fspl_db(1.0e6, LAM)) == pytest.approx(6.0206, abs=1e-4)


def test_fspl_changes_20db_per_decade_of_wavelength():
    assert float(ll.fspl_db(1e6, 155e-9) - ll.fspl_db(1e6, 1550e-9)) == pytest.approx(20.0, abs=1e-9)


# ---- Gaussian beam ----------------------------------------------------------------------------------------------------
def test_beam_radius_grows_linearly_with_range():
    # 100 urad full divergence at 2000 km: radius 100 m (200 m wide)
    assert float(ll.beam_radius_m(2.0e6, 100e-6)) == pytest.approx(100.0)


def test_capture_fraction_far_field_equals_2_a2_over_w2():
    # a = 0.15 m, w = 100 m  ->  2 * 0.0225 / 1e4 = 4.5e-6
    assert float(ll.gaussian_capture_fraction(2.0e6, 100e-6, 0.30)) == pytest.approx(4.5e-6, rel=1e-4)


def test_capture_fraction_saturates_to_one_at_short_range():
    assert float(ll.gaussian_capture_fraction(10.0, 100e-6, 0.30)) == pytest.approx(1.0)


def test_antenna_gain_view_agrees_with_capture_fraction_in_far_field():
    """Pr/Pt = Gt * Gr / FSPL  must equal the Gaussian capture fraction when the beam is much wider than the aperture."""
    for r in (1.0e6, 3.0e6, 5.0e6):
        via_gains_db = ll.tx_gain_db(100e-6) + ll.rx_gain_db(0.30, LAM) - float(ll.fspl_db(r, LAM))
        via_capture_db = 10 * math.log10(float(ll.gaussian_capture_fraction(r, 100e-6, 0.30)))
        assert via_gains_db == pytest.approx(via_capture_db, abs=0.01)


def test_pointing_loss_formula():
    div = 100e-6
    assert ll.pointing_loss_linear(div, 0.0, 0.0) == 1.0
    # 10 urad RMS per axis -> 1/(1 + 16*0.01) = 0.8621 = -0.64 dB
    assert ll.pointing_loss_linear(div, 10e-6) == pytest.approx(1 / 1.16)
    # a static offset of one beam half-width (50 urad) loses exp(-2) of the on-axis power
    assert ll.pointing_loss_linear(div, 0.0, 50e-6) == pytest.approx(math.exp(-2.0))


def test_pointing_loss_matches_monte_carlo():
    rng = np.random.default_rng(1)
    div, sigma = 100e-6, 15e-6
    e = rng.normal(0, sigma, (400_000, 2))
    mc = np.mean(np.exp(-8.0 * np.sum(e ** 2, axis=1) / div ** 2))
    assert ll.pointing_loss_linear(div, sigma) == pytest.approx(mc, rel=0.005)


# ---- received power / budget ------------------------------------------------------------------------------------------
def test_received_power_at_2000km_by_hand():
    cfg = ll.LaserLinkConfig(pointing_jitter_urad=0.0)
    b = ll.link_budget(2000.0, cfg)
    expected_w = 10.0 * 0.8 * 0.8 * 4.5e-6                 # Ptx * eta_tx * eta_rx * capture
    assert float(b["rx_power_w"]) == pytest.approx(expected_w, rel=1e-3)
    assert float(b["rx_power_dbm"]) == pytest.approx(10 * math.log10(expected_w * 1e3), abs=0.01)
    assert float(b["latency_ms"]) == pytest.approx(2.0e6 / ll.C_M_S * 1e3)          # 6.671 ms
    assert float(b["latency_ms"]) == pytest.approx(6.671, abs=0.001)


def test_received_power_falls_6db_per_range_doubling():
    cfg = ll.LaserLinkConfig()
    p = ll.link_budget(np.array([2000.0, 4000.0]), cfg)["rx_power_dbm"]
    assert p[0] - p[1] == pytest.approx(6.02, abs=0.01)


# ---- bit error rate ---------------------------------------------------------------------------------------------------
def test_q_function_known_values():
    assert float(ll.q_function(0.0)) == pytest.approx(0.5)
    assert float(ll.q_function(1.0)) == pytest.approx(0.158655, abs=1e-6)
    assert float(ll.q_function(6.0)) == pytest.approx(9.866e-10, rel=1e-3)


def test_bpsk_ber_textbook_points():
    # Eb/N0 = 9.6 dB -> BER 1e-5 ; Eb/N0 = 0 dB -> 7.865e-2
    assert float(ll.ber_from_eb_n0("BPSK", 10 ** 0.96)) == pytest.approx(1.0e-5, rel=0.15)
    assert float(ll.ber_from_eb_n0("BPSK", 1.0)) == pytest.approx(0.07865, abs=1e-4)
    assert float(ll.ber_from_eb_n0("QPSK", 1.0)) == pytest.approx(float(ll.ber_from_eb_n0("BPSK", 1.0)))


def test_higher_order_modulations_need_more_energy():
    g = 10 ** (10 / 10)     # 10 dB
    b = [float(ll.ber_from_eb_n0(m, g)) for m in ("BPSK", "QPSK", "8PSK", "16QAM")]
    assert b[0] == b[1] < b[2] < b[3]
    assert float(ll.ber_from_eb_n0("16QAM", g)) == pytest.approx(0.75 * float(ll.q_function(math.sqrt(8.0))))


def test_quantum_limit_nine_photons_per_bit_gives_1e9_for_homodyne_bpsk():
    cfg = ll.LaserLinkConfig(detector_efficiency=1.0, implementation_penalty_db=0.0, symbol_rate_gbaud=1.0)
    # choose the power that delivers exactly 9 photons per bit at 1 Gbit/s
    p_rx = 9.0 * cfg.photon_energy_j * 1e9
    es_n0 = cfg.detector_efficiency * p_rx / 1e9 / (0.5 * cfg.photon_energy_j)
    assert es_n0 == pytest.approx(18.0)
    assert float(ll.ber_from_eb_n0("BPSK", es_n0)) == pytest.approx(1e-9, rel=0.05)


def test_ber_is_monotonic_in_range():
    cfg = ll.LaserLinkConfig(implementation_penalty_db=25.0)
    b = ll.link_budget(np.linspace(500, 5200, 50), cfg)["ber"]["16QAM"]
    assert np.all(np.diff(b) >= 0)


# ---- Doppler / range rate ---------------------------------------------------------------------------------------------
def test_doppler_shift_of_7_km_per_s_closing_speed():
    # f = c/1550 nm = 193.41 THz ; shift = v/lambda = 7000/1.55e-6 = 4.516 GHz
    f0 = ll.C_M_S / LAM
    assert f0 == pytest.approx(193.414e12, rel=1e-5)
    assert float(ll.doppler_shift_hz(-7.0, f0)) / 1e9 == pytest.approx(4.516, abs=0.002)
    assert float(ll.doppler_shift_hz(+7.0, f0)) / 1e9 == pytest.approx(-4.516, abs=0.002)


def test_range_rate_is_line_of_sight_component():
    rng, rate = ll.relative_kinematics([[0, 0, 0]], [[0, 0, 0]], [[1000, 0, 0]], [[3.0, 4.0, 0]])
    assert rng[0] == pytest.approx(1000.0)
    assert rate[0] == pytest.approx(3.0)            # only the x component is along the line of sight


# ---- turret pointing ---------------------------------------------------------------------------------------------------
def _circular_pair(sep_deg):
    r = oe.R_EARTH + 600.0
    n = math.sqrt(oe.MU_EARTH / r ** 3)
    s = math.radians(sep_deg)
    r1, v1 = np.array([[r, 0.0, 0.0]]), np.array([[0.0, r * n, 0.0]])
    r2 = np.array([[r * math.cos(s), r * math.sin(s), 0.0]])
    v2 = np.array([[-r * n * math.sin(s), r * n * math.cos(s), 0.0]])
    return r1, v1, r2, v2


def test_two_satellites_in_the_same_circular_orbit_need_no_slewing():
    p = ll.turret_pointing(*_circular_pair(30.0))
    assert p.slew_rate_deg_s[0] == pytest.approx(0.0, abs=1e-9)
    assert p.azimuth_deg[0] == pytest.approx(0.0, abs=1e-6)          # straight ahead along the flight direction
    assert p.elevation_deg[0] == pytest.approx(-15.0, abs=1e-6)      # chord of a 30 degree arc dips 15 degrees below the local horizon
    # satellite 1 sees satellite 2 ahead, satellite 2 sees satellite 1 behind
    behind = ll.turret_pointing(*(lambda a, b, c, d: (c, d, a, b))(*_circular_pair(30.0)))
    assert behind.azimuth_deg[0] == pytest.approx(180.0, abs=1e-6)


def test_slew_rate_equals_orbit_rate_when_target_is_fixed_in_inertial_space():
    """A distant 'target' that does not move in inertial space appears to rotate at the orbital rate -> slew = n (0.062 deg/s)."""
    r = oe.R_EARTH + 600.0
    n = math.sqrt(oe.MU_EARTH / r ** 3)
    p = ll.turret_pointing([[r, 0, 0]], [[0, r * n, 0]], [[5.0e5, 5.0e5, 0.0]], [[0.0, 0.0, 0.0]])
    # for a far-away fixed point the direction vector is fixed (up to the tiny parallax), so the pointing vector turns at -n in the frame
    assert p.slew_rate_deg_s[0] == pytest.approx(math.degrees(n), rel=0.02)


def test_azimuth_rate_matches_finite_difference():
    """Check the analytic az/el rates against numerically differentiated angles for two satellites in different orbit planes."""
    t = np.arange(0.0, 1200.0, 1.0)
    a, b = oe.SatelliteConfig("a", ltan_hours=10.5), oe.SatelliteConfig("b", ltan_hours=13.5, true_anomaly_deg=45.0)
    from datetime import datetime, timezone
    start = datetime(2026, 12, 21, tzinfo=timezone.utc)
    sun, g = oe.sun_position_eci(start, t), oe.gmst_rad(start, t)
    ta, tb = oe.compute_track(a, start, t, sun, g), oe.compute_track(b, start, t, sun, g)
    p = ll.turret_pointing(ta.r_eci, ta.v_eci, tb.r_eci, tb.v_eci)
    fd_az = np.gradient(np.unwrap(np.radians(p.azimuth_deg)), t)
    fd_el = np.gradient(np.radians(p.elevation_deg), t)
    assert np.degrees(fd_az[5:-5]) == pytest.approx(p.azimuth_rate_deg_s[5:-5], abs=2e-4)
    assert np.degrees(fd_el[5:-5]) == pytest.approx(p.elevation_rate_deg_s[5:-5], abs=2e-4)


# ---- end-to-end -------------------------------------------------------------------------------------------------------
def test_crosslink_masks_blocked_times_and_throughput():
    """Satellite-to-satellite crosslink (needs two satellites): numbers exist only while the Earth is not in the way."""
    import simulation as sim
    r = sim.run_simulation(sim.SimulationConfig(satellites=sim.two_satellites(), duration_h=3, step_s=10))
    link = r.crosslinks[(0, 1)]
    assert link.visible.any() and (~link.visible).any()
    assert np.all(np.isnan(link.rx_power_dbm[~link.visible])) and np.all(np.isfinite(link.rx_power_dbm[link.visible]))
    assert np.all(link.throughput_gbps["16QAM"][~link.visible] == 0)
    assert np.all(np.isnan(link.ber["QPSK"][~link.visible]))
    vis = link.visible
    assert np.all(link.throughput_gbps["BPSK"][vis] == pytest.approx(250 / 1.07))
    assert np.all(link.throughput_gbps["QPSK"][vis] == pytest.approx(2 * 250 / 1.07))
    assert np.all(link.adaptive_throughput_gbps[vis] >= link.throughput_gbps["QPSK"][vis])
    assert link.pointing_loss_db[0] == pytest.approx(0.6446, abs=0.001)
    assert link.budget_constants["carrier_frequency_thz"] == pytest.approx(193.41, abs=0.01)


def test_extra_loss_and_divergence_overrides_in_budget():
    cfg = ll.LaserLinkConfig(pointing_jitter_urad=0.0)
    base = ll.link_budget(1000.0, cfg)
    lossy = ll.link_budget(1000.0, cfg, extra_loss_db=10.0)
    assert float(base["rx_power_dbm"] - lossy["rx_power_dbm"]) == pytest.approx(10.0, abs=1e-9)
    wide = ll.link_budget(1000.0, cfg, divergence_urad=200.0)       # twice the divergence = 4x the area = -6 dB (far field)
    assert float(base["rx_power_dbm"] - wide["rx_power_dbm"]) == pytest.approx(6.02, abs=0.05)
    shaky = ll.link_budget(1000.0, cfg, jitter_urad=np.array([0.0, 20.0, 40.0]))
    assert np.all(np.diff(shaky["rx_power_dbm"]) < 0)
