"""Ground-station geometry, atmosphere / weather / turbulence, and the ground-to-satellite link budget."""
import math

import numpy as np
import pytest

import atmosphere as at
import laser_link_engine as ll
import orbital_engine as oe
import simulation as sim
from helpers import direct_cfg


# ---- geometry ---------------------------------------------------------------------------------------------------------
def test_satellite_straight_overhead_has_90_degrees_elevation_and_range_equals_altitude():
    gs = oe.GroundStation("g", 10.0, 20.0, 0.0, 20.0)
    r = oe.geodetic_to_ecef(10.0, 20.0, 600.0)[None, :]
    t = oe.topocentric_state(gs, r, np.zeros((1, 3)))
    assert t.elevation_deg[0] == pytest.approx(90.0, abs=1e-6) and t.range_km[0] == pytest.approx(600.0, abs=1e-6)


def test_azimuth_points_north_east_south_west():
    gs = oe.GroundStation("g", 0.0, 0.0, 0.0)
    base = oe.geodetic_to_ecef(0.0, 0.0, 0.0)
    for (dx, dy, dz), expected in (((0, 0, 500), 0.0), ((0, 500, 0), 90.0), ((0, 0, -500), 180.0), ((0, -500, 0), 270.0)):
        r = (base + np.array([100.0, dy, dz]))[None, :]            # 100 km up plus the sideways offset
        t = oe.topocentric_state(gs, r, np.zeros((1, 3)))
        assert t.azimuth_deg[0] == pytest.approx(expected, abs=1e-6)


def test_horizon_distance_formula():
    # a satellite at 600 km is visible down to elevation 0 within acos(Re/(Re+h)) = 23.9 degrees of the station
    assert oe.horizon_cap_radius_deg(600.0, 0.0) == pytest.approx(math.degrees(math.acos(6378.137 / 6978.137)), abs=1e-9)
    assert oe.horizon_cap_radius_deg(600.0, 0.0) == pytest.approx(23.93, abs=0.01)
    assert oe.horizon_cap_radius_deg(600.0, 20.0) < oe.horizon_cap_radius_deg(600.0, 0.0)
    assert oe.horizon_cap_radius_deg(600.0, 90.0) == pytest.approx(0.0, abs=1e-9)


def test_slant_range_at_the_horizon_matches_pythagoras():
    gs = oe.GroundStation("g", 0.0, 0.0, 0.0)
    h, re = 600.0, oe.R_EARTH
    rho = math.acos(re / (re + h))                                 # satellite on the geometric horizon
    r = np.array([[(re + h) * math.cos(rho), (re + h) * math.sin(rho), 0.0]])
    t = oe.topocentric_state(gs, r, np.zeros((1, 3)))
    assert t.elevation_deg[0] == pytest.approx(0.0, abs=0.2)       # spherical vs ellipsoidal station position
    assert t.range_km[0] == pytest.approx(math.sqrt((re + h) ** 2 - re ** 2), rel=2e-3)


def test_topocentric_rates_match_finite_differences():
    r = sim.run_simulation(direct_cfg(duration_h=24, step_s=1.0))
    k = int(np.argmax(r.primary_link.locked))
    sl = slice(k + 20, k + 200)
    t = r.topo[0]
    az = np.unwrap(np.radians(t.azimuth_deg))
    assert np.degrees(np.gradient(az, r.t_s))[sl] == pytest.approx(t.azimuth_rate_deg_s[sl], abs=2e-3)
    assert np.degrees(np.gradient(np.radians(t.elevation_deg), r.t_s))[sl] == pytest.approx(t.elevation_rate_deg_s[sl], abs=2e-3)


def test_sun_elevation_at_noon_on_the_equator_in_equinox():
    from datetime import datetime, timezone
    start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    t = np.array([0.0])
    sun, g = oe.sun_position_eci(start, t), oe.gmst_rad(start, t)
    assert oe.sun_elevation_deg(oe.GroundStation("g", 0.0, 0.0, 0.0), sun, g)[0] > 80.0       # near the subsolar point (equation of time ~ 7 min)


def test_doppler_for_a_ground_pass_is_largest_at_the_horizon_and_zero_at_closest_approach():
    r = sim.run_simulation(direct_cfg(duration_h=24, step_s=1.0))
    link = r.primary_link
    w = r.primary_windows[int(np.argmax([e - s for s, e in r.primary_windows]))]
    idx = np.flatnonzero((r.t_s >= w[0]) & (r.t_s <= w[1]))
    rate = link.range_rate_km_s[idx]
    assert rate[0] < 0 < rate[-1]                                  # approaching, then receding
    assert abs(rate[0]) > 4.0 and abs(rate[-1]) > 4.0              # several km/s near the edges
    closest = idx[np.argmin(link.range_km[idx])]
    assert abs(link.range_rate_km_s[closest]) < 0.3


# ---- atmosphere --------------------------------------------------------------------------------------------------------
def test_air_mass_known_values():
    assert float(at.air_mass(90.0)) == pytest.approx(1.0, abs=2e-3)
    assert float(at.air_mass(30.0)) == pytest.approx(2.0, abs=0.01)
    assert float(at.air_mass(10.0)) == pytest.approx(5.6, abs=0.1)


def test_kim_visibility_model_values():
    assert at.kim_exponent(100.0) == 1.6 and at.kim_exponent(23.0) == 1.3 and at.kim_exponent(0.2) == 0.0
    # beta(550 nm) = 3.912 / V : visibility is defined by a 2 % contrast threshold
    assert at.aerosol_extinction_per_km(23.0, 550.0) == pytest.approx(3.912 / 23.0)
    assert at.aerosol_extinction_per_km(23.0, 1550.0) == pytest.approx(3.912 / 23.0 * (1550 / 550) ** -1.3)


def test_clear_air_loss_grows_toward_the_horizon_and_drops_with_altitude():
    cfg = at.AtmosphereConfig()
    el = np.array([90.0, 45.0, 20.0, 10.0])
    sea = at.clear_air_loss_db(el, 0.0, cfg)
    mountain = at.clear_air_loss_db(el, 4500.0, cfg)
    assert np.all(np.diff(sea) > 0) and np.all(mountain < sea)
    assert 0.1 < float(sea[0]) < 0.4                               # a fraction of a dB straight up in clear air


def test_haze_costs_more_than_clear_air():
    clear = at.clear_air_loss_db(45.0, 0.0, at.AtmosphereConfig(visibility_km=40.0))
    hazy = at.clear_air_loss_db(45.0, 0.0, at.AtmosphereConfig(visibility_km=4.0))
    assert hazy > 3 * clear


def test_weather_loss_scales_with_path_length_and_adds_up():
    cfg = at.AtmosphereConfig(cloud_loss_db=10.0, fog_loss_db=5.0, rain_mm_h=0.0)
    assert float(at.weather_loss_db(90.0, cfg)) == pytest.approx(15.0)
    assert float(at.weather_loss_db(30.0, cfg)) == pytest.approx(30.0)               # 1/sin(30) = 2
    rain = at.AtmosphereConfig(rain_mm_h=10.0)
    assert float(at.weather_loss_db(90.0, rain)) == pytest.approx(1.076 * 10 ** 0.67 * 2.0)


def test_turbulence_clear_site_values_are_sensible():
    cfg = at.AtmosphereConfig()
    r0 = float(at.fried_parameter_m(90.0, 0.0, cfg))
    assert 0.1 < r0 < 0.5                                          # Fried parameter at 1550 nm straight up: tens of cm
    assert float(at.fried_parameter_m(20.0, 0.0, cfg)) < r0         # worse near the horizon
    assert 0.5 * r0 < float(at.fried_parameter_m(90.0, 0.0, at.AtmosphereConfig(turbulence_cn2_ground=1.7e-15))) * 3 and \
        float(at.fried_parameter_m(90.0, 0.0, at.AtmosphereConfig(turbulence_cn2_ground=1.7e-15))) > r0   # calmer air, bigger r0
    assert float(at.fried_parameter_m(90.0, 4500.0, cfg)) > r0     # high site: less air above it


def test_rytov_and_scintillation_monotonic_and_weak_at_zenith():
    cfg = at.AtmosphereConfig()
    ryt = at.rytov_variance(np.array([90.0, 45.0, 20.0]), 0.0, cfg)
    assert np.all(np.diff(ryt) > 0) and ryt[0] < 0.2
    assert float(at.scintillation_index(0.01)) == pytest.approx(0.01, rel=0.05)      # weak turbulence: sigma_I^2 ~ sigma_R^2
    assert float(at.scintillation_index(100.0)) < 3.0                                 # saturates (strong turbulence), does not blow up
    fades = at.scintillation_fade_db(np.array([0.01, 0.1, 0.5]))
    assert np.all(np.diff(fades) > 0)


def test_lognormal_fade_matches_monte_carlo():
    rng = np.random.default_rng(5)
    si = 0.2
    s2 = math.log1p(si)
    intensity = np.exp(rng.normal(-s2 / 2, math.sqrt(s2), 2_000_000))
    p = 1e-2
    assert float(at.scintillation_fade_db(si, p)) == pytest.approx(-10 * math.log10(np.quantile(intensity, p)), abs=0.1)


# ---- ground link budget -----------------------------------------------------------------------------------------------
def _link(atmosphere=None, **kw):
    cfg = direct_cfg(duration_h=24, **kw)
    if atmosphere:
        cfg.atmosphere = atmosphere
    return sim.run_simulation(cfg)


def test_ground_link_budget_by_hand_at_the_peak_of_a_pass():
    r = _link()
    link = r.primary_link
    k = int(np.argmin(np.where(link.locked, link.range_km, np.inf)))
    rng_m = link.range_km[k] * 1000.0
    ch = link.channel
    div = math.hypot(100.0, ch["beam_spread_urad"][k]) * 1e-6
    capture = 1 - math.exp(-2 * (0.15 / (div * rng_m / 2)) ** 2)
    point = 1 / (1 + 16 * (10e-6) ** 2 / div ** 2)
    loss_db = ch["clear_air_loss_db"][k] + ch["weather_loss_db"][k] + ch["scintillation_fade_db"][k]
    expected_w = 10.0 * 0.64 * capture * point * 10 ** (-loss_db / 10)
    assert link.rx_power_dbm[k] == pytest.approx(10 * math.log10(expected_w * 1e3), abs=1e-6)
    assert link.latency_ms[k] == pytest.approx(rng_m / ll.C_M_S * 1e3, rel=1e-9)


def test_closer_satellite_gives_more_power_and_a_pass_is_best_overhead():
    link = _link().primary_link
    v = link.locked
    r, p = link.range_km[v], link.rx_power_dbm[v]
    assert np.corrcoef(r, p)[0, 1] < -0.8


def test_clouds_cut_the_received_power_and_can_kill_the_link():
    clear = _link().primary_link
    cloudy = _link(atmosphere=at.AtmosphereConfig(cloud_loss_db=10.0)).primary_link
    k = int(np.argmax(clear.locked))
    drop = clear.rx_power_dbm[k] - cloudy.rx_power_dbm[k]
    assert drop == pytest.approx(10.0 / math.sin(math.radians(clear.channel["elevation_deg"][k])), abs=0.01)
    thick = _link(atmosphere=at.AtmosphereConfig(cloud_loss_db=60.0)).primary_link
    assert thick.adaptive_throughput_gbps.max() == 0.0               # opaque cloud: geometry fine, no usable data
    assert thick.locked.any()                                        # ... and the beam state is 'degraded', not 'blocked'


def test_higher_divergence_costs_power_and_bigger_aperture_gains_it():
    base = _link().primary_link
    wide = sim.run_simulation(direct_cfg(link=ll.LaserLinkConfig(divergence_urad=200.0))).primary_link
    k = int(np.argmax(base.locked))
    assert base.rx_power_dbm[k] - wide.rx_power_dbm[k] == pytest.approx(6.0, abs=0.7)
    big = sim.run_simulation(direct_cfg(link=ll.LaserLinkConfig(rx_aperture_m=0.6))).primary_link
    assert big.rx_power_dbm[k] - base.rx_power_dbm[k] == pytest.approx(6.0, abs=0.7)


def test_ground_telescope_slew_peaks_for_a_high_pass_and_stays_finite():
    r = _link()
    link = r.primary_link
    assert np.all(np.isfinite(link.pointing_a.slew_rate_deg_s))
    assert 0.1 < np.max(link.pointing_a.slew_rate_deg_s[link.locked]) < 15.0     # tracking a LEO satellite is a few tenths of a degree per second or more
