"""Tests for orbital_engine.py. Each one checks the code against a known physical value or an independent hand calculation."""
import math
from datetime import datetime, timezone

import numpy as np
import pytest

import orbital_engine as oe

START = datetime(2026, 12, 21, 0, 0, 0, tzinfo=timezone.utc)


# ---- orbital period ---------------------------------------------------------------------------------------------------
def test_period_formula_600km():
    # T = 2*pi*sqrt(a^3/mu) = 5801.2 s = 96.69 min for a 600 km circular orbit
    assert oe.orbital_period_s(oe.R_EARTH + 600.0) == pytest.approx(5801.23, abs=0.05)


def test_two_body_orbit_closes_after_one_period():
    a = oe.R_EARTH + 600.0
    r0, v0 = oe.elements_to_state(a, 0.0, math.radians(97.8), 1.0, 0.0, 0.3)
    period = oe.orbital_period_s(a)
    r, v = oe.propagate_state(r0, v0, np.array([0.0, period]), use_j2=False)
    assert np.linalg.norm(r[-1] - r0) < 1e-3       # back to the start within a metre
    assert np.linalg.norm(v[-1] - v0) < 1e-6


def test_two_body_conserves_energy_and_momentum():
    r0, v0 = oe.elements_to_state(oe.R_EARTH + 800.0, 0.05, 0.9, 2.0, 1.0, 0.5)
    t = np.linspace(0, 86400, 500)
    r, v = oe.propagate_state(r0, v0, t, use_j2=False)
    energy = 0.5 * np.sum(v ** 2, axis=1) - oe.MU_EARTH / np.linalg.norm(r, axis=1)
    assert np.ptp(energy) / abs(energy[0]) < 1e-9
    h = np.linalg.norm(np.cross(r, v), axis=1)
    assert np.ptp(h) / h[0] < 1e-9


def test_elements_round_trip():
    el = (7200.0, 0.02, 0.8, 1.1, 2.2, 0.7)
    r, v = oe.elements_to_state(*el)
    got = oe.state_to_elements(r, v)
    assert got == pytest.approx(el, rel=1e-9, abs=1e-9)


# ---- Sun-synchronous orbit --------------------------------------------------------------------------------------------
def test_sso_inclination_600km():
    # published value for a 600 km Sun-synchronous orbit: about 97.79 degrees
    assert oe.sso_inclination_deg(600.0) == pytest.approx(97.79, abs=0.02)


def test_sso_precession_rate_is_one_revolution_per_year():
    inc = math.radians(oe.sso_inclination_deg(600.0))
    raan_dot, _, _ = oe.j2_secular_rates(oe.R_EARTH + 600.0, 0.0, inc)
    deg_per_day = math.degrees(raan_dot) * 86400.0
    assert deg_per_day == pytest.approx(360.0 / 365.2421897, abs=1e-6)   # +0.98565 degrees/day, eastward


def test_numerical_j2_propagation_reproduces_sso_precession():
    """Integrate the real J2 acceleration for 3 days and fit the RAAN drift of the orbit plane."""
    inc = math.radians(oe.sso_inclination_deg(600.0))
    r0, v0 = oe.elements_to_state(oe.R_EARTH + 600.0, 0.0, inc, 1.0, 0.0, 0.0)
    t = np.arange(0, 3 * 86400, 60.0)
    r, v = oe.propagate_state(r0, v0, t, use_j2=True)
    h = np.cross(r, v)
    raan = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))          # node longitude from the orbit-normal vector
    slope_deg_per_day = np.degrees(np.polyfit(t, raan, 1)[0]) * 86400.0
    # the osculating start isn't exactly the mean orbit, so allow 2 % (0.02 deg/day)
    assert slope_deg_per_day == pytest.approx(0.9856, abs=0.02)


def test_prograde_orbit_regresses_westward():
    raan_dot, _, _ = oe.j2_secular_rates(oe.R_EARTH + 500.0, 0.0, math.radians(51.6))
    assert math.degrees(raan_dot) * 86400 == pytest.approx(-5.0, abs=0.3)   # ISS-like orbit: about -5 degrees/day


def test_ltan_to_raan_puts_node_at_requested_local_time():
    raan = oe.raan_for_ltan_deg(10.5, START)
    ra_sun = oe.sun_right_ascension_deg(START)
    assert (raan - ra_sun) % 360.0 == pytest.approx(-22.5, abs=1e-9) or (raan - ra_sun) % 360.0 == pytest.approx(337.5, abs=1e-9)


# ---- Sun and frames ---------------------------------------------------------------------------------------------------
def test_sun_position_solstice():
    s = oe.sun_position_eci(START, [0.0])[0]
    dist_au = np.linalg.norm(s) / oe.AU_KM
    dec = math.degrees(math.asin(s[2] / np.linalg.norm(s)))
    assert dist_au == pytest.approx(0.9837, abs=0.001)          # near perihelion in early January
    assert dec == pytest.approx(-23.44, abs=0.1)                # December solstice declination


def test_sun_position_equinox_declination_zero():
    s = oe.sun_position_eci(datetime(2026, 3, 20, 14, 46, tzinfo=timezone.utc), [0.0])[0]
    assert math.degrees(math.asin(s[2] / np.linalg.norm(s))) == pytest.approx(0.0, abs=0.05)


def test_gmst_known_value():
    # 2000-01-01 12:00:00 UTC: GMST = 18h 41m 50.5s = 280.4606 degrees
    g = oe.gmst_rad(datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc), [0.0])[0]
    assert math.degrees(g) == pytest.approx(280.4606, abs=0.01)


def test_geodetic_round_trip_and_known_points():
    lat, lon, alt = oe.ecef_to_geodetic(oe.geodetic_to_ecef(np.array([0.0, 45.0, -80.0, 89.0]), np.array([0.0, 10.0, -120.0, 170.0]),
                                                           np.array([0.0, 600.0, 25.0, 1000.0])))
    assert lat == pytest.approx([0.0, 45.0, -80.0, 89.0], abs=1e-9)
    assert lon == pytest.approx([0.0, 10.0, -120.0, 170.0], abs=1e-9)
    assert alt == pytest.approx([0.0, 600.0, 25.0, 1000.0], abs=1e-6)
    assert oe.geodetic_to_ecef(0.0, 0.0, 0.0) == pytest.approx([oe.R_EARTH, 0.0, 0.0])
    assert oe.geodetic_to_ecef(90.0, 0.0, 0.0)[2] == pytest.approx(oe.R_POLAR)


# ---- line of sight ----------------------------------------------------------------------------------------------------
def test_los_blocked_through_the_earth_and_clear_above_it():
    r = oe.R_EARTH + 600.0
    a = np.array([[r, 0, 0]])
    assert not oe.line_of_sight(a, np.array([[-r, 0, 0]]))[0][0]          # opposite sides of the planet
    assert oe.line_of_sight(a, np.array([[r * math.cos(0.3), r * math.sin(0.3), 0]]))[0][0]   # 17 degrees apart: clear


def test_los_grazing_limit_matches_geometry_with_100km_buffer():
    """Two equal-altitude satellites see each other until the chord dips to R_E + 100 km above the centre (equatorial plane)."""
    r = oe.R_EARTH + 600.0
    r_clear = oe.R_EARTH + oe.ATMOSPHERE_BUFFER_KM
    half = math.acos(r_clear / r)                                  # half the separation angle at the limit
    for sep, expect in ((2 * half - 0.01, True), (2 * half + 0.01, False)):
        p2 = np.array([[r * math.cos(sep), r * math.sin(sep), 0.0]])
        assert oe.line_of_sight(np.array([[r, 0, 0]]), p2)[0][0] == expect
    # the limit separation is about 43.6 degrees, i.e. 5185 km, for 600 km orbits with a 100 km buffer
    assert math.degrees(2 * half) == pytest.approx(43.6, abs=0.1)
    assert 2 * r * math.sin(half) == pytest.approx(5185, abs=5)


def test_los_buffer_matters():
    """A path that clears the bare surface by 50 km is blocked once the 100 km atmosphere buffer is applied."""
    d = oe.R_EARTH + 50.0
    a, b = np.array([[d, -3000.0, 0]]), np.array([[d, 3000.0, 0]])
    assert oe.line_of_sight(a, b, buffer_km=0.0)[0][0]
    assert not oe.line_of_sight(a, b, buffer_km=100.0)[0][0]


def test_los_uses_the_flattened_ellipsoid():
    """Over the pole the surface is 21 km closer to the centre than at the equator, so the same chord clears at the pole only."""
    dist = oe.R_POLAR + 100.0 + 5.0       # 5 km above the buffered polar surface
    over_pole = oe.line_of_sight(np.array([[-3000.0, 0, dist]]), np.array([[3000.0, 0, dist]]))
    equator = oe.line_of_sight(np.array([[-3000.0, dist, 0]]), np.array([[3000.0, dist, 0]]))
    assert over_pole[0][0] and not equator[0][0]


def test_access_interval_edges_are_interpolated():
    t = np.arange(0, 100.0, 10.0)
    clearance = 3.0 - np.abs(t - 45.0) / 5.0 * 1.0     # +ve for |t-45|<15
    win = oe.mask_to_intervals(clearance >= 0, t, clearance)
    assert win.shape == (1, 2)
    assert win[0] == pytest.approx([30.0, 60.0], abs=1e-9)


# ---- shadow geometry --------------------------------------------------------------------------------------------------
SUN = np.array([[oe.AU_KM, 0.0, 0.0]])


def test_satellite_on_sun_side_is_fully_lit():
    assert oe.sun_visibility(np.array([[oe.R_EARTH + 600, 0.0, 0.0]]), SUN)[0] == 1.0


def test_satellite_directly_behind_earth_is_in_umbra():
    for model in ("conical", "cylindrical"):
        assert oe.sun_visibility(np.array([[-(oe.R_EARTH + 600), 0.0, 0.0]]), SUN, model)[0] == 0.0


def test_satellite_at_terminator_is_lit_and_penumbra_is_between():
    r = oe.R_EARTH + 600.0
    assert oe.sun_visibility(np.array([[0.0, r, 0.0]]), SUN)[0] == 1.0
    # walk around the orbit past the shadow edge: visibility decreases monotonically from 1 to 0 through the penumbra
    ang = np.radians(np.linspace(110, 130, 400))                       # measured from the Sun direction
    pos = np.stack([r * np.cos(ang), r * np.sin(ang), 0 * ang], axis=1)
    nu = oe.sun_visibility(pos, np.repeat(SUN, len(ang), axis=0))
    assert nu[0] == 1.0 and nu[-1] == 0.0
    assert np.all(np.diff(nu) <= 1e-12)
    assert np.any((nu > 0.01) & (nu < 0.99))


def test_cylinder_edge_is_at_asin_re_over_r():
    """The cylindrical shadow edge for an in-plane orbit is at 180 - asin(Re/r) degrees from the Sun direction."""
    r = oe.R_EARTH + 600.0
    edge_deg = 180.0 - math.degrees(math.asin(oe.R_EARTH / r))
    for ang, expect in ((edge_deg - 0.1, 1.0), (edge_deg + 0.1, 0.0)):
        a = math.radians(ang)
        assert oe.sun_visibility(np.array([[r * math.cos(a), r * math.sin(a), 0]]), SUN, "cylindrical")[0] == expect


def test_conical_umbra_is_shorter_than_cylinder_and_penumbra_is_thin():
    r = oe.R_EARTH + 600.0
    ang = np.radians(np.linspace(100, 180, 8001))
    pos = np.stack([r * np.cos(ang), r * np.sin(ang), 0 * ang], axis=1)
    sun = np.repeat(SUN, len(ang), axis=0)
    nu_con = oe.sun_visibility(pos, sun, "conical")
    nu_cyl = oe.sun_visibility(pos, sun, "cylindrical")
    umbra_start_con = ang[np.argmax(nu_con <= 0)]
    umbra_start_cyl = ang[np.argmax(nu_cyl <= 0)]
    assert umbra_start_con > umbra_start_cyl                    # umbra begins later in the cone model
    pen_width_deg = math.degrees(umbra_start_con - ang[np.argmax(nu_con < 1)])
    assert 0.2 < pen_width_deg < 0.8       # roughly the Sun's 0.53 degree disc seen through the Earth limb


def test_eclipse_duration_for_600km_orbit_with_sun_in_plane():
    """Circular equatorial orbit, Sun in the orbit plane: eclipse fraction = asin(Re/r)/180 deg  ->  about 35.5 minutes."""
    r = oe.R_EARTH + 600.0
    period = oe.orbital_period_s(r)
    expected = period * math.asin(oe.R_EARTH / r) / math.pi
    assert expected / 60.0 == pytest.approx(35.5, abs=0.1)
    t = oe.time_grid(period, 1.0)
    ang = 2 * math.pi * t / period
    pos = np.stack([r * np.cos(ang), r * np.sin(ang), 0 * ang], axis=1)
    nu = oe.sun_visibility(pos, np.repeat(SUN, len(t), axis=0), "conical")
    dark = np.sum(nu < 0.5) * 1.0
    assert dark == pytest.approx(expected, abs=10.0)           # conical shadow is shorter by a few seconds only


def _orbit_with_beta(beta_deg, n=4000):
    """Circular 600 km orbit whose plane makes angle beta with the Sun direction (+x): plane spanned by y and (cos b, 0, sin b)."""
    r, b = oe.R_EARTH + 600.0, math.radians(beta_deg)
    ang = np.linspace(0, 2 * math.pi, n)
    e2 = np.array([math.cos(b), 0.0, math.sin(b)])
    return r * (np.cos(ang)[:, None] * e2 + np.sin(ang)[:, None] * np.array([0.0, 1.0, 0.0]))


def test_eclipse_exists_below_and_vanishes_above_the_critical_beta_angle():
    """Eclipse needs beta < asin(Re/r) = 66.1 degrees for a 600 km orbit."""
    limit = math.degrees(math.asin(oe.R_EARTH / (oe.R_EARTH + 600.0)))
    assert limit == pytest.approx(66.1, abs=0.1)
    for beta, eclipsed in ((limit - 2.0, True), (limit + 2.0, False)):
        pos = _orbit_with_beta(beta)
        nu = oe.sun_visibility(pos, np.repeat(SUN, len(pos), axis=0), "cylindrical")
        assert (nu.min() == 0.0) == eclipsed


def test_eclipse_duration_shrinks_with_beta_as_the_textbook_formula_says():
    """Cylindrical shadow: half-arc = acos( sqrt(r^2 - Re^2) / (r cos beta) )."""
    r = oe.R_EARTH + 600.0
    for beta in (0.0, 30.0, 55.0):
        pos = _orbit_with_beta(beta, n=72001)
        nu = oe.sun_visibility(pos, np.repeat(SUN, len(pos), axis=0), "cylindrical")
        measured_deg = 360.0 * np.mean(nu == 0.0)
        expected_deg = 2.0 * math.degrees(math.acos(math.sqrt(r * r - oe.R_EARTH ** 2) / (r * math.cos(math.radians(beta)))))
        assert measured_deg == pytest.approx(expected_deg, abs=0.05)


def test_eclipse_state_codes():
    nu = np.array([1.0, 0.4, 0.0])
    assert list(oe.eclipse_state(nu)) == [oe.ECLIPSE_SUNLIT, oe.ECLIPSE_PENUMBRA, oe.ECLIPSE_UMBRA]


# ---- thermal / power --------------------------------------------------------------------------------------------------
def test_radiator_equilibrium_matches_stefan_boltzmann():
    cfg = oe.ThermalConfig(internal_dissipation_w=10_000, radiator_area_m2=40, emissivity=0.85, solar_absorptivity=0.15)
    r = np.array([[oe.R_EARTH + 600.0, 0, 0]])
    sun_behind = np.array([[-oe.AU_KM, 0, 0]])
    t_k, absorbed, solar = oe.thermal_and_power(r, sun_behind, np.array([0.0]), cfg)
    expected = (10_000 / (0.85 * oe.STEFAN_BOLTZMANN * 40)) ** 0.25
    assert t_k[0] == pytest.approx(expected, abs=0.01)
    assert absorbed[0] == 0.0 and solar[0] == 0.0       # eclipse: no sunlight, no array power
    t_sun, _, solar_sun = oe.thermal_and_power(r, np.array([[oe.AU_KM, 0, 0]]), np.array([1.0]), cfg)
    assert t_sun[0] > t_k[0]                            # sunlit radiator is warmer
    assert solar_sun[0] == pytest.approx(1361 * 0.30 * 60 * (1 / 1.0) , rel=0.01)


# ---- full track -------------------------------------------------------------------------------------------------------
def test_track_is_sso_altitude_and_inclination_and_stays_in_bounds():
    t = oe.time_grid(6 * 3600.0, 10.0)
    sun, g = oe.sun_position_eci(START, t), oe.gmst_rad(START, t)
    tr = oe.compute_track(oe.SatelliteConfig("A"), START, t, sun, g)
    assert 570.0 < tr.alt_km.min() and tr.alt_km.max() < 640.0          # circular at 600 km +- Earth oblateness and J2 wobble
    assert np.abs(tr.lat_deg).max() == pytest.approx(82.2, abs=0.6)     # a 97.79 degree orbit reaches 180 - 97.79 = 82.2 degrees
    assert tr.eclipse.max() == oe.ECLIPSE_UMBRA
    assert tr.beta_deg[0] == pytest.approx(-17.1, abs=1.0)


def test_skyfield_sun_cross_check_if_ephemeris_is_available():
    """Optional: compare our analytic Sun with Skyfield's DE421 ephemeris. Skipped when the file is not already on disk."""
    from pathlib import Path
    from skyfield.api import Loader
    path = Path("data/de421.bsp")
    if not path.exists():
        pytest.skip("data/de421.bsp not present (offline); optional cross-check")
    eph = Loader("data")("de421.bsp")
    ts = oe._TS
    t = ts.utc(2026, 12, 21)
    sky = (eph["earth"].at(t).observe(eph["sun"]).position.km)
    ours = oe.sun_position_eci(START, [0.0])[0]
    cosang = np.dot(sky, ours) / (np.linalg.norm(sky) * np.linalg.norm(ours))
    assert math.degrees(math.acos(min(cosang, 1.0))) < 0.5          # ICRS vs of-date differ by the ~0.37 deg precession
