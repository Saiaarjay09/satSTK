"""
orbital_engine.py - the orbit, line-of-sight, eclipse and thermal maths of satSTK.

Everything here is plain NumPy / SciPy (plus Skyfield's built-in time scale), so it runs the same on macOS, Linux and
Windows, needs no licence, and needs no internet connection.

What is in here (read top to bottom if you are new to orbital mechanics):

  1. Constants                      Earth size, gravity, the J2 "bulge" term, the Sun.
  2. Time and frames                UTC time grid, Earth rotation angle (GMST), ECI <-> ECEF, latitude/longitude/altitude.
  3. Sun position                   a short analytic formula (about 0.01 degree) so no ephemeris download is needed.
  4. Orbits                         Sun-synchronous inclination, local-time-of-ascending-node -> RAAN, and a J2 propagator.
  5. Line of sight                  does the straight line between two satellites clear the Earth ellipsoid + a 100 km
                                    atmosphere buffer?  (exact 3D geometry, with the grazing altitude).
  6. Eclipse                        conical (umbra / penumbra / sunlit) and cylindrical Earth-shadow models.
  7. Thermal and power              first-order radiator equilibrium temperature and solar-array output.
  8. Interval helpers               turn "true/false at every time step" into start/stop windows for the Gantt chart.

Frames used
  ECI  = Earth-centred inertial. The x axis points at the mean equinox *of date*, z at the pole *of date*. Precession and
         nutation are not modelled separately; because the Sun and Earth-rotation formulas below use the same "of date"
         convention the results are self-consistent.
  ECEF = Earth-centred Earth-fixed (rotates with the planet). Obtained from ECI by one rotation about z by the GMST angle.
Distances are in kilometres, speeds in km/s, angles in radians inside functions unless the name ends in `_deg`.

Honest limits: this is a high-quality teaching/design-study model, not a flight-dynamics tool. It has no atmospheric drag, no
solar-radiation pressure, no lunar/solar gravity, no polar motion or nutation, and the Earth shadow ignores atmospheric
refraction. Agreement with Ansys STK has NOT been validated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.integrate import solve_ivp
from skyfield.api import load as _skyfield_load

# ----------------------------------------------------------------------------------------------------------------------
# 1. Constants (WGS-84 / EGM-96 values)
# ----------------------------------------------------------------------------------------------------------------------
MU_EARTH = 398600.4418            # km^3/s^2   Earth's gravitational parameter
R_EARTH = 6378.137                # km         equatorial radius (WGS-84 semi-major axis)
F_EARTH = 1.0 / 298.257223563     # flattening
R_POLAR = R_EARTH * (1.0 - F_EARTH)   # km     polar radius (semi-minor axis)
E2_EARTH = F_EARTH * (2.0 - F_EARTH)  # first eccentricity squared
J2 = 1.08262668e-3                # second zonal harmonic: the equatorial bulge, the biggest perturbation to a LEO orbit
OMEGA_EARTH = 7.2921150e-5        # rad/s      Earth's rotation rate
SUN_RADIUS_KM = 695700.0
AU_KM = 149597870.7
SOLAR_CONSTANT = 1361.0           # W/m^2 at 1 AU
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4
SPEED_OF_LIGHT_KM_S = 299792.458
TROPICAL_YEAR_S = 365.2421897 * 86400.0
SSO_NODAL_RATE = 2.0 * math.pi / TROPICAL_YEAR_S   # rad/s: the node must drift east by one turn per year to follow the Sun
ATMOSPHERE_BUFFER_KM = 100.0      # a laser grazing lower than this above the surface is treated as blocked

ECLIPSE_SUNLIT, ECLIPSE_PENUMBRA, ECLIPSE_UMBRA = 0, 1, 2
ECLIPSE_NAMES = {ECLIPSE_SUNLIT: "Sunlit", ECLIPSE_PENUMBRA: "Penumbra", ECLIPSE_UMBRA: "Umbra"}

_TS = _skyfield_load.timescale()   # built-in leap-second / delta-T tables: no download needed


# ----------------------------------------------------------------------------------------------------------------------
# 2. Time and frames
# ----------------------------------------------------------------------------------------------------------------------
def as_utc(t: datetime) -> datetime:
    """Return `t` as a timezone-aware UTC datetime (naive datetimes are taken to be UTC)."""
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def time_grid(duration_s: float, step_s: float) -> np.ndarray:
    """Seconds since the start, 0, step, 2*step ... duration (both ends included)."""
    if step_s <= 0 or duration_s <= 0:
        raise ValueError("duration and step must be positive")
    n = int(round(duration_s / step_s))
    return np.arange(n + 1, dtype=float) * step_s


def _skyfield_times(start: datetime, t_s: np.ndarray):
    s = as_utc(start)
    return _TS.utc(s.year, s.month, s.day, s.hour, s.minute, s.second + s.microsecond * 1e-6 + np.asarray(t_s, dtype=float))


def gmst_rad(start: datetime, t_s) -> np.ndarray:
    """Greenwich mean sidereal time (the angle between the inertial x axis and the Greenwich meridian), radians."""
    hours = np.atleast_1d(_skyfield_times(start, np.atleast_1d(t_s)).gmst)
    return np.mod(hours * (math.pi / 12.0), 2.0 * math.pi)


def julian_days_since_j2000(start: datetime, t_s) -> np.ndarray:
    return np.atleast_1d(_skyfield_times(start, np.atleast_1d(t_s)).tt) - 2451545.0


def eci_to_ecef(r_eci: np.ndarray, gmst: np.ndarray) -> np.ndarray:
    """Rotate inertial vectors (n,3) into the Earth-fixed frame using the GMST angle (n,)."""
    c, s = np.cos(gmst), np.sin(gmst)
    out = np.empty_like(r_eci, dtype=float)
    out[..., 0] = c * r_eci[..., 0] + s * r_eci[..., 1]
    out[..., 1] = -s * r_eci[..., 0] + c * r_eci[..., 1]
    out[..., 2] = r_eci[..., 2]
    return out


def ecef_to_geodetic(r_ecef: np.ndarray):
    """Earth-fixed position (n,3) km -> geodetic latitude, longitude (degrees) and altitude above the WGS-84 ellipsoid (km)."""
    x, y, z = r_ecef[..., 0], r_ecef[..., 1], r_ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2_EARTH))             # first guess, then fixed-point iteration (converges in ~4 steps)
    for _ in range(6):
        n = R_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)
        lat = np.arctan2(z + E2_EARTH * n * np.sin(lat), p)
    n = R_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)
    cos_lat = np.cos(lat)
    alt = np.where(np.abs(cos_lat) > 1e-9, p / np.where(np.abs(cos_lat) > 1e-9, cos_lat, 1.0) - n, np.abs(z) - n * (1.0 - E2_EARTH))
    return np.degrees(lat), np.degrees(lon), alt


def geodetic_to_ecef(lat_deg, lon_deg, alt_km=0.0) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = R_EARTH / np.sqrt(1.0 - E2_EARTH * np.sin(lat) ** 2)
    return np.stack([(n + alt_km) * np.cos(lat) * np.cos(lon),
                     (n + alt_km) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - E2_EARTH) + alt_km) * np.sin(lat)], axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Sun position (analytic, no ephemeris file)
# ----------------------------------------------------------------------------------------------------------------------
def sun_position_eci(start: datetime, t_s) -> np.ndarray:
    """
    Geocentric Sun position (n,3) km in the mean-equator-of-date frame.

    Uses the low-precision formulas of the Astronomical Almanac (accurate to about 0.01 degree = 36 arc-seconds, i.e. a few
    seconds on eclipse entry/exit). Chosen so the simulator works offline; Skyfield's planetary ephemeris would need a
    multi-megabyte download on first run.
    """
    n = julian_days_since_j2000(start, t_s)
    mean_long = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)                       # mean anomaly of the Sun (really of the Earth)
    lam = mean_long + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2.0 * g)   # ecliptic longitude
    eps = np.radians(23.439 - 0.0000004 * n)                      # obliquity of the ecliptic
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
    x = np.cos(lam)
    y = np.cos(eps) * np.sin(lam)
    z = np.sin(eps) * np.sin(lam)
    return AU_KM * r_au[:, None] * np.stack([x, y, z], axis=-1)


def sun_right_ascension_deg(start: datetime) -> float:
    s = sun_position_eci(start, [0.0])[0]
    return math.degrees(math.atan2(s[1], s[0])) % 360.0


# ----------------------------------------------------------------------------------------------------------------------
# 4. Orbits
# ----------------------------------------------------------------------------------------------------------------------
def orbital_period_s(a_km: float) -> float:
    """Keplerian period 2*pi*sqrt(a^3/mu). (With J2 the true nodal period differs by about 0.1 %.)"""
    return 2.0 * math.pi * math.sqrt(a_km ** 3 / MU_EARTH)


def j2_secular_rates(a_km: float, ecc: float, inc_rad: float):
    """
    Average (secular) drift rates caused by J2, in rad/s:  (RAAN rate, argument-of-perigee rate, mean-anomaly rate).
    The RAAN rate is the "orbit plane precession": negative (westward) for prograde orbits, positive (eastward) for
    retrograde ones. A Sun-synchronous orbit is the retrograde inclination that makes it exactly +360 degrees per year.
    """
    n = math.sqrt(MU_EARTH / a_km ** 3)
    p = a_km * (1.0 - ecc ** 2)
    k = 1.5 * J2 * (R_EARTH / p) ** 2 * n
    raan_dot = -k * math.cos(inc_rad)
    argp_dot = k * (2.0 - 2.5 * math.sin(inc_rad) ** 2)
    mean_dot = n + 0.5 * k * math.sqrt(1.0 - ecc ** 2) * (3.0 * math.cos(inc_rad) ** 2 - 1.0)
    return raan_dot, argp_dot, mean_dot


def sso_inclination_deg(altitude_km: float, ecc: float = 0.0) -> float:
    """Inclination (degrees) at which J2 turns the orbit plane at the Sun's mean rate, about 0.9856 degrees/day."""
    a = R_EARTH + altitude_km
    n = math.sqrt(MU_EARTH / a ** 3)
    p = a * (1.0 - ecc ** 2)
    cos_i = -SSO_NODAL_RATE / (1.5 * J2 * (R_EARTH / p) ** 2 * n)
    if abs(cos_i) > 1.0:
        raise ValueError("no Sun-synchronous inclination exists at this altitude")
    return math.degrees(math.acos(cos_i))


def raan_for_ltan_deg(ltan_hours: float, start: datetime) -> float:
    """
    RAAN (degrees) that puts the ascending node at local (apparent) solar time `ltan_hours` on the start date.
    Local time at the node = 12 h + (node longitude - Sun longitude)/15, so RAAN = Sun right ascension + 15*(LTAN - 12).
    """
    return (sun_right_ascension_deg(start) + 15.0 * (ltan_hours - 12.0)) % 360.0


def elements_to_state(a_km, ecc, inc_rad, raan_rad, argp_rad, nu_rad):
    """Classical orbital elements -> inertial position (km) and velocity (km/s)."""
    p = a_km * (1.0 - ecc ** 2)
    r = p / (1.0 + ecc * math.cos(nu_rad))
    r_pf = np.array([r * math.cos(nu_rad), r * math.sin(nu_rad), 0.0])
    v_pf = math.sqrt(MU_EARTH / p) * np.array([-math.sin(nu_rad), ecc + math.cos(nu_rad), 0.0])
    cO, sO, ci, si, cw, sw = (math.cos(raan_rad), math.sin(raan_rad), math.cos(inc_rad), math.sin(inc_rad),
                              math.cos(argp_rad), math.sin(argp_rad))
    rot = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                    [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                    [sw * si, cw * si, ci]])
    return rot @ r_pf, rot @ v_pf


def state_to_elements(r: np.ndarray, v: np.ndarray):
    """Inertial state -> (a km, ecc, inc rad, raan rad, argp rad, true anomaly rad). Used for checks and displays."""
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(node)
    e_vec = ((vn ** 2 - MU_EARTH / rn) * r - np.dot(r, v) * v) / MU_EARTH
    ecc = np.linalg.norm(e_vec)
    a = 1.0 / (2.0 / rn - vn ** 2 / MU_EARTH)
    inc = math.acos(h[2] / hn)
    raan = math.atan2(node[1], node[0]) % (2 * math.pi) if nn > 1e-12 else 0.0
    if ecc > 1e-10 and nn > 1e-12:
        argp = math.acos(np.clip(np.dot(node, e_vec) / (nn * ecc), -1, 1))
        if e_vec[2] < 0:
            argp = 2 * math.pi - argp
    else:
        argp = 0.0
    if ecc > 1e-10:
        nu = math.acos(np.clip(np.dot(e_vec, r) / (ecc * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:   # circular: measure from the ascending node
        nu = math.acos(np.clip(np.dot(node, r) / (nn * rn), -1, 1)) if nn > 1e-12 else math.atan2(r[1], r[0])
        if nn > 1e-12 and r[2] < 0:
            nu = 2 * math.pi - nu
    return a, ecc, inc, raan, argp, nu


def _j2_two_body_rhs(_t, y, use_j2):
    x, yy, z, vx, vy, vz = y
    r2 = x * x + yy * yy + z * z
    r = math.sqrt(r2)
    mu_r3 = MU_EARTH / (r2 * r)
    ax, ay, az = -mu_r3 * x, -mu_r3 * yy, -mu_r3 * z
    if use_j2:
        k = 1.5 * J2 * (R_EARTH / r) ** 2
        z2 = 5.0 * z * z / r2
        ax *= 1.0 + k * (1.0 - z2)
        ay *= 1.0 + k * (1.0 - z2)
        az *= 1.0 + k * (3.0 - z2)
    return [vx, vy, vz, ax, ay, az]


def propagate_state(r0: np.ndarray, v0: np.ndarray, t_s: np.ndarray, use_j2: bool = True):
    """
    Integrate two-body gravity (plus the full J2 acceleration if `use_j2`) from the state at t_s[0].
    Returns positions (n,3) km and velocities (n,3) km/s at every requested time. This integrates the real J2
    acceleration (not just the averaged drift), so the short-period wobble of the orbit is included.
    """
    t_s = np.asarray(t_s, dtype=float)
    t0 = t_s[0]
    if len(t_s) == 1:
        return np.array([r0], dtype=float), np.array([v0], dtype=float)
    sol = solve_ivp(_j2_two_body_rhs, (t0, t_s[-1]), np.concatenate([r0, v0]), t_eval=t_s, args=(use_j2,),
                    method="DOP853", rtol=1e-11, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"orbit propagation failed: {sol.message}")
    return sol.y[:3].T.copy(), sol.y[3:].T.copy()


@dataclass
class SatelliteConfig:
    """
    One spacecraft. Leave `inclination_deg` as None to get the Sun-synchronous inclination for the altitude.
    Give either `ltan_hours` (local time of the ascending node, the usual way to describe a Sun-synchronous orbit) or `raan_deg`.
    `true_anomaly_deg` is where on the orbit the satellite starts (0 = at the ascending node when ecc = 0).
    """
    name: str
    altitude_km: float = 600.0
    ecc: float = 0.0
    inclination_deg: float | None = None
    ltan_hours: float | None = 10.5
    raan_deg: float | None = None
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    def initial_elements(self, start: datetime):
        a = R_EARTH + self.altitude_km
        inc = self.inclination_deg if self.inclination_deg is not None else sso_inclination_deg(self.altitude_km, self.ecc)
        if self.raan_deg is not None:
            raan = self.raan_deg
        elif self.ltan_hours is not None:
            raan = raan_for_ltan_deg(self.ltan_hours, start)
        else:
            raise ValueError("give ltan_hours or raan_deg")
        return a, self.ecc, math.radians(inc), math.radians(raan), math.radians(self.arg_perigee_deg), \
            math.radians(self.true_anomaly_deg)


# ----------------------------------------------------------------------------------------------------------------------
# 5. Line of sight against the Earth ellipsoid + atmosphere buffer
# ----------------------------------------------------------------------------------------------------------------------
def line_of_sight(r1: np.ndarray, r2: np.ndarray, buffer_km: float = ATMOSPHERE_BUFFER_KM):
    """
    Exact 3D visibility between two points (arrays (n,3), km, ECI or ECEF - both share the same polar axis).

    The blocker is the WGS-84 ellipsoid grown by `buffer_km` on both axes. Squashing the z axis by a_eff/b_eff turns that
    ellipsoid into a sphere of radius a_eff, so "does the segment hit the ellipsoid?" becomes "does the closest point of the
    segment to the origin lie inside a sphere?" - a closed-form calculation, no sampling along the beam.

    Returns (visible, clearance_km):  clearance is the closest approach of the beam to the blocker surface, in the squashed
    space (positive = clear; accurate to ~0.3 % of itself and exactly right in sign). Negative values mean the Earth blocks it.
    """
    a_eff = R_EARTH + buffer_km
    b_eff = R_POLAR + buffer_km
    squash = np.array([1.0, 1.0, a_eff / b_eff])
    p1 = np.atleast_2d(r1) * squash
    p2 = np.atleast_2d(r2) * squash
    d = p2 - p1
    dd = np.einsum("ij,ij->i", d, d)
    s = np.where(dd > 0, -np.einsum("ij,ij->i", p1, d) / np.where(dd > 0, dd, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    closest = p1 + s[:, None] * d
    clearance = np.linalg.norm(closest, axis=1) - a_eff
    return clearance >= 0.0, clearance


# ----------------------------------------------------------------------------------------------------------------------
# 5b. Ground station (the fixed origin of the laser beam)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class GroundStation:
    """
    A laser ground station at a fixed place on the Earth. The beam starts here and is pointed at the satellite.
    `min_elevation_deg` is the lowest angle above the horizon at which the beam may be used (optical stations normally need 15-30
    degrees: below that the beam crosses too much air and, in practice, trees, buildings and hills).
    """
    name: str = "GS-INDIA"
    lat_deg: float = 13.0             # default: the South India ground station (near Bengaluru); change it freely
    lon_deg: float = 77.5
    alt_m: float = 900.0              # assumed: the Deccan plateau; edit to your site
    min_elevation_deg: float = 10.0

    def ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m / 1000.0)

    def enu_matrix(self) -> np.ndarray:
        """Rows are the East, North and Up unit vectors (in ECEF) at the station."""
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        return np.array([[-math.sin(lon), math.cos(lon), 0.0],
                         [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
                         [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]])


def ecef_to_eci(r_ecef: np.ndarray, gmst: np.ndarray) -> np.ndarray:
    """Inverse of `eci_to_ecef`."""
    c, s = np.cos(gmst), np.sin(gmst)
    out = np.empty_like(r_ecef, dtype=float)
    out[..., 0] = c * r_ecef[..., 0] - s * r_ecef[..., 1]
    out[..., 1] = s * r_ecef[..., 0] + c * r_ecef[..., 1]
    out[..., 2] = r_ecef[..., 2]
    return out


def station_state_eci(gs: GroundStation, gmst: np.ndarray):
    """Station position (n,3) km and velocity (n,3) km/s in the inertial frame (the Earth carries it round at 0.46 km/s)."""
    r = ecef_to_eci(np.repeat(gs.ecef()[None, :], len(gmst), axis=0), gmst)
    v = np.stack([-OMEGA_EARTH * r[:, 1], OMEGA_EARTH * r[:, 0], np.zeros(len(gmst))], axis=1)
    return r, v


@dataclass
class Topocentric:
    """Where the satellite is as seen from the station: the pointing angles the ground telescope needs and how fast they change."""
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray             # clockwise from true north
    range_km: np.ndarray
    elevation_rate_deg_s: np.ndarray
    azimuth_rate_deg_s: np.ndarray
    slew_rate_deg_s: np.ndarray


def topocentric_state(gs: GroundStation, r_ecef: np.ndarray, v_ecef: np.ndarray) -> Topocentric:
    """
    Elevation, azimuth, range and (analytic) angular rates of a satellite seen from the station.
    `v_ecef` is the satellite velocity measured in the rotating Earth frame (v_inertial rotated, minus omega x r).
    """
    enu = gs.enu_matrix()
    d = (np.atleast_2d(r_ecef) - gs.ecef()) @ enu.T
    v = np.atleast_2d(v_ecef) @ enu.T
    rng = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    u = d / rng[:, None]
    u_dot = (v - np.einsum("ij,ij->i", v, u)[:, None] * u) / rng[:, None]
    e, n, up = u[:, 0], u[:, 1], u[:, 2]
    horiz2 = np.maximum(e * e + n * n, 1e-18)
    az = np.degrees(np.arctan2(e, n)) % 360.0
    el = np.degrees(np.arcsin(np.clip(up, -1, 1)))
    az_rate = np.degrees((n * u_dot[:, 0] - e * u_dot[:, 1]) / horiz2)
    el_rate = np.degrees(u_dot[:, 2] / np.sqrt(np.maximum(1 - up * up, 1e-18)))
    return Topocentric(el, az, rng, el_rate, az_rate, np.degrees(np.linalg.norm(u_dot, axis=1)))


def sun_elevation_deg(gs: GroundStation, sun_eci: np.ndarray, gmst: np.ndarray) -> np.ndarray:
    """Elevation of the Sun above the station's horizon (negative = night); tells you whether the sky is dark."""
    s_ecef = eci_to_ecef(sun_eci / np.linalg.norm(sun_eci, axis=1)[:, None], gmst)
    return np.degrees(np.arcsin(np.clip(s_ecef @ gs.enu_matrix()[2], -1, 1)))


def horizon_cap_radius_deg(altitude_km: float, min_elevation_deg: float = 0.0) -> float:
    """Ground radius (great-circle degrees) within which a satellite at `altitude_km` is above `min_elevation_deg`."""
    r = R_EARTH + altitude_km
    el = math.radians(min_elevation_deg)
    return math.degrees(math.acos(R_EARTH * math.cos(el) / r) - el)


# ----------------------------------------------------------------------------------------------------------------------
# 5c. Geostationary relay satellites
# ----------------------------------------------------------------------------------------------------------------------
SIDEREAL_DAY_S = 23 * 3600 + 56 * 60 + 4        # 23 h 56 min 04 s: one turn of the Earth relative to the stars
GEO_RADIUS_KM = (MU_EARTH * (SIDEREAL_DAY_S / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)    # 42,164 km from the Earth's centre
GEO_ALTITUDE_KM = GEO_RADIUS_KM - R_EARTH       # about 35,786 km above the equator


@dataclass
class GeoRelayConfig:
    """
    One geostationary relay satellite: equatorial (0 degree inclination), 35,786 km up, one orbit per sidereal day, so it hangs at a fixed
    longitude over the equator. Real GEO satellites hold that spot with small thruster burns ("station keeping"); the simulator
    idealises this as a perfectly fixed point in the Earth-fixed frame.
    """
    name: str
    short: str
    lon_deg: float                      # east longitude of the fixed point on the equator


def default_geo_relays() -> list:
    """The three-node ring: Indian Ocean (above the South India station), Atlantic, Pacific. Spacing is about 120 degrees."""
    return [GeoRelayConfig("GEO-Relay-IND", "GEO-IND", 77.5),
            GeoRelayConfig("GEO-Relay-ATL", "GEO-ATL", 315.0 - 360.0),       # 315 E = 45 W
            GeoRelayConfig("GEO-Relay-PAC", "GEO-PAC", 195.0 - 360.0)]       # 195 E = 165 W


@dataclass
class RelayTrack:
    name: str
    short: str
    lon_deg: float
    r_eci: np.ndarray            # (n,3) km
    v_eci: np.ndarray            # (n,3) km/s
    r_ecef: np.ndarray           # (n,3) km, constant


def geo_relay_track(cfg: GeoRelayConfig, gmst: np.ndarray) -> RelayTrack:
    """Position and velocity of a geostationary satellite: fixed in the Earth frame, carried round by the Earth's rotation in the inertial frame."""
    lon = math.radians(cfg.lon_deg)
    r_fixed = GEO_RADIUS_KM * np.array([math.cos(lon), math.sin(lon), 0.0])
    r_ecef = np.repeat(r_fixed[None, :], len(gmst), axis=0)
    r_eci = ecef_to_eci(r_ecef, gmst)
    v_eci = np.stack([-OMEGA_EARTH * r_eci[:, 1], OMEGA_EARTH * r_eci[:, 0], np.zeros(len(gmst))], axis=1)
    return RelayTrack(cfg.name, cfg.short, cfg.lon_deg, r_eci, v_eci, r_ecef)


# ----------------------------------------------------------------------------------------------------------------------
# 6. Eclipse (Earth shadow)
# ----------------------------------------------------------------------------------------------------------------------
def sun_visibility(r_sat: np.ndarray, r_sun: np.ndarray, model: str = "conical") -> np.ndarray:
    """
    Fraction of the Sun's disc visible from the satellite: 1 = full sunlight, 0 = umbra, in between = penumbra.

    model="conical"      the Earth and the Sun are discs of different apparent size; the overlap area of the two discs gives the
                         visible fraction (Montenbruck & Gill, Satellite Orbits, section 3.4.2). Produces a real penumbra.
    model="cylindrical"  the shadow is a cylinder of Earth's radius pointing away from the Sun: only sunlit or umbra.
    """
    r_sat = np.atleast_2d(r_sat)
    r_sun = np.atleast_2d(r_sun)
    if model == "cylindrical":
        s_hat = r_sun / np.linalg.norm(r_sun, axis=1)[:, None]
        along = np.einsum("ij,ij->i", r_sat, s_hat)
        perp = np.linalg.norm(r_sat - along[:, None] * s_hat, axis=1)
        return np.where((along < 0.0) & (perp < R_EARTH), 0.0, 1.0)
    if model != "conical":
        raise ValueError("model must be 'conical' or 'cylindrical'")
    to_sun = r_sun - r_sat
    d_sun = np.linalg.norm(to_sun, axis=1)
    d_sat = np.linalg.norm(r_sat, axis=1)
    a = np.arcsin(np.clip(SUN_RADIUS_KM / d_sun, -1, 1))      # apparent angular radius of the Sun
    b = np.arcsin(np.clip(R_EARTH / d_sat, -1, 1))            # apparent angular radius of the Earth
    cosc = np.einsum("ij,ij->i", -r_sat, to_sun) / (d_sat * d_sun)
    c = np.arccos(np.clip(cosc, -1, 1))                        # angular separation of the two disc centres
    nu = np.ones_like(c)
    umbra = c <= (b - a)
    annular = (c <= (a - b)) & ~umbra
    partial = (c < (a + b)) & ~umbra & ~annular
    nu[umbra] = 0.0
    nu[annular] = 1.0 - (b[annular] / a[annular]) ** 2
    ap, bp, cp = a[partial], b[partial], c[partial]
    x = (cp ** 2 + ap ** 2 - bp ** 2) / (2.0 * cp)
    y = np.sqrt(np.maximum(ap ** 2 - x ** 2, 0.0))
    area = ap ** 2 * np.arccos(np.clip(x / ap, -1, 1)) + bp ** 2 * np.arccos(np.clip((cp - x) / bp, -1, 1)) - cp * y
    nu[partial] = 1.0 - area / (math.pi * ap ** 2)
    return np.clip(nu, 0.0, 1.0)


def eclipse_state(nu: np.ndarray) -> np.ndarray:
    """Sun-visibility fraction -> integer code: 0 sunlit, 1 penumbra, 2 umbra."""
    nu = np.asarray(nu)
    return np.where(nu >= 1.0 - 1e-9, ECLIPSE_SUNLIT, np.where(nu <= 1e-9, ECLIPSE_UMBRA, ECLIPSE_PENUMBRA)).astype(np.int8)


def beta_angle_deg(r: np.ndarray, v: np.ndarray, sun_eci: np.ndarray) -> np.ndarray:
    """Angle between the Sun direction and the orbit plane. Beyond ~66 degrees at 600 km the satellite never enters eclipse."""
    h = np.cross(r, v)
    h /= np.linalg.norm(h, axis=1)[:, None]
    s = sun_eci / np.linalg.norm(sun_eci, axis=1)[:, None]
    return np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", h, s), -1, 1)))


# ----------------------------------------------------------------------------------------------------------------------
# 7. Thermal and power
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ThermalConfig:
    """
    First-order spacecraft model. All numbers are placeholders to edit, not data about a real vehicle.

    Radiator: a flat plate that faces zenith (away from the Earth), so it sees deep space and no Earth infrared. In
    equilibrium   epsilon*sigma*A*(T^4 - T_space^4) = P_internal + alpha*S*A*(sun visibility)*max(0, n.s_hat).
    This ignores thermal mass, so temperatures jump instantly at eclipse entry/exit; real hardware lags by minutes.
    """
    internal_dissipation_w: float = 10_000.0   # waste heat from the compute payload that the radiator must reject
    radiator_area_m2: float = 40.0
    emissivity: float = 0.85                   # infrared emissivity of the radiator coating
    solar_absorptivity: float = 0.15           # fraction of sunlight the coating absorbs (white paint / optical solar reflector)
    space_temperature_k: float = 3.0
    solar_array_area_m2: float = 60.0
    solar_array_efficiency: float = 0.30       # end-of-life cell efficiency; array assumed to track the Sun (cos loss = 1)


def thermal_and_power(r_sat: np.ndarray, sun_eci: np.ndarray, nu: np.ndarray, cfg: ThermalConfig):
    """Returns (radiator equilibrium temperature in kelvin, absorbed solar power on radiator in W, solar-array output in W)."""
    d_sun = np.linalg.norm(sun_eci, axis=1)
    flux = SOLAR_CONSTANT * (AU_KM / d_sun) ** 2
    r_hat = r_sat / np.linalg.norm(r_sat, axis=1)[:, None]
    s_hat = sun_eci / d_sun[:, None]
    cos_inc = np.clip(np.einsum("ij,ij->i", r_hat, s_hat), 0.0, None)
    absorbed = cfg.solar_absorptivity * flux * nu * cos_inc * cfg.radiator_area_m2
    heat = cfg.internal_dissipation_w + absorbed
    t4 = heat / (cfg.emissivity * STEFAN_BOLTZMANN * cfg.radiator_area_m2) + cfg.space_temperature_k ** 4
    solar_w = flux * nu * cfg.solar_array_efficiency * cfg.solar_array_area_m2
    return t4 ** 0.25, absorbed, solar_w


# ----------------------------------------------------------------------------------------------------------------------
# One satellite's full time history
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SatelliteTrack:
    name: str
    r_eci: np.ndarray            # (n,3) km
    v_eci: np.ndarray            # (n,3) km/s
    r_ecef: np.ndarray           # (n,3) km
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    alt_km: np.ndarray
    sun_visibility: np.ndarray   # 0..1
    eclipse: np.ndarray          # 0 sunlit / 1 penumbra / 2 umbra
    beta_deg: np.ndarray
    radiator_temp_c: np.ndarray
    solar_power_w: np.ndarray
    radiator_solar_load_w: np.ndarray
    elements0: tuple = field(default_factory=tuple)


def compute_track(cfg: SatelliteConfig, start: datetime, t_s: np.ndarray, sun_eci: np.ndarray, gmst: np.ndarray,
                  thermal: ThermalConfig | None = None, shadow_model: str = "conical", use_j2: bool = True) -> SatelliteTrack:
    thermal = thermal or ThermalConfig()
    a, e, i, raan, argp, nu0 = cfg.initial_elements(start)
    r0, v0 = elements_to_state(a, e, i, raan, argp, nu0)
    r, v = propagate_state(r0, v0, t_s, use_j2=use_j2)
    r_ecef = eci_to_ecef(r, gmst)
    lat, lon, alt = ecef_to_geodetic(r_ecef)
    nu = sun_visibility(r, sun_eci, shadow_model)
    t_k, absorbed, solar_w = thermal_and_power(r, sun_eci, nu, thermal)
    return SatelliteTrack(cfg.name, r, v, r_ecef, lat, lon, alt, nu, eclipse_state(nu), beta_angle_deg(r, v, sun_eci),
                          t_k - 273.15, solar_w, absorbed, (a, e, i, raan, argp, nu0))


# ----------------------------------------------------------------------------------------------------------------------
# 8. Interval helpers
# ----------------------------------------------------------------------------------------------------------------------
def mask_to_intervals(mask: np.ndarray, t_s: np.ndarray, level: np.ndarray | None = None) -> np.ndarray:
    """
    Turn a boolean time series into an (m,2) array of [start, end] times in seconds.

    Without `level`, edges are placed halfway between the last sample of one state and the first of the next (error at most
    half a time step). With `level` - a continuous quantity that is >= 0 exactly where `mask` is True, such as the
    beam-to-Earth clearance - the edge is found by linear interpolation of the zero crossing, which is much sharper.
    """
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if n == 0:
        return np.zeros((0, 2))
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)          # first True index
    ends = np.flatnonzero(edges == -1) - 1       # last True index

    def crossing(i_false, i_true):
        if level is None:
            return 0.5 * (t_s[i_false] + t_s[i_true])
        f0, f1 = level[i_false], level[i_true]
        if f1 == f0:
            return 0.5 * (t_s[i_false] + t_s[i_true])
        frac = np.clip(f0 / (f0 - f1), 0.0, 1.0)
        return t_s[i_false] + frac * (t_s[i_true] - t_s[i_false])

    out = []
    for s, e in zip(starts, ends):
        t_start = t_s[0] if s == 0 else crossing(s - 1, s)
        t_end = t_s[-1] if e == n - 1 else crossing(e + 1, e)
        out.append((t_start, t_end))
    return np.array(out, dtype=float).reshape(-1, 2)


def state_intervals(state: np.ndarray, t_s: np.ndarray, code: int) -> np.ndarray:
    return mask_to_intervals(np.asarray(state) == code, t_s)
