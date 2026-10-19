"""
atmosphere.py - what the air does to a 1550 nm laser beam between a ground station and a satellite.

Between two satellites there is nothing in the way. Between the ground and a satellite the beam must cross the whole atmosphere, and
three things happen to it:

  1. It is ABSORBED AND SCATTERED (extinction): by gas molecules, by haze/aerosol, and - by far the biggest effect - by cloud, fog and rain.
     The loss grows as the beam gets closer to the horizon because the path through the air is longer.
  2. It is DISTORTED by turbulence (the same effect that makes stars twinkle): the beam wobbles, spreads slightly, and the received power
     flickers ("scintillation"). We report the fade depth you must allow for to keep the link up 99.9 % of the time.
  3. Anything else in the way (a bird, an aircraft, an ash cloud) is handled in events.py.

Models used (all are standard, simple, and free of licences):
  * Aerosol extinction: the Kim/Kruse visibility model, scaled with a 1.2 km aerosol scale height and the Kasten-Young air mass.
  * Cloud, fog, rain: you give the loss straight overhead (zenith) in dB; it scales with 1/sin(elevation). Rain uses the common optical
    rule of thumb gamma = 1.076 R^0.67 dB/km (Carbonneau & Wisely) over a 2 km rain layer.
  * Turbulence: Hufnagel-Valley (HV 5/7 style) Cn^2 profile -> Fried parameter r0, Rytov variance, log-normal scintillation fade.

Honest limits: these are engineering rules of thumb for exploring "what if", not weather forecasts or a validated channel model. The
weather presets are illustrative round numbers. Real sites vary enormously (a mountain-top station like Teide sits above most of the aerosol and
turbulence; the preset ignores everything about local climate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

AEROSOL_SCALE_HEIGHT_KM = 1.2
RAIN_LAYER_KM = 2.0
MOLECULAR_ZENITH_DB = 0.05          # clear-air gas absorption at 1550 nm, straight up, at sea level
MIN_MODEL_ELEVATION_DEG = 3.0       # the flat-layer formulas are not trusted below this; losses are evaluated at this elevation


@dataclass
class AtmosphereConfig:
    """The conditions of the sky above the station. Every number can be changed live in the Conditions panel."""
    visibility_km: float = 40.0         # meteorological visibility at 550 nm: 40 = clear, 10 = light haze, 2 = thick haze
    cloud_loss_db: float = 0.0          # extra loss straight up: ~1-3 thin cirrus, ~10-30 thick cloud (effectively a blocked beam)
    fog_loss_db: float = 0.0            # extra loss straight up through a fog layer: ~5 light fog, 30+ dense fog
    rain_mm_h: float = 0.0              # rain rate: 1 drizzle, 10 moderate, 50 downpour
    turbulence_cn2_ground: float = 1.7e-14   # Hufnagel-Valley ground-level strength A (m^-2/3): 1e-15 calm night, 1.7e-14 typical, 1e-13 hot afternoon
    wind_rms_m_s: float = 21.0          # high-altitude wind speed for the Hufnagel-Valley profile
    label: str = "Clear"


WEATHER_PRESETS = {
    "Clear":          AtmosphereConfig(visibility_km=40.0, label="Clear"),
    "Hazy":           AtmosphereConfig(visibility_km=6.0, label="Hazy"),
    "Thin cirrus":    AtmosphereConfig(visibility_km=30.0, cloud_loss_db=2.0, label="Thin cirrus"),
    "Light fog":      AtmosphereConfig(visibility_km=3.0, fog_loss_db=8.0, label="Light fog"),
    "Dense fog":      AtmosphereConfig(visibility_km=0.5, fog_loss_db=35.0, label="Dense fog"),
    "Overcast":       AtmosphereConfig(visibility_km=15.0, cloud_loss_db=25.0, label="Overcast"),
    "Rain":           AtmosphereConfig(visibility_km=8.0, cloud_loss_db=8.0, rain_mm_h=10.0, label="Rain"),
    "Thunderstorm":   AtmosphereConfig(visibility_km=4.0, cloud_loss_db=30.0, rain_mm_h=40.0, turbulence_cn2_ground=5e-14, label="Thunderstorm"),
    "Hot afternoon":  AtmosphereConfig(visibility_km=25.0, turbulence_cn2_ground=1e-13, label="Hot afternoon"),
}


def air_mass(elevation_deg):
    """Relative length of the path through the atmosphere (1 straight up, ~38 at the horizon): Kasten & Young (1989)."""
    el = np.maximum(np.asarray(elevation_deg, dtype=float), MIN_MODEL_ELEVATION_DEG)
    return 1.0 / (np.sin(np.radians(el)) + 0.50572 * (el + 6.07995) ** -1.6364)


def kim_exponent(visibility_km: float) -> float:
    v = visibility_km
    if v > 50:
        return 1.6
    if v > 6:
        return 1.3
    if v > 1:
        return 0.16 * v + 0.34
    if v > 0.5:
        return v - 0.5
    return 0.0


def aerosol_extinction_per_km(visibility_km: float, wavelength_nm: float = 1550.0) -> float:
    """Extinction coefficient (per km) near the ground from the Kim model: (3.912/V) (lambda/550 nm)^-q."""
    v = max(visibility_km, 0.05)
    return 3.912 / v * (wavelength_nm / 550.0) ** (-kim_exponent(v))


def clear_air_loss_db(elevation_deg, station_alt_m: float, cfg: AtmosphereConfig, wavelength_nm: float = 1550.0):
    """Molecules + aerosol/haze. A high-altitude station sits above most of the aerosol, so its loss is much smaller."""
    m = air_mass(elevation_deg)
    h_km = station_alt_m / 1000.0
    aerosol_zenith_db = 4.343 * aerosol_extinction_per_km(cfg.visibility_km, wavelength_nm) * AEROSOL_SCALE_HEIGHT_KM * math.exp(-h_km / AEROSOL_SCALE_HEIGHT_KM)
    molecular_zenith_db = MOLECULAR_ZENITH_DB * math.exp(-h_km / 8.0)
    return m * (aerosol_zenith_db + molecular_zenith_db)


def weather_loss_db(elevation_deg, cfg: AtmosphereConfig):
    """Cloud + fog + rain along the slant path (thin layers: path scales as 1/sin(elevation))."""
    el = np.maximum(np.asarray(elevation_deg, dtype=float), MIN_MODEL_ELEVATION_DEG)
    path = 1.0 / np.sin(np.radians(el))
    rain_zenith = 1.076 * cfg.rain_mm_h ** 0.67 * RAIN_LAYER_KM if cfg.rain_mm_h > 0 else 0.0
    return path * (cfg.cloud_loss_db + cfg.fog_loss_db + rain_zenith)


# ----------------------------------------------------------------------------------------------------------------------
# Turbulence
# ----------------------------------------------------------------------------------------------------------------------
def hufnagel_valley_cn2(h_asl_m, station_alt_m: float, cfg: AtmosphereConfig):
    """Refractive-index structure constant Cn^2 (m^-2/3) at height above sea level (HV model, ground term measured from the site)."""
    h = np.asarray(h_asl_m, dtype=float)
    above_ground = np.maximum(h - station_alt_m, 0.0)
    w = cfg.wind_rms_m_s
    return (0.00594 * (w / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
            + 2.7e-16 * np.exp(-h / 1500.0)
            + cfg.turbulence_cn2_ground * np.exp(-above_ground / 100.0))


def turbulence_integrals(station_alt_m: float, cfg: AtmosphereConfig):
    """Zenith integrals of Cn^2 and of Cn^2 * h^(5/6) over the 30 km above the station (h measured from the station)."""
    h = np.concatenate([np.linspace(0.0, 200.0, 401), np.linspace(200.0, 30000.0, 6000)[1:]])
    cn2 = hufnagel_valley_cn2(station_alt_m + h, station_alt_m, cfg)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(cn2, h)), float(trapz(cn2 * h ** (5.0 / 6.0), h))


def fried_parameter_m(elevation_deg, station_alt_m: float, cfg: AtmosphereConfig, wavelength_nm: float = 1550.0):
    """Fried parameter r0: the width of a patch of air over which the wavefront stays flat. Smaller = worse turbulence."""
    k = 2.0 * math.pi / (wavelength_nm * 1e-9)
    i0, _ = turbulence_integrals(station_alt_m, cfg)
    r0_zenith = (0.423 * k * k * i0) ** (-3.0 / 5.0)
    el = np.maximum(np.asarray(elevation_deg, dtype=float), MIN_MODEL_ELEVATION_DEG)
    return r0_zenith * np.sin(np.radians(el)) ** (3.0 / 5.0)


def rytov_variance(elevation_deg, station_alt_m: float, cfg: AtmosphereConfig, wavelength_nm: float = 1550.0):
    """Rytov variance (plane-wave form): the strength of the scintillation. Below about 1 the fluctuations are 'weak'."""
    k = 2.0 * math.pi / (wavelength_nm * 1e-9)
    _, i56 = turbulence_integrals(station_alt_m, cfg)
    el = np.maximum(np.asarray(elevation_deg, dtype=float), MIN_MODEL_ELEVATION_DEG)
    return 2.25 * k ** (7.0 / 6.0) * (1.0 / np.sin(np.radians(el))) ** (11.0 / 6.0) * i56


def scintillation_index(rytov_var):
    """Normalised intensity variance for weak-to-strong turbulence (Andrews & Phillips, plane wave, point receiver)."""
    s = np.asarray(rytov_var, dtype=float)
    return np.exp(0.49 * s / (1.0 + 1.11 * s ** (6.0 / 5.0)) ** (7.0 / 6.0) + 0.51 * s / (1.0 + 0.69 * s ** (6.0 / 5.0)) ** (5.0 / 6.0)) - 1.0


def scintillation_fade_db(scint_index, outage_probability: float = 1e-3):
    """Fade depth (dB, positive) that the received power falls below only `outage_probability` of the time (log-normal fading)."""
    from scipy.stats import norm
    sigma2 = np.log1p(np.asarray(scint_index, dtype=float))
    z = -norm.ppf(outage_probability)                       # about 3.09 for 0.1 %
    fade_ln = z * np.sqrt(sigma2) + 0.5 * sigma2
    return 10.0 * fade_ln / math.log(10.0)


def channel(elevation_deg, station_alt_m: float, cfg: AtmosphereConfig, wavelength_nm: float = 1550.0) -> dict:
    """Everything the atmosphere does, as arrays over elevation (use elevation below the mask only for display, not for links)."""
    el = np.asarray(elevation_deg, dtype=float)
    r0 = fried_parameter_m(el, station_alt_m, cfg, wavelength_nm)
    ryt = rytov_variance(el, station_alt_m, cfg, wavelength_nm)
    si = scintillation_index(ryt)
    return {
        "clear_air_loss_db": clear_air_loss_db(el, station_alt_m, cfg, wavelength_nm),
        "weather_loss_db": weather_loss_db(el, cfg),
        "fried_r0_m": r0,
        "rytov_variance": ryt,
        "scintillation_index": si,
        "scintillation_fade_db": scintillation_fade_db(si),
        "beam_spread_urad": wavelength_nm * 1e-9 / r0 * 1e6,       # extra full-angle spreading of an uplink beam by turbulence
    }
