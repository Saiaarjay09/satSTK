"""
simulation.py - glue between the physics engines: describe a scenario, run it, get one result object.

    cfg = SimulationConfig()                  # one 600 km Sun-synchronous satellite, a ground station, clear sky, 24 h, 10 s steps
    result = run_simulation(cfg)              # a second or two on a laptop
    print(result.summary_text())

The scenario is: a laser GROUND STATION at a fixed place (you choose it) aims its beam at a satellite. The beam crosses the atmosphere
(clear air, haze, cloud, turbulence) and may be interrupted by random events (birds, aircraft, storms ...). With two or more satellites
you can pick which one the beam tracks; satellite-to-satellite crosslinks between them are also computed.

No GUI code lives here, so the same result object drives the desktop app, the headless command line and the tests.
Nothing is written to disk unless you ask (`write_csv`).
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

import atmosphere as atm
import events as ev
import orbital_engine as oe
from laser_link_engine import LaserLinkConfig, LinkResult, MODULATIONS, compute_ground_link, compute_link

DEFAULT_START = datetime(2026, 12, 21, 0, 0, 0, tzinfo=timezone.utc)   # northern winter solstice: long eclipses for a morning orbit

GROUND_STATION_PRESETS = {
    "Teide Observatory, Tenerife (ESA OGS)": oe.GroundStation("Teide OGS", 28.30, -16.51, 2390.0),
    "Table Mountain, California (JPL OCTL)": oe.GroundStation("Table Mountain", 34.38, -117.68, 2280.0),
    "Haleakala, Hawaii": oe.GroundStation("Haleakala", 20.71, -156.26, 3055.0),
    "Hanle, Ladakh, India (high desert)": oe.GroundStation("Hanle", 32.78, 78.96, 4500.0),
    "Bengaluru, India": oe.GroundStation("Bengaluru", 12.97, 77.59, 920.0),
    "Hyderabad, India": oe.GroundStation("Hyderabad", 17.39, 78.49, 540.0),
    "Cerro Paranal, Chile": oe.GroundStation("Paranal", -24.63, -70.40, 2635.0),
    "Alice Springs, Australia": oe.GroundStation("Alice Springs", -23.70, 133.88, 545.0),
    "Svalbard, Norway (polar)": oe.GroundStation("Svalbard", 78.23, 15.41, 450.0),
    "Cape Town, South Africa": oe.GroundStation("Cape Town", -33.93, 18.42, 25.0),
}


def default_satellites() -> list[oe.SatelliteConfig]:
    """One 600 km Sun-synchronous satellite (local node time 10:30): the optical datacentre node the beam is aimed at."""
    return [oe.SatelliteConfig("Orbital-Node-Alpha", altitude_km=600.0, ltan_hours=10.5, true_anomaly_deg=0.0)]


def second_satellite() -> oe.SatelliteConfig:
    """Optional second node in a different orbit plane (local node time 13:30): a second target, or a crosslink partner."""
    return oe.SatelliteConfig("Orbital-Node-Beta", altitude_km=600.0, ltan_hours=13.5, true_anomaly_deg=45.0)


def two_satellites() -> list[oe.SatelliteConfig]:
    return default_satellites() + [second_satellite()]


@dataclass
class SimulationConfig:
    start: datetime = DEFAULT_START
    duration_h: float = 24.0
    step_s: float = 10.0                       # 1.0 for the finest result, 10.0 for a fast one
    satellites: list = field(default_factory=default_satellites)
    ground: oe.GroundStation = field(default_factory=oe.GroundStation)
    atmosphere: atm.AtmosphereConfig = field(default_factory=atm.AtmosphereConfig)
    events: ev.EventsConfig = field(default_factory=ev.EventsConfig)
    thermal: oe.ThermalConfig = field(default_factory=oe.ThermalConfig)
    link: LaserLinkConfig = field(default_factory=LaserLinkConfig)
    target_index: int = 0                      # which satellite the ground beam tracks
    shadow_model: str = "conical"              # or "cylindrical"
    los_buffer_km: float = oe.ATMOSPHERE_BUFFER_KM   # for satellite-to-satellite crosslinks
    use_j2: bool = True

    def validate(self) -> None:
        if len(self.satellites) < 1:
            raise ValueError("a scenario needs at least one satellite")
        if not (0 <= self.target_index < len(self.satellites)):
            raise ValueError("target satellite does not exist")
        if not (0 < self.step_s <= 600):
            raise ValueError("step must be between 0 and 600 seconds")
        if not (0 < self.duration_h <= 24 * 14):
            raise ValueError("duration must be between 0 and 14 days")
        if self.shadow_model not in ("conical", "cylindrical"):
            raise ValueError("shadow_model must be 'conical' or 'cylindrical'")
        if not (-90.0 <= self.ground.lat_deg <= 90.0 and -180.0 <= self.ground.lon_deg <= 180.0):
            raise ValueError("ground station latitude must be within +-90 and longitude within +-180 degrees")
        if not (0.0 <= self.ground.min_elevation_deg < 90.0):
            raise ValueError("minimum elevation must be between 0 and 90 degrees")

    # JSON-friendly round trip, for saving a scenario by hand if you wish
    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = oe.as_utc(self.start).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        d = dict(d)
        start = datetime.fromisoformat(d.pop("start")) if "start" in d else DEFAULT_START
        sats = [oe.SatelliteConfig(**s) for s in d.pop("satellites", [asdict(s) for s in default_satellites()])]
        events_d = dict(d.pop("events", {}))
        manual = [ev.ManualEvent(**m) for m in events_d.pop("manual", [])]
        return cls(start=oe.as_utc(start), satellites=sats, ground=oe.GroundStation(**d.pop("ground", {})),
                   atmosphere=atm.AtmosphereConfig(**d.pop("atmosphere", {})), events=ev.EventsConfig(manual=manual, **events_d),
                   thermal=oe.ThermalConfig(**d.pop("thermal", {})), link=LaserLinkConfig(**d.pop("link", {})), **d)


@dataclass
class SimulationResult:
    config: SimulationConfig
    t_s: np.ndarray
    sun_eci: np.ndarray
    gmst: np.ndarray
    tracks: list
    station_eci: np.ndarray           # (n,3) station position, inertial frame
    station_sun_elevation_deg: np.ndarray
    topo: dict                        # {sat_index: Topocentric}
    ground_links: dict                # {sat_index: LinkResult}: the beam from the station to each satellite
    access_windows: dict              # {sat_index: (m,2) seconds}: when the beam is on the satellite (above the mask, no outage)
    crosslinks: dict                  # {(i, j): LinkResult} between satellites (empty with one satellite)
    crosslink_windows: dict
    eclipse_windows: list             # per satellite: {"penumbra": (m,2), "umbra": (m,2), "sunlit": (m,2)}
    events: list                      # list of events.Event
    effects: ev.EventEffects

    @property
    def target(self) -> int:
        return self.config.target_index

    @property
    def primary_link(self) -> LinkResult:
        return self.ground_links[self.target]

    @property
    def primary_windows(self) -> np.ndarray:
        return self.access_windows[self.target]

    @property
    def n(self) -> int:
        return len(self.t_s)

    def utc(self, seconds: float) -> datetime:
        return oe.as_utc(self.config.start) + timedelta(seconds=float(seconds))

    def index_at(self, seconds: float) -> int:
        return int(np.clip(round(seconds / self.config.step_s), 0, self.n - 1))

    def beam_state(self, i: int):
        """
        What the beam to the target satellite is doing at time index `i`: ("locked" | "degraded" | "blocked", reason).
          locked   - on the satellite and usable data is flowing
          degraded - on the satellite but the losses are too high for any usable data
          blocked  - the satellite is below the station's minimum elevation, or an event has cut the beam
        """
        link = self.primary_link
        if not link.visible[i]:
            return "blocked", "below minimum elevation"
        if link.outage_fraction[i] >= 0.5:
            names = [e.name for e in ev.active_events(self.events, float(self.t_s[i]), self.config.step_s) if e.outage]
            return "blocked", "event: " + (", ".join(names) if names else "outage")
        if link.adaptive_throughput_gbps[i] <= 0:
            return "degraded", "losses too high for usable data"
        return "locked", "usable data flowing"

    # ------------------------------------------------------------------------------------------------------------------
    def summary_text(self) -> str:
        cfg = self.config
        gs = cfg.ground
        lines = [f"Scenario start {oe.as_utc(cfg.start):%Y-%m-%d %H:%M:%S} UTC, {cfg.duration_h:g} h, step {cfg.step_s:g} s, "
                 f"{cfg.shadow_model} shadow, J2 {'on' if cfg.use_j2 else 'off'}",
                 f"Beam origin: {gs.name} at {gs.lat_deg:+.3f}, {gs.lon_deg:+.3f}, {gs.alt_m:.0f} m, minimum elevation {gs.min_elevation_deg:g} deg; "
                 f"sky: {cfg.atmosphere.label} (visibility {cfg.atmosphere.visibility_km:g} km, cloud {cfg.atmosphere.cloud_loss_db:g} dB, "
                 f"fog {cfg.atmosphere.fog_loss_db:g} dB, rain {cfg.atmosphere.rain_mm_h:g} mm/h)"]
        for k, tr in enumerate(self.tracks):
            a, e, i, raan, argp, nu = tr.elements0
            ecl = self.eclipse_windows[k]
            umbra_s = float(np.sum(ecl["umbra"][:, 1] - ecl["umbra"][:, 0]))
            pen_s = float(np.sum(ecl["penumbra"][:, 1] - ecl["penumbra"][:, 0]))
            lines.append(f"{tr.name}: a={a:.1f} km, inc={np.degrees(i):.3f} deg, RAAN={np.degrees(raan):.2f} deg, period "
                         f"{oe.orbital_period_s(a)/60:.2f} min, beta {tr.beta_deg[0]:+.1f} deg, umbra {len(ecl['umbra'])} passes "
                         f"({umbra_s/60:.1f} min total), penumbra {pen_s/60:.1f} min, sunlit {100*np.mean(tr.eclipse == 0):.1f} %")
        for k, link in self.ground_links.items():
            w = self.access_windows[k]
            dur = float(np.sum(w[:, 1] - w[:, 0]))
            above = float(np.mean(link.visible)) * cfg.duration_h * 60
            lines.append(f"Beam to {self.tracks[k].name}{' (target)' if k == self.target else ''}: {len(w)} passes above the mask, "
                         f"{dur/60:.1f} min with the beam on the satellite ({above:.1f} min above the horizon mask)")
            usable = link.adaptive_throughput_gbps > 0
            if link.locked.any():
                v = link.locked
                lines.append(f"   range {np.nanmin(link.range_km[v]):.0f}-{np.nanmax(link.range_km[v]):.0f} km, peak elevation "
                             f"{np.nanmax(link.channel['elevation_deg'][v]):.1f} deg, received power {np.nanmin(link.rx_power_dbm[v]):.1f} to "
                             f"{np.nanmax(link.rx_power_dbm[v]):.1f} dBm, atmosphere+weather loss "
                             f"{np.nanmin(link.channel['atmosphere_loss_db'][v]):.1f}-{np.nanmax(link.channel['atmosphere_loss_db'][v]):.1f} dB, "
                             f"Doppler {np.nanmin(link.doppler_ghz[v]):+.1f} to {np.nanmax(link.doppler_ghz[v]):+.1f} GHz, "
                             f"peak ground-telescope slew {np.nanmax(link.pointing_a.slew_rate_deg_s[v]):.2f} deg/s")
                data_min = float(np.sum(link.adaptive_throughput_gbps > 0)) * cfg.step_s / 60
                lines.append(f"   usable data flowing {data_min:.1f} min; best-modulation data delivered "
                             f"{np.sum(link.adaptive_throughput_gbps) * cfg.step_s / 8 / 1000:.1f} TB")
        for (i, j), w in self.crosslink_windows.items():
            dur = float(np.sum(w[:, 1] - w[:, 0]))
            lines.append(f"Crosslink {self.tracks[i].name} <-> {self.tracks[j].name}: {len(w)} windows, {dur/60:.1f} min of line of sight")
        if self.events:
            lines.append(f"Events: {len(self.events)} (" + ", ".join(f"{e.name} @ {e.start_s/3600:.2f} h for {e.duration_s:.0f} s" for e in self.events[:6])
                         + (" ..." if len(self.events) > 6 else "") + ")")
        else:
            lines.append("Events: none")
        return "\n".join(lines)

    def write_csv(self, path: str | Path) -> None:
        """Write the primary (ground to target satellite) link's time series as one CSV - only on explicit request."""
        link = self.primary_link
        tr = self.tracks[self.target]
        ch = link.channel
        cols = {"t_s": self.t_s, "utc": [f"{self.utc(t):%Y-%m-%dT%H:%M:%SZ}" for t in self.t_s],
                "elevation_deg": ch["elevation_deg"], "azimuth_deg": ch["azimuth_deg"], "above_mask": link.visible.astype(int),
                "locked": link.locked.astype(int), "range_km": link.range_km, "range_rate_km_s": link.range_rate_km_s,
                "doppler_ghz": link.doppler_ghz, "latency_ms": link.latency_ms, "rx_power_dbm": link.rx_power_dbm, "snr_db": link.snr_db,
                "clear_air_loss_db": ch["clear_air_loss_db"], "weather_loss_db": ch["weather_loss_db"],
                "turbulence_fade_db": ch["scintillation_fade_db"], "event_loss_db": ch["event_loss_db"], "event_outage_fraction": link.outage_fraction,
                "station_slew_deg_s": link.pointing_a.slew_rate_deg_s, "sat_slew_deg_s": link.pointing_b.slew_rate_deg_s}
        for m in MODULATIONS:
            cols[f"ber_{m}"] = link.ber[m]
            cols[f"throughput_{m}_gbps"] = link.throughput_gbps[m]
        cols["throughput_best_gbps"] = link.adaptive_throughput_gbps
        cols[f"{tr.name}_lat_deg"], cols[f"{tr.name}_lon_deg"], cols[f"{tr.name}_alt_km"] = tr.lat_deg, tr.lon_deg, tr.alt_km
        cols[f"{tr.name}_sun_visibility"] = tr.sun_visibility
        cols[f"{tr.name}_radiator_c"], cols[f"{tr.name}_solar_power_w"] = tr.radiator_temp_c, tr.solar_power_w
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols.keys())
            for k in range(self.n):
                w.writerow([c[k] if not isinstance(c[k], (float, np.floating)) else f"{c[k]:.6g}" for c in cols.values()])


def run_simulation(cfg: SimulationConfig, progress: Callable[[str, float], None] | None = None) -> SimulationResult:
    """Propagate every satellite, find eclipses, roll the events, and compute the ground beam (and any crosslinks)."""
    cfg.validate()
    say = progress or (lambda msg, frac: None)
    say("Preparing time grid, Sun and Earth rotation ...", 0.02)
    t_s = oe.time_grid(cfg.duration_h * 3600.0, cfg.step_s)
    sun = oe.sun_position_eci(cfg.start, t_s)
    gmst = oe.gmst_rad(cfg.start, t_s)
    tracks = []
    for k, sat in enumerate(cfg.satellites):
        say(f"Propagating {sat.name} (J2) ...", 0.05 + 0.4 * k / len(cfg.satellites))
        tracks.append(oe.compute_track(sat, cfg.start, t_s, sun, gmst, cfg.thermal, cfg.shadow_model, cfg.use_j2))
    say("Rolling events ...", 0.5)
    event_list = ev.generate_events(cfg.events, float(t_s[-1]))
    effects = ev.apply_events(event_list, t_s, cfg.step_s)
    say("Ground station geometry and atmosphere ...", 0.6)
    r_st, v_st = oe.station_state_eci(cfg.ground, gmst)
    sun_el = oe.sun_elevation_deg(cfg.ground, sun, gmst)
    topo, ground_links, access = {}, {}, {}
    omega = np.array([0.0, 0.0, oe.OMEGA_EARTH])
    for k, tr in enumerate(tracks):
        v_ecef = oe.eci_to_ecef(tr.v_eci, gmst) - np.cross(omega, tr.r_ecef)       # velocity seen from the rotating Earth
        topo[k] = oe.topocentric_state(cfg.ground, tr.r_ecef, v_ecef)
        link = compute_ground_link(t_s, cfg.ground, r_st, v_st, tr, topo[k], cfg.link, cfg.atmosphere, effects)
        ground_links[k] = link
        access[k] = oe.mask_to_intervals(link.locked, t_s)
    crosslinks, cross_windows = {}, {}
    pairs = list(combinations(range(len(tracks)), 2))
    for n, (i, j) in enumerate(pairs):
        say(f"Crosslink: {tracks[i].name} <-> {tracks[j].name} ...", 0.8 + 0.15 * n / max(1, len(pairs)))
        vis, clear = oe.line_of_sight(tracks[i].r_eci, tracks[j].r_eci, cfg.los_buffer_km)
        crosslinks[(i, j)] = compute_link(t_s, tracks[i], tracks[j], cfg.link, vis, clear)
        cross_windows[(i, j)] = oe.mask_to_intervals(vis, t_s, clear)
    eclipse = [{"penumbra": oe.state_intervals(tr.eclipse, t_s, oe.ECLIPSE_PENUMBRA),
                "umbra": oe.state_intervals(tr.eclipse, t_s, oe.ECLIPSE_UMBRA),
                "sunlit": oe.state_intervals(tr.eclipse, t_s, oe.ECLIPSE_SUNLIT)} for tr in tracks]
    say("Done", 1.0)
    return SimulationResult(cfg, t_s, sun, gmst, tracks, r_st, sun_el, topo, ground_links, access, crosslinks, cross_windows, eclipse,
                            event_list, effects)
