"""
simulation.py - glue between the two physics engines: describe a scenario, run it, get one result object.

    cfg = SimulationConfig()                  # two 600 km Sun-synchronous satellites, 24 h, 10 s steps
    result = run_simulation(cfg)              # a second or two on a laptop
    print(result.summary_text())

No GUI code lives here, so the same result object drives the desktop app, the headless command line and the tests.
Nothing is written to disk unless you ask (`write_csv`).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

import orbital_engine as oe
from laser_link_engine import LaserLinkConfig, LinkResult, MODULATIONS, compute_link

DEFAULT_START = datetime(2026, 12, 21, 0, 0, 0, tzinfo=timezone.utc)   # northern winter solstice: long eclipses for a morning orbit


def default_satellites() -> list[oe.SatelliteConfig]:
    """
    Two 600 km Sun-synchronous nodes in different orbit planes (local node times 10:30 and 13:30), so the laser path is clear
    for part of each orbit and blocked by the Earth for the rest - the interesting case for a link-budget study.
    """
    return [oe.SatelliteConfig("Orbital-Node-Alpha", altitude_km=600.0, ltan_hours=10.5, true_anomaly_deg=0.0),
            oe.SatelliteConfig("Orbital-Node-Beta", altitude_km=600.0, ltan_hours=13.5, true_anomaly_deg=45.0)]


@dataclass
class SimulationConfig:
    start: datetime = DEFAULT_START
    duration_h: float = 24.0
    step_s: float = 10.0                       # 1.0 for the finest result, 10.0 for a fast one
    satellites: list = field(default_factory=default_satellites)
    thermal: oe.ThermalConfig = field(default_factory=oe.ThermalConfig)
    link: LaserLinkConfig = field(default_factory=LaserLinkConfig)
    shadow_model: str = "conical"              # or "cylindrical"
    los_buffer_km: float = oe.ATMOSPHERE_BUFFER_KM
    use_j2: bool = True

    def validate(self) -> None:
        if len(self.satellites) < 2:
            raise ValueError("a link needs at least two satellites")
        if not (0 < self.step_s <= 600):
            raise ValueError("step must be between 0 and 600 seconds")
        if not (0 < self.duration_h <= 24 * 14):
            raise ValueError("duration must be between 0 and 14 days")
        if self.shadow_model not in ("conical", "cylindrical"):
            raise ValueError("shadow_model must be 'conical' or 'cylindrical'")

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
        thermal = oe.ThermalConfig(**d.pop("thermal", {}))
        link = LaserLinkConfig(**d.pop("link", {}))
        return cls(start=oe.as_utc(start), satellites=sats, thermal=thermal, link=link, **d)


@dataclass
class SimulationResult:
    config: SimulationConfig
    t_s: np.ndarray
    sun_eci: np.ndarray
    gmst: np.ndarray
    tracks: list
    links: dict                       # {(i, j): LinkResult} for every satellite pair, i < j
    access_windows: dict              # {(i, j): (m,2) seconds}
    eclipse_windows: list             # per satellite: {"penumbra": (m,2), "umbra": (m,2), "sunlit": (m,2)}

    @property
    def primary_pair(self) -> tuple:
        return (0, 1)

    @property
    def primary_link(self) -> LinkResult:
        return self.links[self.primary_pair]

    @property
    def n(self) -> int:
        return len(self.t_s)

    def utc(self, seconds: float) -> datetime:
        return oe.as_utc(self.config.start) + timedelta(seconds=float(seconds))

    def index_at(self, seconds: float) -> int:
        return int(np.clip(round(seconds / self.config.step_s), 0, self.n - 1))

    # ------------------------------------------------------------------------------------------------------------------
    def summary_text(self) -> str:
        cfg = self.config
        lines = [f"Scenario start {oe.as_utc(cfg.start):%Y-%m-%d %H:%M:%S} UTC, {cfg.duration_h:g} h, step {cfg.step_s:g} s, "
                 f"{cfg.shadow_model} shadow, J2 {'on' if cfg.use_j2 else 'off'}"]
        for k, tr in enumerate(self.tracks):
            a, e, i, raan, argp, nu = tr.elements0
            ecl = self.eclipse_windows[k]
            umbra_s = float(np.sum(ecl["umbra"][:, 1] - ecl["umbra"][:, 0]))
            pen_s = float(np.sum(ecl["penumbra"][:, 1] - ecl["penumbra"][:, 0]))
            lines.append(f"{tr.name}: a={a:.1f} km, inc={np.degrees(i):.3f} deg, RAAN={np.degrees(raan):.2f} deg, period "
                         f"{oe.orbital_period_s(a)/60:.2f} min, beta {tr.beta_deg[0]:+.1f} deg, umbra {len(ecl['umbra'])} passes "
                         f"({umbra_s/60:.1f} min total), penumbra {pen_s/60:.1f} min, sunlit {100*np.mean(tr.eclipse == 0):.1f} %")
        for (i, j), link in self.links.items():
            w = self.access_windows[(i, j)]
            dur = float(np.sum(w[:, 1] - w[:, 0]))
            head = f"Link {self.tracks[i].name} <-> {self.tracks[j].name}: {len(w)} access windows, {dur/60:.1f} min of line of sight " \
                   f"({100*dur/max(self.t_s[-1], 1):.1f} % of the scenario)"
            lines.append(head)
            if link.visible.any():
                v = link.visible
                lines.append(f"   range {np.nanmin(link.range_km[v]):.0f}-{np.nanmax(link.range_km[v]):.0f} km, received power "
                             f"{np.nanmin(link.rx_power_dbm[v]):.1f} to {np.nanmax(link.rx_power_dbm[v]):.1f} dBm, latency "
                             f"{np.nanmin(link.latency_ms[v]):.2f}-{np.nanmax(link.latency_ms[v]):.2f} ms, Doppler "
                             f"{np.nanmin(link.doppler_ghz[v]):+.1f} to {np.nanmax(link.doppler_ghz[v]):+.1f} GHz, peak turret slew "
                             f"{np.nanmax(link.pointing_a.slew_rate_deg_s[v]):.3f} deg/s")
        return "\n".join(lines)

    def write_csv(self, path: str | Path) -> None:
        """Write the primary link's time series (and both satellites' state) as one CSV - only on explicit request."""
        link = self.primary_link
        a, b = self.tracks[self.primary_pair[0]], self.tracks[self.primary_pair[1]]
        cols = {"t_s": self.t_s, "utc": [f"{self.utc(t):%Y-%m-%dT%H:%M:%SZ}" for t in self.t_s],
                "los": link.visible.astype(int), "range_km": link.range_km, "range_rate_km_s": link.range_rate_km_s,
                "doppler_ghz": link.doppler_ghz, "latency_ms": link.latency_ms, "rx_power_dbm": link.rx_power_dbm,
                "snr_db": link.snr_db, "slew_a_deg_s": link.pointing_a.slew_rate_deg_s, "slew_b_deg_s": link.pointing_b.slew_rate_deg_s}
        for m in MODULATIONS:
            cols[f"ber_{m}"] = link.ber[m]
            cols[f"throughput_{m}_gbps"] = link.throughput_gbps[m]
        for tr in (a, b):
            cols[f"{tr.name}_lat_deg"] = tr.lat_deg
            cols[f"{tr.name}_lon_deg"] = tr.lon_deg
            cols[f"{tr.name}_alt_km"] = tr.alt_km
            cols[f"{tr.name}_sun_visibility"] = tr.sun_visibility
            cols[f"{tr.name}_radiator_c"] = tr.radiator_temp_c
            cols[f"{tr.name}_solar_power_w"] = tr.solar_power_w
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols.keys())
            for k in range(self.n):
                w.writerow([c[k] if not isinstance(c[k], float) else f"{c[k]:.6g}" for c in cols.values()])


def run_simulation(cfg: SimulationConfig, progress: Callable[[str, float], None] | None = None) -> SimulationResult:
    """Propagate every satellite, find eclipses and line of sight, and compute the laser-link budget for every pair."""
    cfg.validate()
    say = progress or (lambda msg, frac: None)
    say("Preparing time grid, Sun and Earth rotation ...", 0.02)
    t_s = oe.time_grid(cfg.duration_h * 3600.0, cfg.step_s)
    sun = oe.sun_position_eci(cfg.start, t_s)
    gmst = oe.gmst_rad(cfg.start, t_s)
    tracks = []
    for k, sat in enumerate(cfg.satellites):
        say(f"Propagating {sat.name} (J2) ...", 0.05 + 0.5 * k / len(cfg.satellites))
        tracks.append(oe.compute_track(sat, cfg.start, t_s, sun, gmst, cfg.thermal, cfg.shadow_model, cfg.use_j2))
    links, access = {}, {}
    pairs = list(combinations(range(len(tracks)), 2))
    for n, (i, j) in enumerate(pairs):
        say(f"Line of sight and link budget: {tracks[i].name} <-> {tracks[j].name} ...", 0.6 + 0.35 * n / len(pairs))
        vis, clear = oe.line_of_sight(tracks[i].r_eci, tracks[j].r_eci, cfg.los_buffer_km)
        links[(i, j)] = compute_link(t_s, tracks[i], tracks[j], cfg.link, vis, clear)
        access[(i, j)] = oe.mask_to_intervals(vis, t_s, clear)
    eclipse = [{"penumbra": oe.state_intervals(tr.eclipse, t_s, oe.ECLIPSE_PENUMBRA),
                "umbra": oe.state_intervals(tr.eclipse, t_s, oe.ECLIPSE_UMBRA),
                "sunlit": oe.state_intervals(tr.eclipse, t_s, oe.ECLIPSE_SUNLIT)} for tr in tracks]
    say("Done", 1.0)
    return SimulationResult(cfg, t_s, sun, gmst, tracks, links, access, eclipse)
