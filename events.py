"""
events.py - random and rare things that can interrupt or degrade a laser link: weather disasters, birds, aircraft, earthquakes, solar storms ...

Each kind of event has
  * a typical RATE (events per day, averaged over a year - illustrative round numbers, NOT statistics for any real place),
  * a DURATION range (picked log-uniformly: short ones are far more common than long ones),
  * an EFFECT: extra loss in dB, a complete outage, and/or a pointing-jitter multiplier.

`generate_events` rolls the dice with a seed, so the same seed always gives the same "bad day" and you can compare settings fairly. Events
arrive as a Poisson process (random, independent, a constant average rate). You can switch kinds on/off, scale all rates (x0.1 ... x1000, to
make rare disasters show up in a 24 h run) or add events by hand at a chosen time.

Events shorter than the simulation time step (a bird crossing for 1 s in a 10 s step) are applied as the FRACTION of the step they cover, so
the throughput for that step is reduced proportionally instead of the event vanishing between samples.

Honest limits: the effect sizes and rates are plausible engineering guesses for exploration ("what if a volcano erupts upwind?"), not measured
data. Real laser ground stations deal with aircraft by switching the beam off (eye safety) - that is modelled here as an outage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NATURE, ANIMALS, TECHNICAL = "Nature", "Animals & aircraft", "Technical & space"
CATEGORIES = [NATURE, ANIMALS, TECHNICAL]
CATEGORY_COLORS = {NATURE: "#ff7a45", ANIMALS: "#ffd93d", TECHNICAL: "#c77dff"}


@dataclass(frozen=True)
class EventType:
    key: str
    name: str
    category: str
    description: str
    rate_per_day: float                 # average occurrences per day at the "ground station" scope
    duration_s: tuple                   # (shortest, longest) seconds
    loss_db: tuple | None = None        # (mildest, worst) extra loss in dB; None = no loss effect
    outage: bool = False                # True = the beam is completely lost while it lasts
    jitter_factor: tuple = (1.0, 1.0)   # multiplier on the pointing jitter (earthquake shaking the telescope)


EVENT_TYPES = {t.key: t for t in [
    # ---- animals and aircraft -------------------------------------------------------------------------------------------
    EventType("bird", "Bird crosses the beam", ANIMALS, "A single bird flies through the beam: a very brief total blockage.", 6.0, (0.2, 4.0), outage=True),
    EventType("flock", "Flock of birds", ANIMALS, "A flock of birds or bats crosses the beam path.", 2.0, (2.0, 25.0), outage=True),
    EventType("aircraft", "Aircraft crosses the beam", ANIMALS, "Laser stations switch the beam off when an aircraft is near (eye safety): a short outage.", 4.0, (2.0, 20.0), outage=True),
    EventType("insects", "Insect swarm", ANIMALS, "A swarm of insects (mayflies, locusts) drifts through the beam: scattering loss.", 1.0, (30.0, 600.0), loss_db=(2.0, 10.0)),
    # ---- nature ---------------------------------------------------------------------------------------------------------
    EventType("cirrus", "Cirrus cloud passes", NATURE, "Thin high ice cloud drifts over the station.", 0.5, (1200.0, 7200.0), loss_db=(1.0, 4.0)),
    EventType("fog", "Fog bank rolls in", NATURE, "Dense low fog: the beam is heavily scattered.", 0.10, (1800.0, 21600.0), loss_db=(12.0, 40.0)),
    EventType("storm", "Thunderstorm cell", NATURE, "Thick cloud and heavy rain over the station.", 0.15, (1200.0, 10800.0), loss_db=(20.0, 45.0)),
    EventType("dust", "Dust storm", NATURE, "A cloud of desert dust (for example a Saharan dust outbreak).", 0.05, (3600.0, 43200.0), loss_db=(3.0, 12.0)),
    EventType("smoke", "Wildfire smoke plume", NATURE, "Smoke from a wildfire drifts over the station.", 0.03, (7200.0, 86400.0), loss_db=(3.0, 15.0)),
    EventType("ash", "Volcanic ash cloud", NATURE, "An eruption fills the sky with ash: the link is lost for many hours.", 0.002, (21600.0, 259200.0), loss_db=(30.0, 60.0), outage=True),
    EventType("hurricane", "Hurricane / typhoon", NATURE, "A tropical cyclone passes over the station.", 0.002, (21600.0, 172800.0), loss_db=(40.0, 60.0), outage=True),
    EventType("earthquake", "Earthquake at the station", NATURE, "Ground shaking throws the telescope off target; it needs minutes to re-acquire the satellite.", 0.003, (60.0, 900.0), loss_db=(6.0, 30.0), outage=True, jitter_factor=(5.0, 40.0)),
    EventType("flood", "Flood / station power cut", TECHNICAL, "The station loses mains power or is flooded: out until it is back.", 0.02, (300.0, 7200.0), outage=True),
    # ---- technical and space --------------------------------------------------------------------------------------------
    EventType("solar_storm", "Solar storm hits the satellite", TECHNICAL, "Energetic particles upset the satellite's electronics; the optical terminal reboots.", 0.02, (120.0, 1800.0), outage=True),
    EventType("debris", "Debris avoidance manoeuvre", TECHNICAL, "The satellite turns away to dodge debris: the terminal points elsewhere for a while.", 0.01, (600.0, 3600.0), outage=True),
    EventType("safe_mode", "Satellite enters safe mode", TECHNICAL, "A fault puts the satellite in safe mode; the laser is off until ground control recovers it.", 0.005, (1800.0, 14400.0), outage=True),
]}


@dataclass
class ManualEvent:
    """An event you place yourself: kind, when it starts (seconds from scenario start), how long, and how severe (0 = mildest, 1 = worst)."""
    key: str
    start_s: float
    duration_s: float
    severity: float = 0.5


@dataclass
class EventsConfig:
    enabled: bool = False                 # random events on/off (manual events always apply)
    seed: int = 1
    rate_scale: float = 1.0               # multiplies every rate: 1 = typical, 100 = "everything goes wrong"
    disabled_types: list = field(default_factory=list)   # keys of kinds to leave out
    manual: list = field(default_factory=list)           # list of ManualEvent


@dataclass
class Event:
    key: str
    name: str
    category: str
    start_s: float
    end_s: float
    loss_db: float                # 0 when the kind has no loss effect
    outage: bool
    jitter_factor: float
    source: str = "random"        # or "manual"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _lerp(lohi, u: float) -> float:
    return lohi[0] + u * (lohi[1] - lohi[0])


def _make_event(et: EventType, start_s: float, duration_s: float, severity: float, source: str) -> Event:
    return Event(et.key, et.name, et.category, start_s, start_s + duration_s, _lerp(et.loss_db, severity) if et.loss_db else 0.0,
                 et.outage, _lerp(et.jitter_factor, severity), source)


def generate_events(cfg: EventsConfig, duration_s: float) -> list:
    """Random events (if enabled) plus the manual ones, sorted by start time. Deterministic for a given seed."""
    out = []
    if cfg.enabled:
        rng = np.random.default_rng(cfg.seed)
        for key in sorted(EVENT_TYPES):                                 # sorted: the dice rolls do not depend on dict order
            et = EVENT_TYPES[key]
            if key in cfg.disabled_types:
                continue
            n = rng.poisson(et.rate_per_day * cfg.rate_scale * duration_s / 86400.0)
            lo, hi = math.log(et.duration_s[0]), math.log(et.duration_s[1])
            for _ in range(n):
                start = float(rng.uniform(0.0, duration_s))
                dur = float(math.exp(rng.uniform(lo, hi)))
                out.append(_make_event(et, start, dur, float(rng.uniform()), "random"))
    for m in cfg.manual:
        et = EVENT_TYPES.get(m.key)
        if et is not None and m.start_s < duration_s:
            out.append(_make_event(et, float(m.start_s), float(m.duration_s), float(np.clip(m.severity, 0.0, 1.0)), "manual"))
    return sorted(out, key=lambda e: e.start_s)


@dataclass
class EventEffects:
    loss_db: np.ndarray               # (n,) extra loss from events, dB (transmittances multiply)
    outage_fraction: np.ndarray       # (n,) 0..1: share of this time step during which the beam is completely lost
    jitter_factor: np.ndarray         # (n,) multiplier on the pointing jitter (1 = normal)


def apply_events(events, t_s: np.ndarray, step_s: float) -> EventEffects:
    """Turn the event list into per-time-step effects. Each step covers [t - step/2, t + step/2]."""
    n = len(t_s)
    transmittance = np.ones(n)
    no_outage = np.ones(n)
    jitter = np.ones(n)
    half = step_s / 2.0
    for e in events:
        i0 = max(0, int(math.floor((e.start_s - half) / step_s)))
        i1 = min(n, int(math.ceil((e.end_s + half) / step_s)) + 1)
        if i1 <= i0:
            continue
        t = t_s[i0:i1]
        frac = np.clip(np.minimum(e.end_s, t + half) - np.maximum(e.start_s, t - half), 0.0, step_s) / step_s
        if e.loss_db > 0:
            transmittance[i0:i1] *= 1.0 - frac + frac * 10.0 ** (-e.loss_db / 10.0)
        if e.outage:
            no_outage[i0:i1] *= 1.0 - frac
        if e.jitter_factor > 1.0:
            jitter[i0:i1] = np.maximum(jitter[i0:i1], np.where(frac > 0, e.jitter_factor, 1.0))
    return EventEffects(np.maximum(-10.0 * np.log10(transmittance), 0.0) + 0.0, 1.0 - no_outage, jitter)


def active_events(events, t_s: float, step_s: float) -> list:
    """Events overlapping the time step centred on `t_s` (what the telemetry panel lists as 'happening now')."""
    half = step_s / 2.0
    return [e for e in events if e.start_s <= t_s + half and e.end_s >= t_s - half]
