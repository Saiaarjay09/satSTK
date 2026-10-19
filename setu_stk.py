#!/usr/bin/env python3
"""
SETU: STK ORBIT + ECLIPSE + OPTICAL INTER-SATELLITE LINK (OISL) AUTOMATION
==========================================================================================
Drives Ansys STK (Systems Tool Kit) from Python to
  1. create a 24-hour scenario with two 600 km Sun-synchronous satellites (Orbital-Node-Alpha / -Beta),
  2. report when Node Alpha is in sunlight, penumbra and umbra (the thermal / MPFL design cases),
  3. fit Alpha with a 1550 nm laser transmitter and Beta with a laser receiver (narrow beam, 100 microradians),
  4. compute line-of-sight access and extract Received Power, BER and latency during each access window.

READ THIS FIRST (honest status)
  * Written from Ansys' public STK documentation. It has NOT been run against a live STK, because STK is
    Windows/Linux software and was not available where this was written. Every STK call is isolated in a small
    function with a clear error message, and the script keeps going (and tells you exactly what to do by hand in the
    STK window) when a step is rejected. Expect to adjust a name or two on the first run.
  * Needs STK with the Communications capability, and the STK Python API.
  * `--offline-preview` needs only NumPy. It models the same two orbits with a simple analytic model (circular orbits,
    J2 plane drift, cylindrical Earth shadow) so you can sanity-check the geometry, the eclipse times and the line-of-sight
    pattern before opening STK, and compare them with STK's answers afterwards.

Run:   python setu_stk.py                  (STK desktop, window visible)
       python setu_stk.py --engine         (STK Engine, no window)
       python setu_stk.py --offline-preview
All results are written to ./stk_output/ (CSV files, raw reports and the STK scenario).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ==========================================================================================================
#  >>> CHANGE THESE NUMBERS <<<
# ==========================================================================================================
# --- Scenario ----------------------------------------------------------------------------------------------
SCENARIO_NAME = "Orbital_AI_Datacentre"
SCENARIO_START = "21 Dec 2026 00:00:00.000"      # UTCG. Northern winter: the longest eclipses for a morning orbit
SIM_HOURS = 24                                    # length of the simulation window
REPORT_STEP_S = 30                                # time step of the link-budget samples, seconds

# --- Orbits: 600 km Sun-synchronous, circular -----------------------------------------------------------------
ALTITUDE_KM = 600.0
ECCENTRICITY = 0.0
NODE_ALPHA_LTAN_H = 10.5     # local solar time of the ascending node, hours (10.5 = 10:30, a common morning SSO)
NODE_BETA_LTAN_H = 13.5      # a different orbit plane (13:30), so the two nodes drift in and out of sight of each other
NODE_ALPHA_TRUE_ANOMALY_DEG = 0.0
NODE_BETA_TRUE_ANOMALY_DEG = 45.0   # where Beta starts along its orbit; changes WHEN the line-of-sight windows occur

# --- Optical link (OISL) ---------------------------------------------------------------------------------------
LASER_WAVELENGTH_NM = 1550.0
LASER_POWER_W = 10.0
DATA_RATE_GBPS = 1000.0               # 1000 Gbit/s = 1 Tbit/s
BEAM_DIVERGENCE_URAD = 100.0          # full-angle beam divergence the fast steering mirror delivers
TX_APERTURE_M = 0.30                  # transmit telescope diameter (informational; set in STK if the model asks for it)
RX_APERTURE_M = 0.30                  # receive telescope diameter
EARTH_LIMB_MARGIN_KM = 100.0          # a link must clear the Earth by this much (keeps the beam out of the atmosphere)

# --- Names ------------------------------------------------------------------------------------------------------
ALPHA, BETA = "Orbital-Node-Alpha", "Orbital-Node-Beta"
TX_NAME, RX_NAME = "LaserTx", "LaserRx"
OUTPUT_DIR = Path("stk_output")
# ==========================================================================================================

MU_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
J2 = 1.08262668e-3
C_KM_S = 299792.458
SUN_SYNC_RATE_RAD_S = 2 * math.pi / (365.2422 * 86400.0)      # the node must advance one turn per tropical year
NULL_DB = -9000.0                                              # STK reports "no valid link" as -9999.99 dB


class StkError(RuntimeError):
    """A problem talking to STK, with a message that says what to try."""


# ==========================================================================================================
#  1. PURE ORBIT MATHS (no STK needed; also used by --offline-preview)
# ==========================================================================================================
def parse_utcg(text: str) -> datetime:
    return datetime.strptime(text.strip(), "%d %b %Y %H:%M:%S.%f") if "." in text else datetime.strptime(text.strip(), "%d %b %Y %H:%M:%S")


def format_utcg(t: datetime) -> str:
    return t.strftime("%d %b %Y %H:%M:%S.") + f"{t.microsecond // 1000:03d}"


def sso_inclination_deg(altitude_km: float, ecc: float = 0.0) -> float:
    """Inclination at which the J2 nodal drift equals the Sun's mean motion (about 97.8 degrees at 600 km)."""
    a = R_EARTH_KM + altitude_km
    n = math.sqrt(MU_KM3_S2 / a ** 3)
    p = a * (1 - ecc ** 2)
    cos_i = -SUN_SYNC_RATE_RAD_S / (1.5 * J2 * n * (R_EARTH_KM / p) ** 2)
    if abs(cos_i) > 1:
        raise ValueError(f"No Sun-synchronous inclination exists at {altitude_km} km (try below roughly 5,900 km).")
    return math.degrees(math.acos(cos_i))


def sun_ecliptic_longitude_deg(t: datetime) -> float:
    n = (t - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
    L = 280.460 + 0.9856474 * n
    g = math.radians(357.528 + 0.9856003 * n)
    return (L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0


def sun_right_ascension_deg(t: datetime) -> float:
    lam = math.radians(sun_ecliptic_longitude_deg(t))
    eps = math.radians(23.439)
    return math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360.0


def raan_for_ltan_deg(ltan_hours: float, t: datetime) -> float:
    """RAAN that puts the ascending node at the given local solar time (accurate to about a degree)."""
    return (sun_right_ascension_deg(t) + 15.0 * (ltan_hours - 12.0)) % 360.0


def orbit_elements(ltan_hours: float, true_anomaly_deg: float, start: datetime) -> dict:
    return dict(a_km=R_EARTH_KM + ALTITUDE_KM, ecc=ECCENTRICITY, inc_deg=sso_inclination_deg(ALTITUDE_KM, ECCENTRICITY),
                raan_deg=raan_for_ltan_deg(ltan_hours, start), argp_deg=0.0, ta_deg=true_anomaly_deg)


def offline_propagate(el: dict, start: datetime, seconds):
    """Circular orbit with secular J2 plane drift. Returns (times_s, positions_km[N,3])."""
    import numpy as np
    t = np.asarray(seconds, dtype=float)
    a, inc = el["a_km"], math.radians(el["inc_deg"])
    n = math.sqrt(MU_KM3_S2 / a ** 3)
    raan_rate = -1.5 * n * J2 * (R_EARTH_KM / a) ** 2 * math.cos(inc)
    u = math.radians(el["argp_deg"] + el["ta_deg"]) + n * t          # argument of latitude (circular: nu = M)
    om = math.radians(el["raan_deg"]) + raan_rate * t
    x = a * (np.cos(u) * np.cos(om) - np.sin(u) * math.cos(inc) * np.sin(om))
    y = a * (np.cos(u) * np.sin(om) + np.sin(u) * math.cos(inc) * np.cos(om))
    z = a * np.sin(u) * math.sin(inc)
    return t, np.stack([x, y, z], axis=1)


def offline_sun_vectors(start: datetime, seconds):
    import numpy as np
    eps = math.radians(23.439)
    out = []
    for s in seconds:
        lam = math.radians(sun_ecliptic_longitude_deg(start + timedelta(seconds=float(s))))
        out.append([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])
    return np.array(out)


def intervals(mask, seconds):
    """Start/stop pairs (seconds) where a boolean array is True."""
    import numpy as np
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(float(seconds[a]), float(seconds[min(b, len(seconds)) - 1])) for a, b in zip(edges[::2], edges[1::2])]


def offline_preview(start_text: str = None) -> dict:
    """Analytic estimate of eclipse and line-of-sight for the two nodes. Returns the numbers and prints a summary."""
    import numpy as np
    start = parse_utcg(start_text or SCENARIO_START)
    sec = np.arange(0.0, SIM_HOURS * 3600.0 + 1e-9, 10.0)
    ea = orbit_elements(NODE_ALPHA_LTAN_H, NODE_ALPHA_TRUE_ANOMALY_DEG, start)
    eb = orbit_elements(NODE_BETA_LTAN_H, NODE_BETA_TRUE_ANOMALY_DEG, start)
    _, ra = offline_propagate(ea, start, sec)
    _, rb = offline_propagate(eb, start, sec)
    sun = offline_sun_vectors(start, sec)

    along = np.einsum("ij,ij->i", ra, sun)
    perp = np.linalg.norm(ra - along[:, None] * sun, axis=1)
    eclipse = (along < 0) & (perp < R_EARTH_KM)

    d = rb - ra
    t_star = np.clip(-np.einsum("ij,ij->i", ra, d) / np.einsum("ij,ij->i", d, d), 0, 1)
    closest = np.linalg.norm(ra + t_star[:, None] * d, axis=1)
    los = closest > (R_EARTH_KM + EARTH_LIMB_MARGIN_KM)
    rng = np.linalg.norm(d, axis=1)

    result = dict(inclination_deg=ea["inc_deg"], raan_alpha_deg=ea["raan_deg"], raan_beta_deg=eb["raan_deg"],
                  period_min=2 * math.pi * math.sqrt(ea["a_km"] ** 3 / MU_KM3_S2) / 60.0,
                  eclipse=intervals(eclipse, sec), access=intervals(los, sec), start=start,
                  range_km=(float(rng[los].min()) if los.any() else None, float(rng[los].max()) if los.any() else None),
                  eclipse_fraction=float(eclipse.mean()), los_fraction=float(los.mean()))
    return result


def print_preview(r: dict) -> None:
    def hms(s):
        return str(timedelta(seconds=int(s)))
    print(f"Sun-synchronous inclination at {ALTITUDE_KM:.0f} km: {r['inclination_deg']:.3f} deg; period {r['period_min']:.2f} min")
    print(f"RAAN  Alpha {r['raan_alpha_deg']:.2f} deg (LTAN {NODE_ALPHA_LTAN_H:.2f} h)   Beta {r['raan_beta_deg']:.2f} deg (LTAN {NODE_BETA_LTAN_H:.2f} h)")
    print(f"\nANALYTIC ESTIMATE (not STK): Node Alpha eclipse, cylindrical shadow, {len(r['eclipse'])} passes, "
          f"{100 * r['eclipse_fraction']:.1f}% of the time in shadow")
    for a, b in r["eclipse"][:6]:
        print(f"   start +{hms(a)}  stop +{hms(b)}  duration {(b - a) / 60:.1f} min")
    if len(r["eclipse"]) > 6:
        print(f"   ... {len(r['eclipse']) - 6} more")
    print(f"\nANALYTIC ESTIMATE: line of sight Alpha-Beta (clear of Earth + {EARTH_LIMB_MARGIN_KM:.0f} km): {len(r['access'])} windows, "
          f"{100 * r['los_fraction']:.1f}% of the time")
    for a, b in r["access"][:8]:
        print(f"   start +{hms(a)}  stop +{hms(b)}  duration {(b - a) / 60:.1f} min")
    if r["range_km"][0] is not None:
        print(f"   range while visible: {r['range_km'][0]:,.0f} to {r['range_km'][1]:,.0f} km")


# ==========================================================================================================
#  2. TALKING TO STK
# ==========================================================================================================
def start_stk(use_engine: bool = False, visible: bool = True):
    """Launch STK and return (application, root). Uses the agi.stk12 Python API that ships with STK 12."""
    try:
        if use_engine:
            from agi.stk12.stkengine import STKEngine
            app = STKEngine.StartApplication(noGraphics=True)
            return app, app.NewObjectRoot()
        from agi.stk12.stkdesktop import STKDesktop
        app = STKDesktop.StartApplication(visible=visible, userControl=True)
        return app, app.Root
    except ImportError as e:
        raise StkError(
            "The STK Python API (agi.stk12) is not installed in this Python.\n"
            "Install the wheel that comes with STK, e.g.\n"
            '    pip install "C:\\Program Files\\AGI\\STK 12\\bin\\AgPythonAPI\\agi.stk12-<version>-py3-none-any.whl"\n'
            "(search the STK install folder for agi.stk12*.whl; match your Python version to the STK release notes).\n"
            "STK 13 / Ansys STK 2025+ uses the newer package 'ansys-stk-core' with different names: see the README notes.") from e
    except Exception as e:                                       # COM / licence problems
        raise StkError(f"STK could not be started: {e}\nCheck that STK is installed, licensed and starts from the Start menu.") from e


def connect(root, command: str) -> list[str]:
    """Run a Connect command and return its result lines. Raises StkError naming the command if STK rejects it."""
    try:
        res = root.ExecuteCommand(command)
    except Exception as e:
        raise StkError(f"STK rejected this command:\n    {command}\n  -> {e}") from e
    lines = []
    try:
        for i in range(res.Count):
            lines.append(str(res.Item(i)))
    except Exception:
        pass
    return lines


_ATTR = re.compile(r"^\s*([A-Za-z_][\w\.\[\]]*)\s*=\s*(.*)$")


def list_attributes(root, kind: str, path: str, base: str) -> list[tuple[str, str]]:
    """(attribute name, value text) pairs for a transmitter / receiver attribute group. [] if STK refuses."""
    try:
        lines = connect(root, f"{kind}_RM {path} GetValue {base}")
    except StkError:
        return []
    out = []
    for ln in lines:
        m = _ATTR.match(ln)
        if m and "(Read-only)" not in m.group(2):
            out.append((m.group(1), m.group(2)))
    return out


def set_first_matching(root, kind: str, path: str, pattern: str, value: str, bases=("Model.AntennaControl", "Model")) -> str | None:
    """Set the first writable attribute whose name matches `pattern`. Returns the attribute set, or None if nothing matched.
    Used for the laser antenna settings, whose exact attribute names differ between STK releases."""
    rx = re.compile(pattern, re.I)
    tried = set()
    for base in bases:
        for name, _ in list_attributes(root, kind, path, base):
            if not rx.search(name.split(".")[-1]):
                continue
            for cand in (name, f"Model.{name}", f"{base}.{name}"):
                if cand in tried:
                    continue
                tried.add(cand)
                try:
                    connect(root, f"{kind} {path} SetValue {cand} {value}")
                    return cand
                except StkError:
                    continue
    return None


def build_scenario(root):
    start = parse_utcg(SCENARIO_START)
    stop = start + timedelta(hours=SIM_HOURS)
    root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")
    try:
        root.CloseScenario()
    except Exception:
        pass
    root.NewScenario(SCENARIO_NAME)
    scenario = root.CurrentScenario
    scenario.SetTimePeriod(format_utcg(start), format_utcg(stop))
    try:
        root.Rewind()
    except Exception:
        pass                                                    # not available in Engine / no-graphics mode
    return scenario, start, stop


def add_sso_satellite(scenario, name: str, ltan_h: float, true_anomaly_deg: float, start: datetime):
    from agi.stk12.stkobjects import (AgEClassicalLocation, AgEClassicalSizeShape, AgEOrientationAscNode, AgESTKObjectType,
                                      AgEVePropagatorType)
    from agi.stk12.stkutil import AgEOrbitStateType
    el = orbit_elements(ltan_h, true_anomaly_deg, start)
    sat = scenario.Children.New(AgESTKObjectType.eSatellite, name)
    # J2 propagator: needed so the orbit plane drifts one turn per year (that drift IS "Sun-synchronous").
    sat.SetPropagatorType(AgEVePropagatorType.ePropagatorJ2Perturbation)
    prop = sat.Propagator
    state = prop.InitialState.Representation
    classical = state.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)
    classical.SizeShapeType = AgEClassicalSizeShape.eSizeShapeSemimajorAxis
    classical.SizeShape.SemiMajorAxis = el["a_km"]              # STK default distance unit: km
    classical.SizeShape.Eccentricity = el["ecc"]
    classical.Orientation.Inclination = el["inc_deg"]           # degrees
    classical.Orientation.ArgOfPerigee = el["argp_deg"]
    classical.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeRAAN
    classical.Orientation.AscNode.Value = el["raan_deg"]
    classical.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
    classical.Location.Value = el["ta_deg"]
    state.Assign(classical)
    prop.Propagate()
    return sat, el


# ---- eclipse ---------------------------------------------------------------------------------------------------
_DATE = r"\d{1,2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2}(?:\.\d+)?"
_ROW = re.compile(rf"({_DATE})\s*,?\s+({_DATE})\s*,?\s+([-\d\.eE+]+)")


def parse_lighting_report(lines: list[str]) -> list[dict]:
    """Rows from a STK 'Lighting Times' report: [{'kind': 'Umbra'|'Penumbra'|'Sunlight', 'start', 'stop', 'duration_s'}]."""
    rows, kind = [], None
    for ln in lines:
        low = ln.lower()
        for k in ("umbra", "penumbra", "sunlight"):
            if k in low and not _ROW.search(ln):
                kind = k.capitalize()
        m = _ROW.search(ln)
        if m and kind:
            rows.append(dict(kind=kind, start=m.group(1), stop=m.group(2), duration_s=float(m.group(3))))
    return rows


def alpha_eclipse_times(root, start: datetime, stop: datetime, raw_out: Path) -> list[dict]:
    path = f"*/Satellite/{ALPHA}"
    lines = connect(root, f'Report_RM {path} Style "Lighting Times" TimePeriod "{format_utcg(start)}" "{format_utcg(stop)}" TimeStep 30')
    raw_out.write_text("\n".join(lines))
    return parse_lighting_report(lines)


# ---- laser link ------------------------------------------------------------------------------------------------
def add_laser_terminals(root, scenario) -> dict:
    """Transmitter on Alpha, receiver on Beta, both with STK's laser models. Returns what was and was not configured."""
    from agi.stk12.stkobjects import AgESTKObjectType
    alpha, beta = scenario.Children.Item(ALPHA), scenario.Children.Item(BETA)
    alpha.Children.New(AgESTKObjectType.eTransmitter, TX_NAME)
    beta.Children.New(AgESTKObjectType.eReceiver, RX_NAME)
    tx = f"*/Satellite/{ALPHA}/Transmitter/{TX_NAME}"
    rx = f"*/Satellite/{BETA}/Receiver/{RX_NAME}"
    log = {}

    def step(label, fn):
        try:
            log[label] = fn() or "ok"
        except StkError as e:
            log[label] = f"NOT SET: {e}"

    # STK's own model names (from Transmitter_RM / Receiver_RM GetValue Model:Choices)
    step("transmitter model", lambda: connect(root, f"Transmitter {tx} SetValue Model Laser_Transmitter_Model"))
    step("receiver model", lambda: connect(root, f"Receiver {rx} SetValue Model Laser_Receiver_Model"))
    # Values below are in STK's internal SI units: Hz, W, bit/s, rad.
    freq_hz = C_KM_S * 1000.0 / (LASER_WAVELENGTH_NM * 1e-9)
    step("frequency (1550 nm)", lambda: connect(root, f"Transmitter {tx} SetValue Model.Frequency {freq_hz:.6e}"))
    step("power", lambda: connect(root, f"Transmitter {tx} SetValue Model.Power {LASER_POWER_W}"))
    step("data rate", lambda: connect(root, f"Transmitter {tx} SetValue Model.DataRate {DATA_RATE_GBPS * 1e9:.6e}"))
    step("receiver frequency tracking", lambda: connect(root, f"Receiver {rx} SetValue Model.AutoTrackFrequency true"))
    # Optical antenna: pick an optical / Gaussian model if STK offers a choice, then the beam divergence.
    try:
        choices = connect(root, f"Transmitter_RM {tx} GetValue Model.AntennaControl.Antenna:Choices")
        pick = next((c.strip() for c in choices if re.search(r"optical", c, re.I)), None) or next(
            (c.strip() for c in choices if re.search(r"gauss", c, re.I)), None)
        if pick:
            step(f"transmitter antenna = {pick}", lambda: connect(root, f"Transmitter {tx} SetValue Model.AntennaControl.Antenna {pick}"))
    except StkError:
        pass
    div_rad = BEAM_DIVERGENCE_URAD * 1e-6
    hit = set_first_matching(root, "Transmitter", tx, r"divergence|beamwidth", f"{div_rad:.6e}")
    log["beam divergence (100 urad)"] = f"set via {hit}" if hit else "NOT SET: no divergence/beamwidth attribute found; see stk_output/*_attributes.txt"
    hit = set_first_matching(root, "Transmitter", tx, r"^(diameter|aperture)", f"{TX_APERTURE_M}")
    log["transmit aperture"] = f"set via {hit}" if hit else "left at STK default"
    hit = set_first_matching(root, "Receiver", rx, r"^(diameter|aperture)", f"{RX_APERTURE_M}")
    log["receive aperture"] = f"set via {hit}" if hit else "left at STK default"
    return dict(tx=tx, rx=rx, log=log)


def dump_attributes(root, comm: dict, folder: Path) -> None:
    """Save every attribute STK exposes for the two laser objects, so a name that differed from the script is easy to fix."""
    for kind, path, name in (("Transmitter", comm["tx"], "transmitter"), ("Receiver", comm["rx"], "receiver")):
        try:
            lines = connect(root, f"{kind}_RM {path} GetValue")
        except StkError as e:
            lines = [str(e)]
        (folder / f"{name}_attributes.txt").write_text("\n".join(lines))


def point_at_each_other(alpha, beta) -> str:
    """A 100 microradian beam only works if it is aimed: make each satellite's attitude track the other (object-model call
    from the STK documentation). The fast steering mirror is idealised as perfect pointing."""
    try:
        for sat, other in ((alpha, BETA), (beta, ALPHA)):
            p = sat.Attitude.Pointing
            p.UseTargetPointing = True
            p.Targets.RemoveAll()
            p.Targets.Add(f"Satellite/{other}")
            p.TargetTimes.UseAccessTimes = True
        return "target pointing enabled on both satellites"
    except Exception as e:
        return f"NOT SET ({e}). In STK: satellite Properties > Basic > Attitude > Pointing > Use Target Pointing, target the other node"


def access_windows(access, scenario) -> list[dict]:
    """Start/stop/duration of each line-of-sight window (data provider 'Access Data', as in AGI's own Python example)."""
    prov = access.DataProviders.GetDataPrvIntervalFromPath("Access Data")
    res = prov.ExecElements(scenario.StartTime, scenario.StopTime, ["Start Time", "Stop Time", "Duration"])
    get = lambda name: list(res.DataSets.GetDataSetByName(name).GetValues())
    starts, stops, durs = get("Start Time"), get("Stop Time"), get("Duration")
    return [dict(start=str(a), stop=str(b), duration_s=float(c)) for a, b, c in zip(starts, stops, durs)]


LINK_ELEMENTS = ["Time", "Range", "Rcvd. Iso. Power", "Carrier Power at Rcvr Input", "Eb/No", "BER", "Link Margin", "Propagation Delay"]


def link_budget_samples(access, window: dict) -> list[dict]:
    """Received power, BER and propagation delay across one access window, from STK's 'Link Information' data provider.
    Names follow the STK data-provider reference; the chain variant numbers them with a '1' suffix, tried as a fallback."""
    errors = []
    for suffix in ("", "1"):
        names = [n + suffix if n != "Time" else n for n in LINK_ELEMENTS]
        try:
            prov = access.DataProviders.GetDataPrvTimeVarFromPath("Link Information")
            res = prov.ExecElements(window["start"], window["stop"], REPORT_STEP_S, names)
            cols = {n: list(res.DataSets.GetDataSetByName(n).GetValues()) for n in names}
            rows = []
            for i in range(len(cols["Time"])):
                row = {n.rstrip("1") if suffix and n != "Time" else n: cols[n][i] for n in names}
                rows.append(row)
            return rows
        except Exception as e:
            errors.append(f"{names[2]}: {e}")
    raise StkError("Could not read 'Link Information' for the access. Open the access report in STK (Tx > Access > Link Budget) "
                   "and compare the element names.\n  " + "\n  ".join(errors))


def summarise_link(rows: list[dict]) -> dict:
    """Min / mean / max of received power (dBW), BER, latency (ms), ignoring STK's -9999.99 'no valid link' flag."""
    def col(name):
        out = []
        for r in rows:
            try:
                v = float(r[name])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(v)
        return out
    power = [v for v in col("Rcvd. Iso. Power") if v > NULL_DB]
    ber = [v for v in col("BER") if v >= 0]
    delay = col("Propagation Delay")
    rng = col("Range")
    latency_ms = [d * 1000.0 for d in delay] if delay else [(r / C_KM_S) * 1000.0 for r in rng]
    return dict(samples=len(rows), valid_samples=len(power),
                power_dbw=(min(power), sum(power) / len(power), max(power)) if power else None,
                ber=(min(ber), max(ber)) if ber else None,
                latency_ms=(min(latency_ms), max(latency_ms)) if latency_ms else None)


# ==========================================================================================================
#  3. MAIN
# ==========================================================================================================
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_stk(use_engine: bool, keep_open: bool) -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    app, root = start_stk(use_engine)
    try:
        print("Building scenario ...")
        scenario, start, stop = build_scenario(root)
        alpha, el_a = add_sso_satellite(scenario, ALPHA, NODE_ALPHA_LTAN_H, NODE_ALPHA_TRUE_ANOMALY_DEG, start)
        beta, el_b = add_sso_satellite(scenario, BETA, NODE_BETA_LTAN_H, NODE_BETA_TRUE_ANOMALY_DEG, start)
        print(f"  {ALPHA}: i = {el_a['inc_deg']:.3f} deg, RAAN = {el_a['raan_deg']:.2f} deg   {BETA}: RAAN = {el_b['raan_deg']:.2f} deg")

        print("\nNode Alpha lighting (eclipse) ...")
        try:
            rows = alpha_eclipse_times(root, start, stop, OUTPUT_DIR / "alpha_lighting_report_raw.txt")
            write_csv(OUTPUT_DIR / "alpha_lighting.csv", rows)
            umbra = [r for r in rows if r["kind"] == "Umbra"]
            pen = [r for r in rows if r["kind"] == "Penumbra"]
            print(f"  {len(umbra)} umbra passes (total {sum(r['duration_s'] for r in umbra) / 60:.1f} min), "
                  f"{len(pen)} penumbra passes (total {sum(r['duration_s'] for r in pen) / 60:.1f} min)")
            for r in umbra[:5]:
                print(f"    umbra {r['start']}  ->  {r['stop']}  ({r['duration_s'] / 60:.1f} min)")
            if not rows:
                print("  (the report came back but no rows were recognised: read alpha_lighting_report_raw.txt)")
        except StkError as e:
            print(f"  could not get the Lighting Times report: {e}")

        print("\nLaser terminals ...")
        comm = add_laser_terminals(root, scenario)
        for k, v in comm["log"].items():
            print(f"  {k:32s} {v}")
        print("  pointing:", point_at_each_other(alpha, beta))
        dump_attributes(root, comm, OUTPUT_DIR)

        print("\nAccess (line of sight) ...")
        tx_obj = scenario.Children.Item(ALPHA).Children.Item(TX_NAME)
        rx_obj = scenario.Children.Item(BETA).Children.Item(RX_NAME)
        access = tx_obj.GetAccessToObject(rx_obj)
        access.ComputeAccess()
        windows = access_windows(access, scenario)
        write_csv(OUTPUT_DIR / "access_windows.csv", windows)
        print(f"  {len(windows)} access windows")
        all_rows = []
        for i, w in enumerate(windows, 1):
            print(f"  [{i}] {w['start']} -> {w['stop']}  ({w['duration_s'] / 60:.1f} min)")
            try:
                rows = link_budget_samples(access, w)
            except StkError as e:
                print(f"      link budget unavailable: {e}")
                continue
            s = summarise_link(rows)
            all_rows += [dict(window=i, **r) for r in rows]
            if s["power_dbw"]:
                print(f"      Received power  min/mean/max: {s['power_dbw'][0]:.1f} / {s['power_dbw'][1]:.1f} / {s['power_dbw'][2]:.1f} dBW")
            else:
                print("      Received power: no valid samples (STK returned its -9999.99 flag). Check the antenna pointing and "
                      "that transmitter and receiver frequencies overlap.")
            if s["ber"]:
                print(f"      BER             best/worst: {s['ber'][0]:.2e} / {s['ber'][1]:.2e}")
            if s["latency_ms"]:
                print(f"      Latency         min/max: {s['latency_ms'][0]:.2f} / {s['latency_ms'][1]:.2f} ms (one way, propagation only)")
        write_csv(OUTPUT_DIR / "link_budget.csv", all_rows)
        try:
            root.SaveScenarioAs(str((OUTPUT_DIR / f"{SCENARIO_NAME}.sc").resolve()))
        except Exception:
            pass
        print(f"\nResults are in {OUTPUT_DIR.resolve()}")
        if not use_engine and keep_open:
            input("STK is open. Press Enter here to close it ...")
        return 0
    except StkError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    finally:
        if not keep_open:
            try:
                root.CloseScenario()
                app.ShutDown()
            except Exception:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="STK automation for the Orbital AI Datacentre optical link.")
    ap.add_argument("--offline-preview", action="store_true", help="no STK: analytic eclipse and line-of-sight estimate (needs numpy)")
    ap.add_argument("--engine", action="store_true", help="use STK Engine (no window)")
    ap.add_argument("--keep-open", action="store_true", help="leave STK open when the script finishes (desktop only)")
    args = ap.parse_args(argv)
    try:
        preview = offline_preview()
    except ImportError:
        preview = None
        print("(NumPy is not installed, so the analytic preview is skipped: pip install numpy)")
    if preview:
        print_preview(preview)
    if args.offline_preview:
        return 0 if preview else 1
    print("\n" + "=" * 70 + "\nSTK\n" + "=" * 70)
    return run_stk(args.engine, args.keep_open)


if __name__ == "__main__":
    sys.exit(main())
