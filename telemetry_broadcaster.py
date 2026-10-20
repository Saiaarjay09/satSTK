"""
telemetry_broadcaster.py - stream satSTK's live state to satCAD's digital twin (protocol v1: docs/TELEMETRY_PROTOCOL.md in the satCAD repository).

    UDP, one JSON object per datagram, satSTK -> 127.0.0.1:9876 at 30 Hz of WALL-CLOCK time (whatever the playback speed),
    satCAD -> 127.0.0.1:9877 an optional once-a-second status datagram so the UI can say "satCAD connected".

Three layers, each usable on its own:

  * pure functions   body_axes / sun_sada_angle_deg / turret_for ... and build_packet(): the maths and the schema, no sockets, no clock
  * DigitalTwinSource  turns a finished SimulationResult into a packet at ANY simulation time (it interpolates between the 10 s samples so the twin
                       moves smoothly at 30 Hz even while the engine steps are coarse)
  * TelemetryBroadcaster  the non-blocking, loopback-only UDP sender with the wall-clock rate limiter and the tiny status listener

Nothing here ever blocks or raises because nobody is listening: UDP to a closed localhost port is dropped (or answered by an ICMP refusal, which is
caught and counted). The sender refuses any address that is not loopback.

FRAME CONVENTIONS (agreed with the satCAD side)
  Body frame = LVLH.  +Z along the velocity vector, +Y radial outward (zenith, toward deep space), +X = Y x Z = the orbit normal.
  sun_vector_body_deg  = atan2(sun_z, sun_y) mod 360, the angle of the Sun in the body Y-Z plane (0 = Sun at zenith): the Solar Array Drive rotation
                         about +X that points the arrays at the Sun.
  Turret angles        for a unit direction d to the target in the body frame and a turret with outward normal n = (0, 0, s)
                         (s = +1 forward 'fwd_*', s = -1 aft 'aft_*'):   elevation = asin(s * d_z),   azimuth = s * atan2(-d_x, d_y).
                         The active turret is the one whose outward hemisphere contains the target: fwd_a if d_z >= 0, else aft_a.

WHAT IS COMPUTED AND WHAT IS ASSUMED (also in the README)
  computed by satSTK's engines: timestamp, Sun angle, the locked target (from the routing), turret azimuth/elevation (orbit, ground station and relay
      geometry), radiator temperature (orbit + eclipse + radiator engine, with a thermal-mass lag), modulation / SNR / BER / throughput (the simulation's
      end-to-end link budget for the current route).
  modelled with placeholder parameters (thermal_loop.py): core temperature, coolant flow, bypass valve, throttle mode.
  modelled/assumed: the fast-steering-mirror deflection = half of the DSP receiver's default angle-of-arrival disturbance (250 / -180 urad offset,
      15 urad rms jitter); it is not a measurement, and the per-packet jitter is random.
"""
from __future__ import annotations

import ipaddress
import json
import math
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

import thermal_loop as tl
from laser_link_engine import MODULATIONS

PROTOCOL_VERSION = 1
LOOPBACK = "127.0.0.1"
TELEMETRY_PORT = 9876
STATUS_PORT = 9877
RATE_HZ = 30.0
PAUSED_RATE_HZ = 1.0
STATUS_TIMEOUT_S = 3.0
MAX_STATUS_BYTES = 8192
RELAY_IDS = ("SOUTH-INDIA-GS", "GEO-RELAY-IND", "GEO-RELAY-ATL", "GEO-RELAY-PAC", "NONE")
TURRET_IDS = ("fwd_a", "fwd_b", "aft_a", "aft_b")
THROTTLE_MODES = (tl.NOMINAL, tl.THROTTLE, tl.SHUTDOWN)
FSM_BIAS_URAD = (250.0, -180.0)         # the DSP receiver's default angle-of-arrival offset (dsp.channel.LinkConfig)
FSM_JITTER_URAD = 15.0
FSM_RANGE_MECH_URAD = 1500.0


# ----------------------------------------------------------------------------------------------------------------------------------
# Frames and angles (pure)
# ----------------------------------------------------------------------------------------------------------------------------------
def body_axes(r_eci, v_eci) -> np.ndarray:
    """Rows are the body axes X, Y, Z expressed in the inertial frame: Y radial out, Z along the velocity (made exactly perpendicular), X = Y x Z."""
    r, v = np.asarray(r_eci, dtype=float), np.asarray(v_eci, dtype=float)
    y = r / np.linalg.norm(r)
    z = v - np.dot(v, y) * y
    z /= np.linalg.norm(z)
    return np.vstack([np.cross(y, z), y, z])


def to_body(vec_inertial, axes: np.ndarray) -> np.ndarray:
    return axes @ np.asarray(vec_inertial, dtype=float)


def wrap180(deg: float) -> float:
    """Wrap an angle into (-180, 180]."""
    w = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def sun_sada_angle_deg(sun_body) -> float:
    """Solar-array drive angle, 0 to 360 degrees: the Sun's angle in the body Y-Z plane, atan2(sun_z, sun_y). 0 = Sun at zenith, 90 = ahead (+Z)."""
    return math.degrees(math.atan2(sun_body[2], sun_body[1])) % 360.0


def turret_for(d_body) -> tuple:
    """Active turret (fwd_a or aft_a) for a target direction in the body frame, with its azimuth and elevation in degrees: (id, azimuth, elevation)."""
    d = np.asarray(d_body, dtype=float)
    d = d / np.linalg.norm(d)
    s = 1.0 if d[2] >= 0.0 else -1.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, s * d[2]))))
    az = wrap180(s * math.degrees(math.atan2(-d[0], d[1])))
    return ("fwd_a" if s > 0 else "aft_a"), az, el


def relay_id_for(short_name: str) -> str:
    """'GEO-IND' -> 'GEO-RELAY-IND' (the protocol's fixed identifiers)."""
    s = short_name.upper()
    return "GEO-RELAY-" + s[4:] if s.startswith("GEO-") else s


# ----------------------------------------------------------------------------------------------------------------------------------
# The packet (pure)
# ----------------------------------------------------------------------------------------------------------------------------------
def _num(x, lo: float, hi: float, nd: int = 3, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return round(min(max(v, lo), hi), nd)


def _wrap360(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return round(v % 360.0, 2) % 360.0 if math.isfinite(v) else 0.0


def _ber(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return float(f"{min(max(v, 0.0), 1.0):.3g}")


def format_timestamp(t) -> str:
    """ISO 8601 UTC with milliseconds and a trailing Z; accepts a datetime or an already-formatted string."""
    if isinstance(t, str):
        return t
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def build_packet(*, seq: int, timestamp_utc, sun_vector_body_deg: float, active_relay_id: str, active_turret_id: str, turret_angles_deg: dict,
                 thermal_state: dict, dsp_metrics: dict) -> dict:
    """
    Assemble one protocol-v1 telemetry packet from plain values. Pure: no sockets, no clock. Every number is made finite, clamped to the range the
    receiver accepts and rounded (so a packet stays well under 1 KB); an unknown relay / turret / throttle name is replaced by a safe default.
    """
    ta = turret_angles_deg or {}
    th = thermal_state or {}
    dm = dsp_metrics or {}
    mode = th.get("throttle_mode", tl.NOMINAL)
    return {
        "v": PROTOCOL_VERSION,
        "seq": int(seq),
        "timestamp_utc": format_timestamp(timestamp_utc),
        "sun_vector_body_deg": _wrap360(sun_vector_body_deg),
        "active_relay_id": active_relay_id if active_relay_id in RELAY_IDS else "NONE",
        "active_turret_id": active_turret_id if active_turret_id in TURRET_IDS else "fwd_a",
        "turret_angles_deg": {"azimuth": _num(ta.get("azimuth", 0.0), -720.0, 720.0, 3), "elevation": _num(ta.get("elevation", 0.0), -90.0, 90.0, 3),
                              "fsm_x_urad": _num(ta.get("fsm_x_urad", 0.0), -5000.0, 5000.0, 2), "fsm_y_urad": _num(ta.get("fsm_y_urad", 0.0), -5000.0, 5000.0, 2)},
        "thermal_state": {"core_temp_c": _num(th.get("core_temp_c", 0.0), -273.0, 2000.0, 2), "radiator_temp_c": _num(th.get("radiator_temp_c", 0.0), -273.0, 2000.0, 2),
                          "coolant_flow_lpm": _num(th.get("coolant_flow_lpm", 0.0), 0.0, 1000.0, 2), "bypass_valve_pct": _num(th.get("bypass_valve_pct", 0.0), 0.0, 100.0, 1),
                          "throttle_mode": mode if mode in THROTTLE_MODES else tl.NOMINAL},
        "dsp_metrics": {"modulation": str(dm.get("modulation", "NONE"))[:16], "snr_db": _num(dm.get("snr_db", 0.0), -200.0, 200.0, 2),
                        "ber": _ber(dm.get("ber", 0.5)), "throughput_gbps": _num(dm.get("throughput_gbps", 0.0), 0.0, 1.0e6, 2)},
    }


def encode_packet(packet: dict) -> bytes:
    return json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")


# ----------------------------------------------------------------------------------------------------------------------------------
# From a simulation to packets
# ----------------------------------------------------------------------------------------------------------------------------------
class FsmModel:
    """Fast-steering-mirror deflection: steady state is half the angle-of-arrival disturbance (the beam moves twice the mirror tilt) plus its jitter."""

    def __init__(self, seed: int = 1, bias_urad=FSM_BIAS_URAD, jitter_urad: float = FSM_JITTER_URAD):
        self.rng = np.random.default_rng(seed)
        self.bias, self.jitter = bias_urad, jitter_urad

    def sample(self, locked: bool) -> tuple:
        if not locked:
            return 0.0, 0.0
        x = (self.bias[0] + self.rng.normal(0.0, self.jitter)) / 2.0
        y = (self.bias[1] + self.rng.normal(0.0, self.jitter)) / 2.0
        lim = FSM_RANGE_MECH_URAD
        return float(np.clip(x, -lim, lim)), float(np.clip(y, -lim, lim))


class DigitalTwinSource:
    """Packets for a finished `SimulationResult` at any simulation time `t` (seconds since the scenario start)."""

    def __init__(self, result, fsm_seed: int = 1, loop_cfg: tl.ThermalLoopConfig | None = None):
        self.result = result
        self.step = result.config.step_s
        self.n = result.n
        self.track = result.tracks[result.target]
        self.thermal = tl.simulate_loop(self.track.radiator_solar_load_w, self.step, result.config.thermal, loop_cfg)
        self.fsm = FsmModel(fsm_seed)
        self._names = list(MODULATIONS)

    # -- helpers --------------------------------------------------------------------------------------------------------------
    def _locate(self, t: float):
        f = min(max(t / self.step, 0.0), self.n - 1.0)
        i0 = int(math.floor(f))
        i1 = min(i0 + 1, self.n - 1)
        return f, i0, i1, f - i0

    @staticmethod
    def _lerp(arr, i0, i1, w):
        return arr[i0] * (1.0 - w) + arr[i1] * w

    def peer_at(self, i: int):
        """What the satellite's laser terminal is talking to at sample i: ('GS' | relay index | None)."""
        res = self.result
        state, _ = res.beam_state(i)
        if state == "blocked" or res.route.chosen[i] < 0:
            return None
        nodes = res.route.nodes(i)
        return nodes[-1] if nodes else "GS"

    # -- the packet ------------------------------------------------------------------------------------------------------------
    def snapshot(self, t: float) -> dict:
        """All the raw values for time t (before packaging): useful for tests and for the HUD."""
        res = self.result
        f, i0, i1, w = self._locate(t)
        inear = int(round(f))
        sat_r = self._lerp(self.track.r_eci, i0, i1, w)
        sat_v = self._lerp(self.track.v_eci, i0, i1, w)
        axes = body_axes(sat_r, sat_v)
        sun = self._lerp(res.sun_eci, i0, i1, w)
        sun_body = to_body(sun / np.linalg.norm(sun), axes)
        peer = self.peer_at(inear)
        if peer is None:
            relay_id, turret_id, az, el, fx, fy = "NONE", "fwd_a", 0.0, 0.0, 0.0, 0.0
        else:
            if peer == "GS":
                pos, relay_id = self._lerp(res.station_eci, i0, i1, w), "SOUTH-INDIA-GS"
            else:
                rt = res.relay_tracks[peer]
                pos, relay_id = self._lerp(rt.r_eci, i0, i1, w), relay_id_for(rt.short)
            d = to_body(pos - sat_r, axes)
            turret_id, az, el = turret_for(d)
            fx, fy = self.fsm.sample(True)
        th = self.thermal
        link = res.primary_link
        best = int(link.adaptive_modulation[inear])
        locked = bool(link.locked[inear])
        if locked and best >= 0:
            mod = self._names[best]
            ber = float(link.ber[mod][inear])
            thr = float(link.adaptive_throughput_gbps[inear])
        else:
            mod, ber, thr = "NONE", 0.5, 0.0
        snr = float(link.snr_db[inear]) if np.isfinite(link.snr_db[inear]) else 0.0
        return dict(
            timestamp_utc=res.utc(t), sun_vector_body_deg=sun_sada_angle_deg(sun_body), sun_body=sun_body, active_relay_id=relay_id, active_turret_id=turret_id,
            turret_angles_deg=dict(azimuth=az, elevation=el, fsm_x_urad=fx, fsm_y_urad=fy),
            thermal_state=dict(core_temp_c=self._lerp(th.core_temp_c, i0, i1, w), radiator_temp_c=self._lerp(th.radiator_temp_c, i0, i1, w),
                               coolant_flow_lpm=self._lerp(th.coolant_flow_lpm, i0, i1, w), bypass_valve_pct=self._lerp(th.bypass_valve_pct, i0, i1, w),
                               throttle_mode=th.throttle_mode[inear]),
            dsp_metrics=dict(modulation=mod, snr_db=snr, ber=ber, throughput_gbps=thr))

    def packet_at(self, t: float, seq: int) -> dict:
        s = self.snapshot(t)
        return build_packet(seq=seq, timestamp_utc=s["timestamp_utc"], sun_vector_body_deg=s["sun_vector_body_deg"], active_relay_id=s["active_relay_id"],
                            active_turret_id=s["active_turret_id"], turret_angles_deg=s["turret_angles_deg"], thermal_state=s["thermal_state"],
                            dsp_metrics=s["dsp_metrics"])


# ----------------------------------------------------------------------------------------------------------------------------------
# The sender
# ----------------------------------------------------------------------------------------------------------------------------------
def check_loopback(host: str) -> str:
    """Return the IPv4 loopback address for `host`, or raise ValueError: telemetry never leaves this computer."""
    if host == "localhost":
        return LOOPBACK
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(f"refusing {host!r}: telemetry is loopback-only (127.0.0.1)") from None
    if ip.version != 4 or not ip.is_loopback:
        raise ValueError(f"refusing {host!r}: telemetry is loopback-only (127.0.0.1)")
    return str(ip)


@dataclass
class TwinStatus:
    connected: bool                 # a satCAD status datagram arrived in the last STATUS_TIMEOUT_S seconds
    info: dict | None               # the last status datagram (model name, frames, ...)
    age_s: float | None


class TelemetryBroadcaster:
    """
    Non-blocking loopback UDP sender with a wall-clock rate limiter and an optional status listener.

        b = TelemetryBroadcaster()
        b.tick(lambda seq: source.packet_at(t, seq), playing=True)      # call as often as you like; it sends at most 30 times a second

    `clock` is injectable (a function returning seconds) so the rate limiter can be tested without waiting.
    """

    def __init__(self, host: str = LOOPBACK, port: int = TELEMETRY_PORT, status_port: int = STATUS_PORT, rate_hz: float = RATE_HZ,
                 paused_rate_hz: float = PAUSED_RATE_HZ, listen: bool = True, clock=time.monotonic):
        self.host = check_loopback(host)
        self.port, self.status_port = int(port), int(status_port)
        self.period, self.paused_period = 1.0 / rate_hz, 1.0 / paused_rate_hz
        self._clock = clock
        self._next_due: float | None = None
        self.seq = 0
        self.sent = 0
        self.failed = 0
        self.last_error = ""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._status_lock = threading.Lock()
        self._status_info: dict | None = None
        self._status_time: float | None = None
        self._stop = threading.Event()
        self._feed_stop = threading.Event()
        self._feed: threading.Thread | None = None
        self._listener: threading.Thread | None = None
        self._lsock: socket.socket | None = None
        self.listener_error = ""
        if listen:
            self.start_listener()

    # -- sending --------------------------------------------------------------------------------------------------------------
    def due(self, playing: bool = True) -> bool:
        return self._next_due is None or self._clock() >= self._next_due

    def _advance(self, playing: bool) -> None:
        now = self._clock()
        period = self.period if playing else self.paused_period
        if self._next_due is None:
            self._next_due = now + period
        else:
            self._next_due += period
            if now - self._next_due > period:                # we fell behind (a long pause in the caller): do not burst to catch up
                self._next_due = now + period

    def send(self, packet) -> bool:
        """Send one packet (dict or bytes) without ever blocking or raising. Returns True if the operating system accepted it."""
        try:
            data = packet if isinstance(packet, (bytes, bytearray)) else encode_packet(packet)
            self._sock.sendto(data, (self.host, self.port))
        except (BlockingIOError, InterruptedError):
            self.failed += 1
            self.last_error = "send buffer full (packet dropped)"
            return False
        except OSError as exc:                               # e.g. ConnectionRefusedError after an ICMP 'port unreachable': nobody is listening
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        except (TypeError, ValueError) as exc:               # an unserialisable packet is a bug in the caller, but it must still not take the app down
            self.failed += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.sent += 1
        return True

    def tick(self, make_packet, playing: bool = True) -> bool:
        """If a packet is due (30 Hz while playing, 1 Hz paused), build it with `make_packet(seq)` and send it. Returns True if one was sent."""
        if not self.due(playing):
            return False
        self._advance(playing)
        try:
            packet = make_packet(self.seq)
        except Exception as exc:                             # a failure building one packet never stops the stream or the host application
            self.failed += 1
            self.last_error = f"packet build failed: {type(exc).__name__}: {exc}"
            return False
        if packet is None:                                   # nothing to say right now (for example no scenario computed yet)
            return False
        self.seq += 1
        return self.send(packet)

    def start_feed(self, make_packet, is_playing, poll_s: float = 0.002) -> None:
        """
        Run the stream on its own thread: `make_packet(seq)` builds a packet (or returns None), `is_playing()` says whether to run at 30 Hz or 1 Hz.
        A window that is busy painting cannot starve the stream this way, so the wall-clock rate holds under load.
        """
        if self._feed is not None:
            return
        self._feed_stop.clear()

        def loop():
            while not self._feed_stop.is_set():
                try:
                    self.tick(make_packet, bool(is_playing()))
                except Exception as exc:                     # the feed must outlive any single bad tick
                    self.last_error = f"feed: {type(exc).__name__}: {exc}"
                time.sleep(poll_s)

        self._feed = threading.Thread(target=loop, name="satstk-twin-feed", daemon=True)
        self._feed.start()

    def stop_feed(self) -> None:
        self._feed_stop.set()
        if self._feed is not None:
            self._feed.join(timeout=1.0)
            self._feed = None

    # -- the status listener --------------------------------------------------------------------------------------------------------
    def start_listener(self) -> bool:
        """Listen on 127.0.0.1:status_port for satCAD's status datagrams in a daemon thread. If the port is taken the listener is simply off."""
        if self._listener is not None:
            return True
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.status_port))
            s.settimeout(0.2)
        except OSError as exc:
            self.listener_error = f"status listener off: {type(exc).__name__}: {exc}"
            return False
        self._lsock = s
        if self.status_port == 0:
            self.status_port = s.getsockname()[1]
        self._listener = threading.Thread(target=self._listen, name="satstk-twin-status", daemon=True)
        self._listener.start()
        return True

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._lsock.recvfrom(MAX_STATUS_BYTES + 1)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(data) > MAX_STATUS_BYTES:
                continue
            try:
                obj = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(obj, dict) and obj.get("app") == "satCAD":
                with self._status_lock:
                    self._status_info, self._status_time = obj, self._clock()

    @property
    def status(self) -> TwinStatus:
        with self._status_lock:
            info, t = self._status_info, self._status_time
        if t is None:
            return TwinStatus(False, None, None)
        age = self._clock() - t
        return TwinStatus(age <= STATUS_TIMEOUT_S, info, age)

    def status_text(self) -> str:
        st = self.status
        if st.connected:
            frames = st.info.get("frames") if st.info else None
            return "Digital twin: satCAD connected" + (f" ({frames} frames shown)" if isinstance(frames, int) else "")
        return f"Digital twin: broadcasting to {self.host}:{self.port}, satCAD not connected"

    def close(self) -> None:
        self.stop_feed()
        self._stop.set()
        for s in (self._lsock, self._sock):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        if self._listener is not None:
            self._listener.join(timeout=1.0)
            self._listener = None
