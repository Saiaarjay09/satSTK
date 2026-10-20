"""Digital-twin telemetry: the packet schema, the 30 Hz wall-clock limiter, the loopback-only non-blocking sender, the angle conventions, and real UDP."""
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import events as ev
import simulation as sim
import telemetry_broadcaster as tb
import thermal_loop as tl
from helpers import direct_cfg

ROOT = Path(__file__).resolve().parent.parent


# ---- helpers --------------------------------------------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def validate_schema(p: dict) -> None:
    """The contract from docs/TELEMETRY_PROTOCOL.md (types and ranges), written out independently of the code under test."""
    assert p["v"] == 1 and isinstance(p["seq"], int) and p["seq"] >= 0
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", p["timestamp_utc"])
    assert isinstance(p["sun_vector_body_deg"], float) and 0.0 <= p["sun_vector_body_deg"] < 360.0
    assert p["active_relay_id"] in ("SOUTH-INDIA-GS", "GEO-RELAY-IND", "GEO-RELAY-ATL", "GEO-RELAY-PAC", "NONE")
    assert p["active_turret_id"] in ("fwd_a", "fwd_b", "aft_a", "aft_b")
    ta = p["turret_angles_deg"]
    assert set(ta) == {"azimuth", "elevation", "fsm_x_urad", "fsm_y_urad"} and all(isinstance(v, float) and math.isfinite(v) for v in ta.values())
    assert -180.0 <= ta["azimuth"] <= 180.0 and -90.0 <= ta["elevation"] <= 90.0 and abs(ta["fsm_x_urad"]) <= 5000 and abs(ta["fsm_y_urad"]) <= 5000
    th = p["thermal_state"]
    assert set(th) == {"core_temp_c", "radiator_temp_c", "coolant_flow_lpm", "bypass_valve_pct", "throttle_mode"}
    assert all(isinstance(th[k], float) and math.isfinite(th[k]) for k in th if k != "throttle_mode")
    assert 0.0 <= th["bypass_valve_pct"] <= 100.0 and th["coolant_flow_lpm"] >= 0.0 and th["throttle_mode"] in ("NOMINAL", "THROTTLE", "SHUTDOWN")
    dm = p["dsp_metrics"]
    assert set(dm) == {"modulation", "snr_db", "ber", "throughput_gbps"} and isinstance(dm["modulation"], str) and dm["modulation"]
    assert 0.0 <= dm["ber"] <= 1.0 and dm["throughput_gbps"] >= 0.0 and math.isfinite(dm["snr_db"])
    assert len(tb.encode_packet(p)) < 1024


@pytest.fixture(scope="module")
def world():
    r = sim.run_simulation(sim.SimulationConfig())
    return r, tb.DigitalTwinSource(r)


def udp_receiver():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(1.0)
    return s, s.getsockname()[1]


def drain(s, limit=10000):
    s.setblocking(False)
    out = []
    try:
        while len(out) < limit:
            out.append(json.loads(s.recv(65535).decode("utf-8")))
    except BlockingIOError:
        pass
    return out


# ---- packet schema --------------------------------------------------------------------------------------------------------------------
def test_every_packet_from_the_simulation_matches_the_protocol(world):
    r, src = world
    for t in np.linspace(0.0, r.t_s[-1], 400):
        validate_schema(src.packet_at(float(t), 7))


def test_packets_are_small_and_strict_json(world):
    r, src = world
    sizes = [len(tb.encode_packet(src.packet_at(float(t), 1))) for t in np.linspace(0, r.t_s[-1], 100)]
    assert max(sizes) < 600 and min(sizes) > 300
    raw = tb.encode_packet(src.packet_at(100.0, 3))
    assert json.loads(raw.decode("utf-8"))["seq"] == 3
    assert b"NaN" not in raw and b"Infinity" not in raw


def test_build_packet_sanitises_bad_values():
    p = tb.build_packet(seq=5, timestamp_utc="2026-12-21T00:00:00.000Z", sun_vector_body_deg=725.0, active_relay_id="BOGUS", active_turret_id="zzz",
                        turret_angles_deg={"azimuth": float("nan"), "elevation": 500.0, "fsm_x_urad": 1e9, "fsm_y_urad": float("inf")},
                        thermal_state={"core_temp_c": float("nan"), "radiator_temp_c": 20.0, "coolant_flow_lpm": -3.0, "bypass_valve_pct": 250.0, "throttle_mode": "MELTDOWN"},
                        dsp_metrics={"modulation": "QPSK", "snr_db": float("nan"), "ber": 3.0, "throughput_gbps": -1.0})
    validate_schema(p)
    assert p["active_relay_id"] == "NONE" and p["active_turret_id"] == "fwd_a" and p["thermal_state"]["throttle_mode"] == "NOMINAL"
    assert p["sun_vector_body_deg"] == pytest.approx(5.0) and p["turret_angles_deg"]["elevation"] == 90.0 and p["turret_angles_deg"]["fsm_x_urad"] == 5000.0
    assert p["thermal_state"]["bypass_valve_pct"] == 100.0 and p["dsp_metrics"]["ber"] == 1.0 and p["dsp_metrics"]["throughput_gbps"] == 0.0
    from datetime import datetime, timezone
    assert tb.build_packet(seq=1, timestamp_utc=datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc), sun_vector_body_deg=0, active_relay_id="NONE",
                           active_turret_id="fwd_a", turret_angles_deg={}, thermal_state={}, dsp_metrics={})["timestamp_utc"] == "2026-01-02T03:04:05.678Z"


@pytest.mark.skipif(not (Path(os.environ.get("SATCAD_DIR", Path.home() / "Developer" / "Open-source-CAD-software")) / ".venv" / "bin" / "python").exists(),
                    reason="satCAD checkout with its own virtualenv not found (set SATCAD_DIR)")
def test_packets_are_accepted_by_the_real_satcad_parser(world, tmp_path):
    r, src = world
    satcad = Path(os.environ.get("SATCAD_DIR", Path.home() / "Developer" / "Open-source-CAD-software"))
    lines = [tb.encode_packet(src.packet_at(float(t), k)).decode() for k, t in enumerate(np.linspace(0, r.t_s[-1], 60))]
    f = tmp_path / "packets.jsonl"
    f.write_text("\n".join(lines), encoding="utf-8")
    code = ("import sys; sys.path.insert(0, '.'); from satcad.digital_twin import parse_packet\n"
            "n = sum(1 for l in open(sys.argv[1], encoding='utf-8') if parse_packet(l.strip()))\nprint('ACCEPTED', n)")
    out = subprocess.run([str(satcad / ".venv" / "bin" / "python"), "-c", code, str(f)], cwd=satcad, capture_output=True, text=True, timeout=120)
    if "ModuleNotFoundError" in out.stderr or "ImportError" in out.stderr or "digital_twin" in out.stderr and "No module" in out.stderr:
        pytest.skip("satCAD receiver module not importable here")
    assert "ACCEPTED 60" in out.stdout, out.stderr[-500:]


# ---- angle conventions, by hand -------------------------------------------------------------------------------------------------------
def test_body_axes_for_a_circular_equatorial_orbit():
    axes = tb.body_axes([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    assert axes == pytest.approx(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))            # X = orbit normal (+z), Y = radial out, Z = velocity
    assert np.allclose(np.cross(axes[1], axes[2]), axes[0]) and np.linalg.det(axes) == pytest.approx(1.0)      # X = Y x Z and right-handed


def test_body_axes_stay_orthonormal_with_a_radial_velocity_component():
    axes = tb.body_axes([4000.0, 5000.0, 3000.0], [-3.0, 2.0, 5.0])
    assert axes @ axes.T == pytest.approx(np.eye(3), abs=1e-12)
    assert axes[1] == pytest.approx(np.array([4000.0, 5000.0, 3000.0]) / np.linalg.norm([4000.0, 5000.0, 3000.0]))


def test_sun_angle_is_the_angle_in_the_y_z_plane_zero_at_zenith():
    # body-frame Sun directions (x, y, z) -> expected SADA angle
    for sun, expected in (((0, 1, 0), 0.0), ((0, 0, 1), 90.0), ((0, -1, 0), 180.0), ((0, 0, -1), 270.0), ((0, 1, 1), 45.0), ((0, -1, 1), 135.0),
                          ((0, -1, -1), 225.0), ((0, 1, -1), 315.0), ((0.8, 0.6, 0.0), 0.0), ((0.5, 0.0, 0.866), 90.0)):
        assert tb.sun_sada_angle_deg(sun) == pytest.approx(expected, abs=1e-9)


def test_sun_angle_from_orbit_vectors():
    axes = tb.body_axes([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    assert tb.sun_sada_angle_deg(tb.to_body([1.0, 0.0, 0.0], axes)) == pytest.approx(0.0)            # Sun straight overhead
    assert tb.sun_sada_angle_deg(tb.to_body([0.0, 1.0, 0.0], axes)) == pytest.approx(90.0)           # Sun along the flight direction
    assert tb.sun_sada_angle_deg(tb.to_body([-1.0, 0.0, 0.0], axes)) == pytest.approx(180.0)         # Sun behind the Earth (eclipse side)


@pytest.mark.parametrize("d,expected", [
    ((0, 1, 0), ("fwd_a", 0.0, 0.0)),                       # home boresight: +Y (zenith)
    ((0, 0, 1), ("fwd_a", 0.0, 90.0)),                      # straight ahead along +Z
    ((1, 0, 0), ("fwd_a", -90.0, 0.0)),                     # along the orbit normal: azimuth = atan2(-1, 0) = -90
    ((-1, 0, 0), ("fwd_a", 90.0, 0.0)),
    ((0, -1, 0), ("fwd_a", 180.0, 0.0)),                    # nadir (toward the Earth)
    ((0.5, 0.5, math.sqrt(0.5)), ("fwd_a", -45.0, 45.0)),   # forward hemisphere: el = asin(0.7071), az = atan2(-0.5, 0.5)
    ((0.5, 0.5, -math.sqrt(0.5)), ("aft_a", 45.0, 45.0)),   # aft hemisphere (s = -1): el = asin(0.7071), az = -atan2(-0.5, 0.5) = +45
    ((0, 0, -1), ("aft_a", 0.0, 90.0)),
    ((0, 0.6, -0.8), ("aft_a", 0.0, math.degrees(math.asin(0.8)))),
    ((-0.6, 0.0, 0.8), ("fwd_a", 90.0, math.degrees(math.asin(0.8)))),
])
def test_turret_angles_match_hand_computed_cases(d, expected):
    tid, az, el = tb.turret_for(d)
    assert tid == expected[0] and az == pytest.approx(expected[1], abs=1e-9) and el == pytest.approx(expected[2], abs=1e-9)


def test_active_turret_is_the_one_whose_hemisphere_holds_the_target():
    rng = np.random.default_rng(1)
    for d in rng.normal(size=(500, 3)):
        d /= np.linalg.norm(d)
        tid, az, el = tb.turret_for(d)
        assert tid == ("fwd_a" if d[2] >= 0 else "aft_a") and 0.0 <= el <= 90.0 and -180.0 < az <= 180.0
        s = 1.0 if tid == "fwd_a" else -1.0
        assert math.sin(math.radians(el)) == pytest.approx(s * d[2])                             # elevation = asin(s d_z)


def test_relay_id_mapping_matches_the_protocol_names():
    assert tb.relay_id_for("GEO-IND") == "GEO-RELAY-IND" and tb.relay_id_for("GEO-ATL") == "GEO-RELAY-ATL" and tb.relay_id_for("GEO-PAC") == "GEO-RELAY-PAC"


# ---- rate limiter (wall clock, independent of playback speed) -----------------------------------------------------------------------
@pytest.mark.parametrize("playback_speed", [1, 60, 3600])
def test_rate_limiter_gives_thirty_packets_per_wall_second_at_any_playback_speed(playback_speed):
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, listen=False, clock=clock)
    sim_time, sent = 0.0, 0
    for _ in range(1000):                                     # poll every millisecond for one second of wall time
        clock.t += 0.001
        sim_time += playback_speed * 0.001                    # the simulation clock races ahead, the limiter must not care
        sent += b.tick(lambda seq: {"seq": seq, "t": sim_time}, playing=True)
    b.close()
    assert 29 <= sent <= 31


def test_rate_limiter_is_steady_over_irregular_polling():
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, listen=False, clock=clock)
    rng = np.random.default_rng(2)
    sent, elapsed = 0, 0.0
    while elapsed < 10.0:
        dt = float(rng.uniform(0.002, 0.025))                 # a GUI timer that jitters between 2 and 25 ms
        clock.t += dt
        elapsed += dt
        sent += b.tick(lambda seq: {}, playing=True)
    b.close()
    assert 270 <= sent <= 310                                 # about 30 Hz over 10 s even though polling is irregular


def test_paused_simulation_sends_at_one_hertz():
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, listen=False, clock=clock)
    sent = sum(b.tick(lambda seq: {}, playing=False) for _ in range(5000) if not clock.__setattr__("t", clock.t + 0.001))
    b.close()
    assert 4 <= sent <= 6                                     # 5 s of wall time at 1 Hz


def test_a_stalled_caller_does_not_trigger_a_burst():
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, listen=False, clock=clock)
    assert b.tick(lambda seq: {}) is True
    clock.t += 5.0                                            # the application froze for five seconds
    sent = 0
    for _ in range(20):
        sent += b.tick(lambda seq: {})
    assert sent == 1                                          # one packet, not 150 catch-up packets
    b.close()


def test_real_clock_rate_is_about_thirty_hertz():
    b = tb.TelemetryBroadcaster(port=1, listen=False)
    t0 = time.monotonic()
    sent = 0
    while time.monotonic() - t0 < 1.0:
        sent += b.tick(lambda seq: {}, playing=True)
        time.sleep(0.002)
    b.close()
    assert 24 <= sent <= 36


def test_sequence_numbers_count_up_by_one():
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, listen=False, clock=clock)
    seen = []
    for _ in range(300):
        clock.t += 0.01
        b.tick(lambda seq: seen.append(seq) or {})
    b.close()
    assert seen == list(range(len(seen))) and len(seen) > 80


# ---- safe to run with nobody listening ----------------------------------------------------------------------------------------------
def test_sending_with_no_listener_never_blocks_or_raises(world):
    r, src = world
    b = tb.TelemetryBroadcaster(port=59998, status_port=0)         # nobody is listening on 59998
    t0 = time.monotonic()
    for k in range(2000):
        b.send(src.packet_at(100.0, k))                            # may be accepted, or refused by an ICMP reply: either way no exception
    assert time.monotonic() - t0 < 1.5
    assert b.sent + b.failed == 2000
    b.close()


def test_a_failing_packet_builder_is_contained():
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, listen=False, clock=clock)
    assert b.tick(lambda seq: 1 / 0) is False and b.failed == 1 and "packet build failed" in b.last_error
    clock.t += 1.0
    assert b.tick(lambda seq: object()) is False                    # an unserialisable packet is also contained
    b.close()


def test_a_closed_broadcaster_still_does_not_raise():
    b = tb.TelemetryBroadcaster(port=1, listen=False)
    b.close()
    assert b.send({"a": 1}) is False


# ---- loopback only -------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("host", ["192.168.1.20", "10.0.0.5", "8.8.8.8", "0.0.0.0", "255.255.255.255", "example.com", "::1", "", "224.0.0.1"])
def test_non_loopback_addresses_are_refused(host):
    with pytest.raises(ValueError, match="loopback"):
        tb.TelemetryBroadcaster(host=host, listen=False)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.2"])
def test_loopback_addresses_are_accepted(host):
    b = tb.TelemetryBroadcaster(host=host, port=1, listen=False)
    assert b.host.startswith("127.")
    b.close()


# ---- real UDP --------------------------------------------------------------------------------------------------------------------------------
def test_udp_round_trip_on_an_ephemeral_port(world):
    r, src = world
    rx, port = udp_receiver()
    b = tb.TelemetryBroadcaster(port=port, status_port=0)
    sent = src.packet_at(2400.0, 41)
    assert b.send(sent) is True
    got = json.loads(rx.recv(65535).decode("utf-8"))
    assert got == sent
    validate_schema(got)
    b.close()
    rx.close()


def test_a_stream_arrives_complete_and_in_order_over_loopback(world):
    r, src = world
    rx, port = udp_receiver()
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=port, listen=False, clock=clock)
    for _ in range(600):
        clock.t += 0.005
        b.tick(lambda seq: src.packet_at(100.0 + seq * 2.0, seq), playing=True)
    got = drain(rx)
    assert len(got) == b.sent and len(got) > 80
    assert [p["seq"] for p in got] == list(range(len(got)))
    assert all(a["timestamp_utc"] <= c["timestamp_utc"] for a, c in zip(got, got[1:]))
    b.close()
    rx.close()


def test_status_datagrams_from_satcad_are_recorded_and_time_out():
    clock = FakeClock()
    b = tb.TelemetryBroadcaster(port=1, status_port=0, clock=clock)
    assert b.listener_error == "" and b.status_port > 0 and not b.status.connected
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.sendto(b"not json", ("127.0.0.1", b.status_port))
    client.sendto(json.dumps({"app": "something else"}).encode(), ("127.0.0.1", b.status_port))
    client.sendto(b"x" * 8500, ("127.0.0.1", b.status_port))
    client.sendto(json.dumps({"v": 1, "app": "satCAD", "twin": "online", "model": "Orbital-Node-Alpha", "frames": 1830, "bad_packets": 0, "last_seq": 7}).encode(),
                  ("127.0.0.1", b.status_port))
    deadline = time.time() + 3.0
    while not b.status.connected and time.time() < deadline:
        time.sleep(0.01)
    st = b.status
    assert st.connected and st.info["model"] == "Orbital-Node-Alpha" and st.info["frames"] == 1830
    assert "satCAD connected" in b.status_text() and "1830" in b.status_text()
    clock.t += 2.9
    assert b.status.connected
    clock.t += 0.3                                                    # more than 3 s of silence
    assert not b.status.connected and "not connected" in b.status_text()
    client.close()
    b.close()


def test_a_busy_status_port_just_turns_the_listener_off():
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("127.0.0.1", 0))                                      # no SO_REUSEADDR on this one
    port = holder.getsockname()[1]
    b = tb.TelemetryBroadcaster(port=1, status_port=port)
    assert (b.listener_error == "" and b.status_port == port) or "off" in b.listener_error       # platform dependent: either way no exception
    assert not b.status.connected
    b.close()
    holder.close()


# ---- values derived from the engines ---------------------------------------------------------------------------------------------------
def _pick(r, kind, frac=0.5):
    w = np.flatnonzero(r.route.kind == kind)
    return int(w[int(len(w) * frac)])


def test_locked_target_follows_the_routing(world):
    r, src = world
    step = r.config.step_s
    i1, i2, i3 = _pick(r, 1), _pick(r, 2), _pick(r, 3)
    assert src.packet_at(i1 * step, 0)["active_relay_id"] == "SOUTH-INDIA-GS"                     # direct pass: the satellite talks to the station
    assert src.packet_at(i2 * step, 0)["active_relay_id"] == "GEO-RELAY-IND"                      # one relay: the satellite's partner is that relay
    last = r.relay_tracks[r.route.nodes(i3)[-1]].short
    assert src.packet_at(i3 * step, 0)["active_relay_id"] == tb.relay_id_for(last) and last in ("GEO-ATL", "GEO-PAC")      # two relays: the second one


def test_no_link_gives_none_and_a_home_pose():
    r = sim.run_simulation(direct_cfg())                                                          # relays off: mostly no route
    src = tb.DigitalTwinSource(r)
    i = int(np.flatnonzero(~r.primary_link.visible)[50])
    p = src.packet_at(i * r.config.step_s, 0)
    assert p["active_relay_id"] == "NONE" and p["turret_angles_deg"] == {"azimuth": 0.0, "elevation": 0.0, "fsm_x_urad": 0.0, "fsm_y_urad": 0.0}
    assert p["dsp_metrics"] == {"modulation": "NONE", "snr_db": p["dsp_metrics"]["snr_db"], "ber": 0.5, "throughput_gbps": 0.0}
    validate_schema(p)


def test_an_event_that_blocks_the_beam_drops_the_lock():
    base = sim.run_simulation(sim.SimulationConfig())
    i = _pick(base, 2, 0.3)
    t = float(base.t_s[i])
    r = sim.run_simulation(sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("flock", t - 60.0, 120.0, 0.5)])))
    assert tb.DigitalTwinSource(base).packet_at(t, 0)["active_relay_id"] == "GEO-RELAY-IND"
    assert tb.DigitalTwinSource(r).packet_at(t, 0)["active_relay_id"] == "NONE"


def test_turret_angles_equal_the_formula_applied_to_the_simulation_geometry(world):
    """Recompute the pointing independently, straight from the stored orbit arrays, at sample times (no interpolation)."""
    r, src = world
    for kind in (1, 2, 3):
        for frac in (0.1, 0.5, 0.9):
            i = _pick(r, kind, frac)
            p = src.packet_at(i * r.config.step_s, 0)
            sat_r, sat_v = r.tracks[0].r_eci[i], r.tracks[0].v_eci[i]
            y = sat_r / np.linalg.norm(sat_r)
            z = sat_v - (sat_v @ y) * y
            z /= np.linalg.norm(z)
            x = np.cross(y, z)
            nodes = r.route.nodes(i)
            peer = r.station_eci[i] if not nodes else r.relay_tracks[nodes[-1]].r_eci[i]
            dvec = peer - sat_r
            d = np.array([dvec @ x, dvec @ y, dvec @ z]) / np.linalg.norm(dvec)
            s = 1.0 if d[2] >= 0 else -1.0
            assert p["active_turret_id"] == ("fwd_a" if s > 0 else "aft_a")
            assert p["turret_angles_deg"]["elevation"] == pytest.approx(math.degrees(math.asin(s * d[2])), abs=2e-3)
            az = (s * math.degrees(math.atan2(-d[0], d[1])) + 180.0) % 360.0 - 180.0
            assert p["turret_angles_deg"]["azimuth"] == pytest.approx(az, abs=2e-3)
            sun = r.sun_eci[i] / np.linalg.norm(r.sun_eci[i])
            assert p["sun_vector_body_deg"] == pytest.approx(math.degrees(math.atan2(sun @ z, sun @ y)) % 360.0, abs=2e-2)


def test_the_sun_angle_makes_one_turn_per_orbit_and_is_zero_at_local_noon(world):
    r, src = world
    t = np.arange(0.0, 6000.0, 20.0)
    ang = np.unwrap(np.radians([src.packet_at(float(x), 0)["sun_vector_body_deg"] for x in t]))
    assert abs(ang[-1] - ang[0]) / (2 * math.pi) == pytest.approx(6000.0 / 5801.2, abs=0.03)          # the arrays turn once per 96.7 min orbit
    assert np.all(np.diff(ang) < 0) or np.all(np.diff(ang) > 0)                                       # monotonic: the Sun sweeps one way around the body


def test_values_move_smoothly_between_engine_steps(world):
    """At 30 Hz and x60 playback the engine steps (10 s) are 5 packets apart: the twin must not move in steps."""
    r, src = world
    t = 3000.0 + np.arange(0, 120) * (60.0 / 30.0)                    # 120 consecutive packets at x60 playback
    pk = [src.packet_at(float(x), 0) for x in t]
    sun = np.array([p["sun_vector_body_deg"] for p in pk])
    core = np.array([p["thermal_state"]["core_temp_c"] for p in pk])
    assert np.max(np.abs(np.diff(np.unwrap(np.radians(sun))))) < math.radians(1.5)                     # about 1.2 deg per packet, no 6 deg jumps
    assert np.count_nonzero(np.diff(sun)) > 110                                                        # the Sun angle changes every packet, not once per engine step
    assert np.all(np.diff(core) <= 0.0) or np.all(np.diff(core) >= 0.0)                                  # the slowly varying core temperature is monotonic here: no sawtooth


def test_the_thermal_state_is_in_the_expected_bands_by_default(world):
    r, src = world
    pk = [src.packet_at(float(t), 0)["thermal_state"] for t in np.linspace(0, r.t_s[-1], 600)]
    core = np.array([p["core_temp_c"] for p in pk])
    rad = np.array([p["radiator_temp_c"] for p in pk])
    flow = np.array([p["coolant_flow_lpm"] for p in pk])
    byp = np.array([p["bypass_valve_pct"] for p in pk])
    assert 30.0 <= core.min() and core.max() <= 62.0                                       # "near 30 to 60 degC normally"
    assert -10.0 < rad.min() < 5.0 and 25.0 < rad.max() < 45.0                              # radiator: cold in eclipse, warm in sunlight (engine's range -4 .. 34)
    assert np.all(flow == pytest.approx(flow[0])) and 30.0 < flow[0] < 40.0                  # 10 kW at a 5 K coolant rise: about 33.6 L/min
    assert byp.min() == 0.0 and byp.max() > 60.0                                            # the bypass valve opens in eclipse and closes in sunlight
    assert all(p["throttle_mode"] == "NOMINAL" for p in pk)
    sunlit = np.array([r.tracks[0].eclipse[r.index_at(float(t))] == 0 for t in np.linspace(0, r.t_s[-1], 600)])
    assert byp[~sunlit].mean() > byp[sunlit].mean() + 20.0


def test_a_hotter_payload_drives_the_core_past_85_degrees_and_throttles():
    cfg = sim.SimulationConfig()
    cfg.thermal.internal_dissipation_w = 15_000.0
    r = sim.run_simulation(cfg)
    src = tb.DigitalTwinSource(r)
    modes = np.array(src.thermal.throttle_mode)
    assert src.thermal.core_temp_c.max() >= 85.0 and np.any(modes == "THROTTLE")
    i = int(np.flatnonzero(modes == "THROTTLE")[0])
    assert src.packet_at(i * r.config.step_s, 0)["thermal_state"]["throttle_mode"] == "THROTTLE"
    assert src.thermal.load_fraction[i] == pytest.approx(0.6)                               # throttling cuts the compute heat to 60 %
    cfg.thermal.internal_dissipation_w = 25_000.0
    assert "SHUTDOWN" in tb.DigitalTwinSource(sim.run_simulation(cfg)).thermal.throttle_mode


def test_throttle_has_hysteresis():
    cfg = sim.SimulationConfig()
    cfg.thermal.internal_dissipation_w = 15_000.0
    h = tl.simulate_loop(sim.run_simulation(cfg).tracks[0].radiator_solar_load_w, 10.0, cfg.thermal)
    lc = tl.ThermalLoopConfig()
    for k in range(1, len(h.throttle_mode)):
        if h.throttle_mode[k - 1] == "NOMINAL" and h.throttle_mode[k] == "THROTTLE":
            assert h.core_temp_c[k - 1] >= lc.throttle_c                                    # only enters throttle at 85 degC or more
        if h.throttle_mode[k - 1] == "THROTTLE" and h.throttle_mode[k] == "NOMINAL":
            assert h.core_temp_c[k - 1] <= lc.throttle_release_c                            # and only leaves once it has cooled to 80


def test_thermal_loop_formulas_by_hand():
    cfg = tl.ThermalLoopConfig()
    flow = tl.coolant_flow_lpm(10_000.0, cfg)
    assert flow == pytest.approx(10_000.0 / (5.0 * 3400.0) / 1.05 * 60.0)                   # 33.6 L/min
    assert tl.coolant_flow_lpm(100.0, cfg) == cfg.flow_min_lpm and tl.coolant_flow_lpm(1e6, cfg) == cfg.flow_max_lpm
    # bypass: outlet 3 K above the panel; supply = outlet + b/(1-b) * rise must equal the 18 degC minimum
    b = tl.bypass_fraction(-5.0, 4.0, cfg)
    assert (-5.0 + 3.0) + b / (1 - b) * 4.0 == pytest.approx(18.0)
    assert tl.bypass_fraction(30.0, 4.0, cfg) == 0.0                                       # a warm radiator needs no bypass
    assert tl.bypass_fraction(-200.0, 4.0, cfg) == pytest.approx(0.95)                      # capped


def test_core_temperature_settles_to_the_steady_state_equation():
    from orbital_engine import ThermalConfig
    th = ThermalConfig()
    cfg = tl.ThermalLoopConfig()
    h = tl.simulate_loop(np.zeros(3000), 60.0, th, cfg)                                    # permanent eclipse, 50 hours
    t_rad = float(tl.radiator_equilibrium_c(10_000.0, 0.0, th.radiator_area_m2, th.emissivity, th.space_temperature_k))
    flow = tl.coolant_flow_lpm(10_000.0, cfg)
    rise = 10_000.0 / (flow / 60.0 * cfg.coolant_kg_per_l * cfg.coolant_cp_j_per_kg_k)
    b = tl.bypass_fraction(t_rad, rise, cfg)
    supply = t_rad + cfg.radiator_approach_k + b / (1 - b) * rise
    expected = supply + rise / 2.0 + 10_000.0 * cfg.plate_resistance_k_per_w
    assert h.core_temp_c[-1] == pytest.approx(expected, abs=0.1)
    assert h.radiator_temp_c[-1] == pytest.approx(t_rad, abs=0.1)


def test_dsp_metrics_come_from_the_end_to_end_link(world):
    r, src = world
    link = r.primary_link
    names = ["BPSK", "QPSK", "8PSK", "16QAM"]
    for kind in (1, 2, 3):
        i = _pick(r, kind)
        dm = src.packet_at(i * r.config.step_s, 0)["dsp_metrics"]
        best = int(link.adaptive_modulation[i])
        assert dm["modulation"] == names[best] and dm["throughput_gbps"] == pytest.approx(float(link.adaptive_throughput_gbps[i]), abs=0.01)
        assert dm["snr_db"] == pytest.approx(float(link.snr_db[i]), abs=0.01) and 0.0 <= dm["ber"] <= 0.0038         # best modulation only runs below the FEC threshold


def test_fsm_deflection_is_half_the_aoa_disturbance_with_jitter():
    f = tb.FsmModel(seed=3)
    xs, ys = zip(*(f.sample(True) for _ in range(20000)))
    assert np.mean(xs) == pytest.approx(125.0, abs=0.3) and np.mean(ys) == pytest.approx(-90.0, abs=0.3)    # half of 250 / -180 urad
    assert np.std(xs) == pytest.approx(7.5, rel=0.05)                                                         # half of the 15 urad jitter
    assert max(abs(v) for v in xs + ys) <= tb.FSM_RANGE_MECH_URAD and f.sample(False) == (0.0, 0.0)


def test_timestamp_is_the_simulation_epoch(world):
    r, src = world
    assert src.packet_at(0.0, 0)["timestamp_utc"] == "2026-12-21T00:00:00.000Z"
    assert src.packet_at(3723.5, 0)["timestamp_utc"] == "2026-12-21T01:02:03.500Z"


# ---- the window and the command line ----------------------------------------------------------------------------------------------------------
def test_the_window_streams_at_30hz_while_playing_and_1hz_when_paused():
    import main_app
    QApplication, MainWindow, theme = main_app.build_window_classes()
    app = QApplication.instance() or QApplication([])
    rx, port = udp_receiver()
    win = MainWindow(direct_cfg(duration_h=12, step_s=30), broadcast=True, broadcast_port=port, status_port=0)
    win.conditions.live.setChecked(False)
    win.start_simulation(direct_cfg(duration_h=12, step_s=30), synchronous=True)           # the window stays hidden: no painting in this test loop (see below)
    assert win.broadcaster is not None and win.twin_source is not None and win.broadcast_action.isChecked()
    win.clock.set_speed(600.0)
    win.clock.play()
    t0 = time.time()
    while time.time() - t0 < 1.2:                                                        # warm-up: the first second after the window starts is busy building plots
        app.processEvents()
        time.sleep(0.03)
    drain(rx)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        app.processEvents()
        time.sleep(0.03)                                                                 # sleeping releases the interpreter lock for the feed thread (a tight processEvents loop would hold it)
    playing = drain(rx)
    assert 33 <= len(playing) <= 56                                                      # about 30 Hz for 1.5 s (measured 30.0 Hz on the real window, which does paint)
    assert [p["seq"] for p in playing] == list(range(playing[0]["seq"], playing[0]["seq"] + len(playing)))
    validate_schema(playing[-1])
    assert playing[-1]["timestamp_utc"] > playing[0]["timestamp_utc"]                     # the simulation epoch advanced while playing
    win.clock.pause()
    drain(rx)
    t0 = time.time()
    while time.time() - t0 < 2.5:
        app.processEvents()
        time.sleep(0.03)
    paused = drain(rx)
    assert 1 <= len(paused) <= 5                                                         # about 1 Hz
    assert len({p["timestamp_utc"] for p in paused}) == 1                                 # a paused simulation repeats its state
    # satCAD's status shows in the window
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.sendto(json.dumps({"v": 1, "app": "satCAD", "twin": "online", "frames": 9}).encode(), ("127.0.0.1", win.broadcaster.status_port))
    t0 = time.time()
    while not win.broadcaster.status.connected and time.time() - t0 < 3:
        time.sleep(0.01)
    win._update_twin_label()
    assert "satCAD connected" in win.twin_label.text()
    # switching the stream off stops the packets and the label says so
    win.broadcast_action.setChecked(False)
    assert win.broadcaster is None and win.twin_label.text() == "Digital twin: off"
    win.clock.play()
    drain(rx)
    t0 = time.time()
    while time.time() - t0 < 0.5:
        app.processEvents()
        time.sleep(0.03)
    assert drain(rx) == []
    win.clock.pause()
    win.close()
    client.close()
    rx.close()


def test_the_window_does_not_broadcast_unless_asked():
    import main_app
    QApplication, MainWindow, theme = main_app.build_window_classes()
    app = QApplication.instance() or QApplication([])
    win = MainWindow(direct_cfg(duration_h=2, step_s=30))
    win.start_simulation(direct_cfg(duration_h=2, step_s=30), synchronous=True)
    assert win.broadcaster is None and not win.broadcast_action.isChecked()
    win.close()


def test_headless_broadcast_mode_streams_in_real_time(capsys):
    import main_app
    rx, port = udp_receiver()
    code = main_app.main(["--headless", "--broadcast", "--broadcast-port", str(port), "--status-port", "0", "--speed", "600", "--wall-seconds", "1.2"])
    assert code == 0
    got = drain(rx)
    out = capsys.readouterr().out
    assert 28 <= len(got) <= 44 and "Streaming digital-twin telemetry to 127.0.0.1" in out
    validate_schema(got[-1])
    assert got[-1]["timestamp_utc"] > got[0]["timestamp_utc"]
    rx.close()


def test_plain_headless_does_not_stream(capsys):
    import main_app
    rx, port = udp_receiver()
    assert main_app.main(["--headless", "--hours", "1", "--broadcast-port", str(port)]) == 0
    assert drain(rx) == [] and "GEO relay network" in capsys.readouterr().out
    rx.close()


def test_command_line_flags_have_the_documented_defaults():
    import main_app
    a = main_app.parse_args([])
    assert a.broadcast is None and a.broadcast_port == 9876 and a.status_port == 9877 and a.speed is None and not a.autoplay and not a.no_placement
    a = main_app.parse_args(["--no-broadcast", "--broadcast-port", "9999", "--speed", "60", "--autoplay", "--no-placement"])
    assert a.broadcast is False and a.broadcast_port == 9999 and a.speed == 60.0 and a.autoplay and a.no_placement
    assert main_app.parse_args(["--broadcast"]).broadcast is True


def test_the_launcher_command_is_accepted_end_to_end():
    """./run.sh --broadcast --autoplay --no-placement --speed 60 is what satCAD's launcher calls: it must parse and not ask for the placement window."""
    import main_app
    a = main_app.parse_args(["--broadcast", "--autoplay", "--no-placement", "--speed", "60"])
    assert a.broadcast and a.autoplay and a.no_placement and a.speed == 60.0 and a.lat is None
