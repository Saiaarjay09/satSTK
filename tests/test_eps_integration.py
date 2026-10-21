"""The EPS tab and its connections: the thermal-loop coupler hook, the digital-twin `eps_state`, the real window (offscreen) and the stress-test controls."""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import eps_power_engine as ep
import simulation as sim
import telemetry_broadcaster as tb
import thermal_loop as tl
from helpers import direct_cfg


@pytest.fixture(scope="module")
def world():
    return sim.run_simulation(direct_cfg(duration_h=6, step_s=10))


@pytest.fixture(scope="module")
def run(world):
    return ep.simulate_eps(world)


# ---- the thermal-loop coupler -----------------------------------------------------------------------------------------------------------
def test_no_coupler_keeps_the_old_behaviour_and_an_equivalent_coupler_reproduces_it(world):
    tr = world.tracks[0]
    base = tl.simulate_loop(tr.radiator_solar_load_w, 10.0, world.config.thermal)
    p_nominal = world.config.thermal.internal_dissipation_w
    same = tl.simulate_loop(tr.radiator_solar_load_w, 10.0, world.config.thermal, coupler=lambda i, t, m, f, r: (p_nominal * f, 1.0))
    assert np.allclose(base.core_temp_c, same.core_temp_c) and np.allclose(base.coolant_flow_lpm, same.coolant_flow_lpm) and base.throttle_mode == same.throttle_mode


def test_coupler_heat_and_flow_scale_reach_the_loop(world):
    tr = world.tracks[0]
    cool = tl.simulate_loop(tr.radiator_solar_load_w, 10.0, world.config.thermal, coupler=lambda i, t, m, f, r: (5_000.0, 1.0))
    hot = tl.simulate_loop(tr.radiator_solar_load_w, 10.0, world.config.thermal, coupler=lambda i, t, m, f, r: (9_000.0, 1.0))
    slow = tl.simulate_loop(tr.radiator_solar_load_w, 10.0, world.config.thermal, coupler=lambda i, t, m, f, r: (5_000.0, 0.4))
    assert hot.core_temp_c[-1] > cool.core_temp_c[-1] + 3.0                                          # more heat, hotter core
    assert slow.coolant_flow_lpm[-1] == pytest.approx(0.4 * cool.coolant_flow_lpm[-1]) and slow.core_temp_c[-1] > cool.core_temp_c[-1]
    seen = []
    tl.simulate_loop(tr.radiator_solar_load_w[:50], 10.0, world.config.thermal, coupler=lambda i, t, m, f, r: (seen.append((i, m, f)) or 4_000.0, 1.0))
    assert [s[0] for s in seen] == list(range(50)) and all(s[1] == tl.NOMINAL and s[2] == 1.0 for s in seen)


# ---- the digital-twin packet ----------------------------------------------------------------------------------------------------------
def test_packet_carries_eps_state_only_when_given_and_clamps_it():
    base = dict(seq=1, timestamp_utc="2026-12-21T00:00:00.000Z", sun_vector_body_deg=10.0, active_relay_id="NONE", active_turret_id="fwd_a", turret_angles_deg={}, thermal_state={}, dsp_metrics={})
    assert "eps_state" not in tb.build_packet(**base)
    p = tb.build_packet(**base, eps_state=dict(soc_pct=140.0, busbar_current_a=-5.0, busbar_temp_c=31.234, bank_b_breaker_closed=0, mode="X" * 50))
    e = p["eps_state"]
    assert e["soc_pct"] == 100.0 and e["busbar_current_a"] == 0.0 and e["busbar_temp_c"] == 31.2 and e["bank_b_breaker_closed"] is False and len(e["mode"]) == 20
    assert set(e) >= {"soc_pct", "busbar_current_a", "busbar_temp_c", "bank_b_breaker_closed"}
    assert len(tb.encode_packet(p)) < 1000


def test_twin_source_streams_the_electrical_state_and_the_coupled_thermal_state(world, run):
    src = tb.DigitalTwinSource(world, eps_run=run)
    plain = tb.DigitalTwinSource(world)
    assert "eps_state" not in plain.packet_at(100.0, 0)
    t = 1500.0                                                                                     # inside the first eclipse
    pk = src.packet_at(t, 3)
    e = pk["eps_state"]
    i = run.history.index_at(t)
    assert e["soc_pct"] == pytest.approx(run.history.soc[i] * 100.0, abs=0.01) and e["bank_b_breaker_closed"] is True
    assert e["busbar_current_a"] == pytest.approx(run.history.bus_current_a[i], abs=0.1) and e["busbar_temp_c"] == pytest.approx(run.history.t_bus_c[i], abs=0.1)
    assert e["mode"] == ep.ECLIPSE_CRUISE and e["solar_kw"] == 0.0 and 25.0 < e["load_kw"] < 35.0 and 47.8 < e["min_blade_v"] < 48.0
    assert src.thermal is run.thermal and pk["thermal_state"]["core_temp_c"] == pytest.approx(run.thermal.core_temp_c[i], abs=0.5)
    assert len(tb.encode_packet(pk)) < 1000
    json.loads(tb.encode_packet(pk))


def test_breaker_state_streams_when_bank_b_opens(world):
    run = ep.simulate_eps(world, ep.EPSConfig(soh=0.10, force_eclipse_min=45.0))
    k = run.history.flush_steps[0] + 20
    pk = tb.DigitalTwinSource(world, eps_run=run).packet_at(float(run.history.t_s[k]), 0)
    assert pk["eps_state"]["bank_b_breaker_closed"] is False and pk["eps_state"]["mode"] == ep.ECLIPSE_SURVIVAL


@pytest.mark.skipif(not (Path(os.environ.get("SATCAD_DIR", Path.home() / "Developer" / "Open-source-CAD-software")) / ".venv" / "bin" / "python").exists(),
                    reason="satCAD checkout with its own virtualenv not found (set SATCAD_DIR)")
def test_packets_with_eps_state_are_still_accepted_by_the_real_satcad_parser(world, run, tmp_path):
    satcad = Path(os.environ.get("SATCAD_DIR", Path.home() / "Developer" / "Open-source-CAD-software"))
    f = tmp_path / "p.json"
    f.write_bytes(tb.encode_packet(tb.DigitalTwinSource(world, eps_run=run).packet_at(1500.0, 5)))
    code = "import sys; sys.path.insert(0, '.'); from satcad.digital_twin import parse_packet\np = parse_packet(open(sys.argv[1], 'rb').read()); print('PARSED', p.seq)"
    out = subprocess.run([str(satcad / ".venv" / "bin" / "python"), "-c", code, str(f)], cwd=satcad, capture_output=True, text=True, timeout=120)
    if "No module" in out.stderr:
        pytest.skip("satCAD receiver module not importable here")
    assert "PARSED 5" in out.stdout, out.stderr


# ---- the window ----------------------------------------------------------------------------------------------------------------------------
pytest.importorskip("PySide6")
import main_app  # noqa: E402


@pytest.fixture(scope="module")
def window():
    QApplication, MainWindow, theme = main_app.build_window_classes()
    app = QApplication.instance() or QApplication([])
    theme.apply_dark_theme(app)
    win = MainWindow(direct_cfg(duration_h=12, step_s=30))
    win.resize(1800, 1100)
    win.conditions.live.setChecked(False)
    win.start_simulation(win.cfg, synchronous=True)
    win.show()
    app.processEvents()
    yield win, app
    win._debounce.stop()
    win._cleanup_thread()
    win.payload._stop_thread()
    win.close()
    app.processEvents()


@pytest.fixture()
def panel(window):
    win, app = window
    win.eps.reset_stress()
    return win.eps


def _grab(widget):
    img = widget.grab().toImage()
    return np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4].copy()


def test_tab_is_beside_the_others_and_the_twin_and_scheduler_use_its_run(window, panel):
    win, app = window
    assert win.eps_dock in win.tabifiedDockWidgets(win.tele_dock) and "EPS POWER" in win.eps_dock.windowTitle()
    assert panel.run is not None and win.twin_source is not None and win.twin_source.thermal is panel.run.thermal and win.twin_source.eps is panel.run.history
    assert win.radiation._eps is panel.run.history and win.radiation.orch.eps is panel.run.history
    assert "eps_state" in win.twin_source.packet_at(0.0, 0)


def test_the_tab_follows_the_clock_and_shows_every_view(window, panel):
    win, app = window
    h = panel.run.history
    k = int(np.flatnonzero(h.illumination < 0.5)[200])                                              # deep in an eclipse
    win.clock.seek(float(h.t_s[k]))
    app.processEvents()
    assert panel._idx == k and h.mode[k] == ep.ECLIPSE_CRUISE
    assert "ECLIPSE CRUISE" in panel.status.text() and panel.flow.s["mode"] == ep.ECLIPSE_CRUISE and panel.flow.s["solar_kw"] == 0.0
    assert panel.flow.s["bat_kw"] > 20.0 and panel.flow.s["dvfs"] and panel.flow.s["bank_b_closed"]
    assert len(panel.soc_curve.xData) > 100 and np.nanmax(panel.soc_curve.yData) == pytest.approx(100.0, abs=0.5)
    v = panel.run.history.blades(k)["v_blade"]
    assert np.allclose(panel.bars_a.opts["height"] + panel.bars_a.opts["y0"], v[:16]) and np.allclose(panel.bars_b.opts["height"] + panel.bars_b.opts["y0"], v[16:])
    assert "Blade 1" in panel.v_note.text() and "inside the 45.5-50.0 V" in panel.v_note.text()
    for w in (panel.flow, panel.soc_plot, panel.power_plot, panel.v_plot):
        assert len(np.unique(_grab(w).reshape(-1, 4), axis=0)) > 8, f"{type(w).__name__} looks blank"   # (a hidden dock tab paints sparsely on some offscreen backends)


def test_sunlit_view_shows_solar_charging_and_a_green_battery_arrow(window, panel):
    win, app = window
    h = panel.run.history
    k = int(np.flatnonzero(h.illumination > 0.99)[2000 if (h.illumination > 0.99).sum() > 2000 else 5])
    win.clock.seek(float(h.t_s[k]))
    app.processEvents()
    assert panel.flow.s["solar_kw"] > 60.0 and panel.flow.s["mode"] in (ep.SUNLIT_PEAK, ep.SUNLIT_RECHARGE) and not panel.flow.s["dvfs"]


def test_wing_failure_button_reduces_the_array_and_recomputes_everything(window, panel):
    win, app = window
    before = panel.run.history.solar_w.max()
    seen = []
    panel.run_changed.connect(seen.append)
    panel.wing_btn.setChecked(True)
    assert len(seen) == 1 and panel.run.history.cfg.panels_failed == 1 and panel.run.history.solar_w.max() == pytest.approx(before * 11 / 12, rel=0.01)
    assert win.twin_source.eps is panel.run.history and win.radiation.orch.eps is panel.run.history                  # the twin and the scheduler were re-pointed
    assert "1 panel failed" in panel.status.text()
    panel.run_changed.disconnect(seen.append)


def test_extended_eclipse_button_and_degradation_slider(window, panel):
    win, app = window
    panel.eclipse_btn.setChecked(True)
    ecl = panel.run.history.summary()["eclipses"]
    assert all(e["duration_min"] >= 44.9 for e in ecl[:-1]) and "eclipse ≥ 45 min" in panel.status.text()                  # (the last one may be cut off by the scenario's end)
    panel.deg_slider.setValue(90)
    assert "capacity lost 90 %" in panel.deg_label.text() and "32 kWh left" in panel.deg_label.text()
    panel.rerun()                                                                                                 # (the slider re-runs after a short debounce; do it now)
    h = panel.run.history
    assert h.cfg.soh == pytest.approx(0.10) and h.summary()["flushes"] > 0
    k = h.flush_steps[0] + 2
    win.clock.seek(float(h.t_s[k]))
    app.processEvents()
    assert panel.flow.s["mode"] == ep.ECLIPSE_SURVIVAL and not panel.flow.s["bank_b_closed"] and panel.flow.s["oisl_low"] and panel.flow.s["rpm"] == pytest.approx(0.4)
    assert len(np.unique(_grab(panel.flow).reshape(-1, 4), axis=0)) > 8
    assert panel.soc_curve.yData.min() < 35.0 and panel.soc_curve.yData.min() > 20.0
    panel.reset_stress()
    assert panel.run.history.cfg.soh == 1.0 and panel.run.history.cfg.panels_failed == 0 and panel.run.history.cfg.force_eclipse_min == 0.0
    assert panel.deg_slider.value() == 0 and not panel.wing_btn.isChecked() and not panel.eclipse_btn.isChecked()


def test_slider_is_debounced(window, panel):
    win, app = window
    n = []
    panel.run_changed.connect(n.append)
    for v in (10, 20, 30):
        panel.deg_slider.setValue(v)
    assert n == [] and panel._debounce.isActive()                                                                 # nothing yet: one run after the last move
    t0 = time.time()
    while not n and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.02)
    assert len(n) == 1 and n[0].history.cfg.soh == pytest.approx(0.70)
    panel.run_changed.disconnect(n.append)


def test_scenario_change_recomputes_the_eps_for_the_new_orbit(window, panel):
    win, app = window
    cfg = direct_cfg(duration_h=8, step_s=30)
    win.conditions.load(cfg)
    win.start_simulation(cfg, synchronous=True)
    assert panel.run.history.n == win.result.n and win.twin_source.eps is panel.run.history
    cfg = direct_cfg(duration_h=12, step_s=30)
    win.conditions.load(cfg)
    win.start_simulation(cfg, synchronous=True)


def test_screenshot_with_the_eps_tab_and_stress_flags(tmp_path):
    out = tmp_path / "e.png"
    assert main_app.main(["--screenshot", str(out), "--show-eps", "--eps-eclipse", "--eps-degradation", "90", "--eps-wing-failure", "--at-hours", "0.8", "--hours", "6", "--size", "1800x1300"]) == 0
    assert out.stat().st_size > 50_000


def test_headless_broadcast_includes_the_eps_state(tmp_path, capsys):
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(3.0)
    port = rx.getsockname()[1]
    assert main_app.main(["--headless", "--broadcast", "--broadcast-port", str(port), "--status-port", "0", "--hours", "3", "--wall-seconds", "1.5", "--speed", "300"]) == 0
    got = []
    rx.settimeout(0.2)
    try:
        while True:
            got.append(json.loads(rx.recvfrom(4096)[0]))
    except (socket.timeout, OSError):
        pass
    rx.close()
    assert got and all("eps_state" in p and 0.0 <= p["eps_state"]["soc_pct"] <= 100.0 for p in got)
