"""Build the real window offscreen, drive the clock and the panels, and check that every view actually paints something."""
import pytest

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402

import events as ev  # noqa: E402
import main_app  # noqa: E402
import orbital_engine as oe  # noqa: E402
import simulation as sim  # noqa: E402
from helpers import direct_cfg  # noqa: E402
from ui import geo, map_view  # noqa: E402


@pytest.fixture(scope="module")
def window():
    QApplication, MainWindow, theme = main_app.build_window_classes()
    app = QApplication.instance() or QApplication([])
    theme.apply_dark_theme(app)
    win = MainWindow(direct_cfg(duration_h=24, step_s=30))
    win.resize(1700, 950)
    win.conditions.live.setChecked(False)                                                # no background recomputes while the tests drive the panels
    win.start_simulation(win.cfg, synchronous=True)
    win.show()
    app.processEvents()
    yield win, app
    win._debounce.stop()                                                                 # never leave a worker thread running when the interpreter exits
    win._cleanup_thread()
    win.payload._stop_thread()
    win.close()
    app.processEvents()


def _colors(widget):
    img = widget.grab().toImage()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4]
    return len(np.unique(arr.reshape(-1, 4), axis=0))


def _locked_index(win):
    return int(np.argmax(win.result.primary_link.locked))


def test_all_panes_paint(window):
    win, app = window
    win.clock.seek(float(win.result.t_s[_locked_index(win)]))
    app.processEvents()
    for w in (win.globe, win.map, win.timeline, win.telemetry, win.conditions):
        assert _colors(w) > 40, f"{type(w).__name__} looks blank"


def test_clock_step_play_and_views_stay_in_sync(window):
    win, app = window
    win.clock.seek(0.0)
    win.clock.step(1)
    assert win.clock.t == pytest.approx(30.0) and win.globe.idx == win.map.idx == 1
    win.clock.step(-1)
    assert win.clock.t == 0.0
    win.clock.set_speed(600.0)
    win.clock.play()
    assert win.clock.playing
    win.clock._last -= 1.0                     # pretend a second passed; one frame is capped at 0.25 s of wall time
    win.clock._tick()
    assert win.clock.t == pytest.approx(150.0, rel=0.05) and win.timeline.t == win.clock.t
    win.clock.pause()
    assert not win.clock.playing


def test_next_pass_button_jumps_to_just_before_the_next_pass(window):
    win, app = window
    win.clock.seek(0.0)
    starts = [s for s, e in win.result.primary_windows]
    win.next_pass()
    assert win.clock.t == pytest.approx(max(0.0, starts[0] - 60.0))
    win.next_pass()
    assert win.clock.t == pytest.approx(starts[1] - 60.0 if len(starts) > 1 else starts[0] - 60.0)


def test_telemetry_follows_beam_state(window):
    win, app = window
    link = win.result.primary_link
    win.telemetry.set_index(_locked_index(win))
    assert win.telemetry._link_labels["status"].text() in ("LOCKED", "DEGRADED")
    win.telemetry.set_index(int(np.argmax(~link.visible)))
    assert win.telemetry._link_labels["status"].text() == "BLOCKED"
    assert win.telemetry._link_labels["rx"].text() == "—"


def test_modulation_switch_updates_readout(window):
    win, app = window
    win.telemetry.set_index(_locked_index(win))
    win.telemetry._mod.setCurrentText("BPSK")
    ber_bpsk = win.telemetry._link_labels["ber"].text()
    win.telemetry._mod.setCurrentText("16QAM")
    assert win.telemetry._link_labels["ber"].text() != ber_bpsk
    win.telemetry._mod.setCurrentText("QPSK")


def test_active_events_are_listed_in_telemetry(window):
    win, app = window
    cfg = win.conditions.config(win.cfg)
    mid = float(np.mean(win.result.primary_windows[0]))
    cfg.events = ev.EventsConfig(manual=[ev.ManualEvent("flock", mid - 40.0, 80.0, 0.5)])
    win.start_simulation(cfg, synchronous=True)
    win.clock.seek(mid)
    assert "Flock of birds" in win.telemetry._events_label.text()
    assert win.telemetry._link_labels["status"].text() == "BLOCKED"
    win.start_simulation(win.conditions.config(win.cfg).__class__(), synchronous=True)     # back to the default scenario


def test_conditions_panel_round_trips_and_edits_flow_into_the_config(window):
    win, app = window
    panel = win.conditions
    base = direct_cfg(satellites=sim.two_satellites(), target_index=1)
    panel.load(base)
    cfg = panel.config(base)
    assert len(cfg.satellites) == 2 and cfg.target_index == 1 and cfg.ground.lat_deg == pytest.approx(base.ground.lat_deg)
    panel._w["lat"].setValue(-33.9)
    panel._w["lon"].setValue(18.4)
    panel._w["rain"].setValue(12.0)
    panel._w["div"].setValue(150.0)
    panel.step.setCurrentText("1 s")
    panel.ev_on.setChecked(True)
    panel.ev_rate.setValue(200)
    cfg = panel.config(base)
    assert cfg.ground.lat_deg == -33.9 and cfg.ground.lon_deg == 18.4 and cfg.atmosphere.rain_mm_h == 12.0
    assert cfg.link.divergence_urad == 150.0 and cfg.step_s == 1.0 and cfg.events.enabled and cfg.events.rate_scale == pytest.approx(10.0)
    assert cfg.atmosphere.label == "Custom"
    panel.weather.setCurrentText("Overcast")
    assert panel._w["cloud"].value() == 25.0 and panel.config(base).atmosphere.label == "Overcast"
    panel.second.setChecked(False)
    assert len(panel.config(base).satellites) == 1
    panel.load(direct_cfg())


def test_changing_a_condition_changes_the_result(window):
    win, app = window
    panel = win.conditions
    panel.load(direct_cfg())
    base = sim.run_simulation(panel.config(win.cfg))
    panel.weather.setCurrentText("Overcast")
    cloudy = sim.run_simulation(panel.config(win.cfg))
    panel.load(direct_cfg())
    assert cloudy.primary_link.adaptive_throughput_gbps.max() < base.primary_link.adaptive_throughput_gbps.max()


def test_placement_dialog_and_map_picker(window):
    win, app = window
    from ui.placement_dialog import PlacementDialog
    dlg = PlacementDialog(oe.GroundStation("x", 10.0, 20.0, 100.0, 25.0), 600.0)
    dlg.resize(900, 600)
    dlg.show()
    app.processEvents()
    assert _colors(dlg.map) > 30
    dlg.map.picked.emit(-12.5, 130.25)
    assert dlg.station().lat_deg == -12.5 and dlg.station().lon_deg == 130.25 and dlg.station().min_elevation_deg == 25.0
    dlg.preset.setCurrentText("Hanle, Ladakh, India (high desert)")
    assert dlg.station().lat_deg == pytest.approx(32.78) and dlg.station().alt_m == 4500.0
    dlg.close()


def test_events_dialog_adds_and_removes_manual_events(window):
    win, app = window
    from ui.events_dialog import EventsDialog
    dlg = EventsDialog(ev.EventsConfig(), 24.0)
    dlg.kind.setCurrentIndex(dlg.kind.findData("ash"))
    dlg.start.setValue(2.5)
    dlg._add()
    assert [m.key for m in dlg.config().manual] == ["ash"] and dlg.config().manual[0].start_s == 9000.0
    dlg._rows[0][1].setChecked(False)
    assert dlg.config().disabled_types == [dlg._rows[0][0]]
    dlg.manual_list.setCurrentRow(0)
    dlg._remove()
    assert dlg.config().manual == []


def test_ctrl_click_on_the_map_requests_a_station_move(window):
    win, app = window
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    got = []
    win.map.station_move_requested.connect(lambda la, lo: got.append((la, lo)))
    box = win.map._box()
    pos = QPointF(box.center().x(), box.center().y())
    win.map._cursor = (0.0, 0.0)
    ev_ = QMouseEvent(QMouseEvent.Type.MouseButtonPress, pos, win.map.mapToGlobal(pos), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    win.map.mousePressEvent(ev_)
    assert got == [(0.0, 0.0)]
    assert win.cfg.ground.lat_deg == 0.0 and win.cfg.ground.lon_deg == 0.0               # the window moved the station there (and recomputed)
    restored = direct_cfg(duration_h=24, step_s=30)                                       # put the default station back for the other tests
    win.conditions.load(restored)
    win.start_simulation(restored, synchronous=True)


def test_land_data_and_mask_present():
    assert len(geo.land_rings()) > 100
    mask = geo.land_mask()
    assert 0.2 < mask.mean() < 0.4                                    # land is about 29 % of the planet's surface
    assert mask[int((90 - 40) / 180 * 720), int((-100 + 180) / 360 * 1440)] > 0.5      # central USA is land
    assert mask[int((90 - 0) / 180 * 720), int((-150 + 180) / 360 * 1440)] < 0.5       # equatorial Pacific is sea


def test_footprint_geometry():
    import math
    lon, lat = map_view.footprint_polygon(0.0, 0.0, math.radians(24.4))
    assert lat.max() == pytest.approx(24.4, abs=0.1) and lon.max() == pytest.approx(24.4, abs=0.1)
    lon, lat = map_view.footprint_polygon(85.0, 10.0, math.radians(24.4))       # encloses the north pole
    assert lat[-1] == 90.0 and (lon.max() - lon.min()) > 300


def test_screenshot_cli(tmp_path):
    out = tmp_path / "s.png"
    assert main_app.main(["--screenshot", str(out), "--hours", "6", "--size", "1500x860", "--events", "--weather", "Hazy"]) == 0
    assert out.stat().st_size > 20_000


def test_headless_cli(tmp_path, capsys):
    csv_path = tmp_path / "o.csv"
    assert main_app.main(["--headless", "--lat", "17.39", "--lon", "78.49", "--csv", str(csv_path), "--events", "--event-rate", "5"]) == 0
    out = capsys.readouterr().out
    assert "Hyderabad" not in out and "17.390" in out and "passes above the mask" in out and csv_path.exists()


def test_selftest_cli(tmp_path):
    f = tmp_path / "st.txt"
    assert main_app.main(["--selftest", "--selftest-file", str(f)]) == 0
    assert "SELFTEST OK" in f.read_text(encoding="utf-8")


# ---- GEO relay controls ------------------------------------------------------------------------------------------------
def test_geo_toggle_recomputes_and_the_views_follow(window):
    win, app = window
    assert win.result.network_link is None and not win.geo_btn.isChecked()               # the fixture scenario has the relay off
    win.geo_btn.setChecked(True)                                                          # the toolbar switch drives the Conditions checkbox
    assert win.conditions.geo_on.isChecked() and win.geo_btn.text() == "GEO relay: ON"
    cfg = win.conditions.config(win.cfg)
    assert cfg.relay.enabled
    win.start_simulation(cfg, synchronous=True)
    win.clock.seek(float(win.result.t_s[int(np.argmax(win.result.route.kind == 3))]))
    app.processEvents()
    assert win.result.network_link is not None and win.result.route.uptime_pct == pytest.approx(100.0)
    assert "GEO-ATL" in win.telemetry._route_label.text() or "GEO-PAC" in win.telemetry._route_label.text()
    assert win.telemetry._route["relay"].text() == "ON" and "100.000" in win.telemetry._route["uptime"].text()
    assert win.globe._scale_combo.isVisibleTo(win.globe) is True
    for mode in (1, 0):                                                                   # true scale, then compressed
        win.globe._scale_combo.setCurrentIndex(mode)
        app.processEvents()
        assert _colors(win.globe) > 40
    for w in (win.map, win.timeline, win.telemetry):
        assert _colors(w) > 40
    win.geo_btn.setChecked(False)
    win.start_simulation(win.conditions.config(win.cfg), synchronous=True)
    assert win.result.network_link is None and win.telemetry._route["relay"].text().startswith("OFF")
    assert not win.globe._scale_combo.isVisibleTo(win.globe)


def test_globe_squash_keeps_leo_and_compresses_geo(window):
    win, app = window
    from ui.globe_view import GlobeView
    g = GlobeView()
    leo, geo = np.array([[6978.0, 0.0, 0.0]]), np.array([[oe.GEO_RADIUS_KM, 0.0, 0.0]])
    assert g._squash(leo) == pytest.approx(leo)                                           # LEO is never distorted
    assert 7500 < g._squash(geo)[0, 0] < 15000 and g._squash(geo)[0, 1] == 0.0            # GEO is pulled in, same direction
    g.scale_mode = "true"
    assert g._squash(geo) == pytest.approx(geo)


def test_default_cli_summary_shows_the_relay(capsys):
    assert main_app.main(["--headless"]) == 0
    out = capsys.readouterr().out
    assert "GEO relay network ON" in out and "GS-INDIA" in out
    assert main_app.main(["--headless", "--no-geo-relay"]) == 0
    assert "GEO relay network OFF" in capsys.readouterr().out


# ---- Optical payload transmitter dock ------------------------------------------------------------------------------------------
def _wait_for_worker(panel, app, seconds=60):
    import time
    t0 = time.time()
    while panel._thread is not None and time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert panel._thread is None, "payload worker never finished"


def test_payload_dock_exists_tabbed_with_telemetry_and_follows_the_clock(window):
    win, app = window
    default = sim.SimulationConfig()                                                     # default scenario: relays on
    win.conditions.load(default)
    win.start_simulation(default, synchronous=True)
    win.show()
    assert win.payload_dock in win.tabifiedDockWidgets(win.tele_dock)
    win.clock.seek(float(win.result.t_s[int(np.argmax(win.result.route.kind == 3))]))
    app.processEvents()
    assert "GEO-ATL" in win.payload.route_note.text() or "GEO-PAC" in win.payload.route_note.text()
    assert win.payload.route_combo.count() == 1 + len(win.result.route.labels)
    assert any(win.payload.route_combo.itemText(k).startswith("✗") for k in range(1, win.payload.route_combo.count()))   # unavailable routes are marked
    assert any(win.payload.route_combo.itemText(k).startswith("✓") for k in range(1, win.payload.route_combo.count()))


def test_generate_63kb_button_loads_the_test_payload(window):
    win, app = window
    p = win.payload
    p.gen_btn.click()
    assert p.rb_file.isChecked() and len(p._file_bytes) == 64512 and "63" in p.file_label.text() or "64,512" in p.file_label.text()
    assert pl_verify(p._file_bytes)
    data, name = p.current_payload()
    assert len(data) == 64512 and "generated" in name and p.save_btn.isEnabled()


def pl_verify(data):
    from dsp import payload
    return payload.verify_blocks(data).n_bad == 0


def test_fire_text_payload_through_the_worker_thread_and_see_every_view(window):
    win, app = window
    p = win.payload
    win.clock.seek(float(win.result.t_s[int(np.argmax(win.result.route.kind == 2))]))
    p.rb_text.setChecked(True)
    p.text.setPlainText("Fired from the GUI test: GS-INDIA to NODE-ALPHA.")
    p.mod.setCurrentText("QPSK")
    p.rate.setValue(10.0)
    p.seed.setValue(4)
    p.fire.click()
    assert not p.fire.isEnabled() and p.cancel_btn.isEnabled()                              # running in a background thread
    _wait_for_worker(p, app)
    assert p.fire.isEnabled() and "PASS" in p.badge.text()
    assert p.last.crc_ok and p.last.recovered == p.text.toPlainText().encode()
    assert p.m["route"].text() == "GS-INDIA -> GEO-IND -> NODE-ALPHA" and "QPSK" in p.m["mod"].text() and p.m["ber"].text().startswith("0")
    assert p.diff_tx.toPlainText() == p.diff_rx.toPlainText() == p.text.toPlainText()
    # the three result views hold the data (pixel painting of dock tabs is covered natively and on macOS/Linux; Windows' offscreen backend does not paint hidden tabs)
    import pyqtgraph as pg
    assert any(isinstance(i, pg.ScatterPlotItem) and len(i.data) > 100 for i in p.iq.getPlotItem().items)           # the received symbols
    assert any(isinstance(i, pg.PlotCurveItem) and i.xData is not None and len(i.xData) > 2 for i in p.fsm_plot.getPlotItem().items)   # the FSM trajectory
    assert "BEACON DETECTED" in p.pat_log.toPlainText() and "LOCK" in p.pat_log.toPlainText()


def test_fire_the_63kb_payload_adaptively_and_see_the_block_report(window):
    win, app = window
    p = win.payload
    win.clock.seek(float(win.result.t_s[int(np.argmax(win.result.route.kind == 3))]))
    p.gen_btn.click()
    p.mod.setCurrentText("Auto-Adaptive")
    p.seed.setValue(5)
    p.fire.click()
    _wait_for_worker(p, app)
    assert "PASS" in p.badge.text() and p.last.blocks.n_bad == 0 and p.m["blocks"].text().startswith("0 of 1008")
    assert p.last.modulation in ("8PSK", "QPSK", "16QAM", "BPSK") and "auto:" in p.m["mod"].text()


def test_fsm_open_loop_and_forced_route_in_the_gui(window):
    win, app = window
    p = win.payload
    p.rb_text.setChecked(True)
    p.text.setPlainText("no mirror, no data")
    p.mod.setCurrentText("QPSK")
    p.fsm.setChecked(False)
    p.fire.click()
    _wait_for_worker(p, app)
    assert "FAIL" in p.badge.text() and not p.last.sync_ok
    p.fsm.setChecked(True)
    k = next(k for k in range(1, p.route_combo.count()) if p.route_combo.itemText(k).startswith("✗"))
    p.route_combo.setCurrentIndex(k)                                                        # a route with no line of sight right now
    p.fire.click()
    _wait_for_worker(p, app)
    assert "no line of sight" in p.status.text() or "RouteUnavailable" in p.status.text()
    p.route_combo.setCurrentIndex(0)


def test_streaming_from_the_gui_fills_the_log_and_shows_handovers(window):
    win, app = window
    p = win.payload
    win.clock.seek(float(win.result.direct_windows[0][0][0]) - 1800.0)
    p.gen_btn.click()
    p.stream_on.setChecked(True)
    p.frame_bytes.setValue(2048)
    p.stream_min.setValue(60.0)
    p.mod.setCurrentText("Auto-Adaptive")
    p.rate.setValue(40.0)
    p.seed.setValue(6)
    p.fire.click()
    _wait_for_worker(p, app, seconds=120)
    assert "STREAM" in p.badge.text() and p.table.rowCount() == 32 and p.tabs.currentIndex() == 3
    assert "make-before-break" in p.stream_summary.text() and "MATCHES" in p.stream_summary.text()
    assert all(p.table.item(r, 5).text() == "OK" for r in range(32))
    assert _colors(p.stream_plot) > 20
    p.stream_on.setChecked(False)


def test_diff_view_highlights_exactly_the_wrong_bytes(window):
    win, app = window
    from ui.payload_panel import DiffView
    v = DiffView()
    sent = b"hello world, hello orbit"
    got = bytearray(sent)
    got[4] ^= 0x01
    wrong = np.frombuffer(sent, dtype=np.uint8) != np.frombuffer(bytes(got), dtype=np.uint8)
    v.show_bytes(bytes(got), wrong, hexdump=False)
    text = v.toPlainText()
    assert text[:4] == "hell" and text[4] == "n" and text[5:] == " world, hello orbit"
    sel = []
    cur = v.textCursor()
    for pos in range(len(text)):
        cur.setPosition(pos + 1)                                                          # the char format reported is that of the character just before the cursor
        sel.append(cur.charFormat().background().color().name() != "#000000")
    assert sel.count(True) == 1 and sel[4]
    v.show_bytes(sent, None, hexdump=True)
    assert v.toPlainText().startswith("68 65 6c 6c 6f")


def test_cli_transmit_modes(tmp_path, capsys):
    f = tmp_path / "payload.bin"
    f.write_bytes(bytes(range(256)) * 4)
    assert main_app.main(["--send-file", str(f), "--at-hours", "2.5", "--dsp-rate", "10"]) == 0
    out = capsys.readouterr().out
    assert "CRC-32 PASS" in out and "GS-INDIA" in out and "1,024 bytes" in out
    assert main_app.main(["--send-text", "x" * 50, "--at-hours", "2.5", "--no-fsm"]) == 1
    assert "CRC-32 FAIL" in capsys.readouterr().out
    assert main_app.main(["--send-test-payload", "--stream", "--at-hours", "0", "--stream-minutes", "60", "--frame-bytes", "4096"]) == 0
    assert "frames delivered" in capsys.readouterr().out


def test_screenshot_with_the_payload_fired_into_the_dock(tmp_path):
    out = tmp_path / "p.png"
    assert main_app.main(["--screenshot", str(out), "--send-test-payload", "--at-hours", "2.5", "--size", "1900x1000"]) == 0
    assert out.stat().st_size > 50_000
