"""The Radiation-Tolerant AI Scheduler tab, built inside the real window (offscreen): blade grid, chaos controls, loss chart, map/globe overlay, export and
the 16QAM optical downlink."""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

import main_app  # noqa: E402
import radiation_pytorch_scheduler as rps  # noqa: E402
import seu_injector as si  # noqa: E402
from helpers import direct_cfg  # noqa: E402
from ui import radiation_panel as rp  # noqa: E402


@pytest.fixture(scope="module")
def window():
    QApplication, MainWindow, theme = main_app.build_window_classes()
    app = QApplication.instance() or QApplication([])
    theme.apply_dark_theme(app)
    win = MainWindow(direct_cfg(duration_h=12, step_s=30))
    win.resize(1800, 1000)
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
    p = win.radiation
    p.flare_btn.setChecked(False)
    p.protect_btn.setChecked(True)
    p.hot_btn.setChecked(False)
    p._engine_pref = "numpy"                                    # fast and deterministic; the torch engine is covered in test_radiation_scheduler
    p.restart()
    yield p
    p.flare_btn.setChecked(False)
    p.hot_btn.setChecked(False)
    p.protect_btn.setChecked(True)
    p._engine_pref = "auto"


def _grab(widget):
    img = widget.grab().toImage()
    return np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4].copy()


def _run(panel, seconds, flare_every=None):
    t = panel.orch.t
    end = t + seconds
    while t < end:
        t += 80.0
        panel.set_time(t)


def test_tab_exists_beside_telemetry_and_has_the_32_blade_grid(window):
    win, app = window
    assert win.radiation_dock in win.tabifiedDockWidgets(win.tele_dock) and "RADIATION-TOLERANT AI SCHEDULER" in win.radiation_dock.windowTitle()
    p = win.radiation
    assert len(p.cells) == 32 and p.cells[0].text() == "01" and p.cells[31].text() == "16"
    labels = [lab.text() for lab in p.findChildren(type(p.status)) if lab.text().startswith("BANK")]
    assert labels == ["BANK A", "BANK B"]
    for text in ("Force Solar Flare / CME Event (1000x Flux)", "Toggle ECC & TMR Protection ON/OFF", "Export Recovered .pt Weights"):
        assert any(b.text() == text for b in p.findChildren(type(p.flare_btn)))
    assert any("simulation assumptions" in lab.text() for lab in p.findChildren(type(p.status)))


def test_blades_start_green_and_stay_live_with_the_clock(window, panel):
    win, app = window
    win.clock.seek(0.0)
    _run(panel, 600)
    assert panel.orch.step_no >= 5
    states = set(panel._cell_state)
    assert rps.HEALTHY in states and len(panel.curve_p.xData) == panel.orch.step_no
    assert "zone" in panel.status.text() and "engine numpy" in panel.status.text()
    assert "Blade A01" in panel.cells[0].toolTip()


def test_flare_turns_blades_red_and_yellow_and_the_unprotected_line_dies_while_protected_converges(window, panel):
    win, app = window
    panel.flare_btn.setChecked(True)
    assert panel.orch.flare == si.FLARE_MULT and "SOLAR FLARE" in panel.log.toPlainText()
    seen = set()
    for _ in range(25):
        panel.set_time(panel.orch.t + 80.0)
        seen |= set(panel._cell_state)
    assert rps.QUARANTINED in seen and rps.SCRUBBED in seen                                # red (exponent strike caught) and yellow (mantissa flip scrubbed)
    assert rps.TMR_STATE in seen                                                          # a flare switches to TMR shielding: purple
    assert panel.orch.baseline.poisoned and not panel.orch.protected.poisoned
    yu = np.asarray(panel.curve_u.yData, dtype=float)
    assert np.isnan(yu).any() and np.isfinite(np.asarray(panel.curve_p.yData)).all()      # the red line ends in NaN, the green one does not
    assert len(panel.nan_mark.data) == 1 and "NaN at step" in panel.nan_text.textItem.toPlainText()
    assert "training destroyed" in panel.stats.text() and "flare ACTIVE" in panel.status.text()


def test_protection_toggle_and_the_green_line_follows_the_red_one(window, panel):
    win, app = window
    panel.protect_btn.setChecked(False)
    panel.flare_btn.setChecked(True)
    for _ in range(25):
        panel.set_time(panel.orch.t + 80.0)
    assert panel.orch.protected.poisoned and "protection OFF" in panel.status.text() and "protection is OFF" in panel.stats.text()
    assert rps.TMR_STATE not in panel._cell_state


def test_hot_core_hibernates_bank_b_in_grey(window, panel):
    win, app = window
    panel.hot_btn.setChecked(True)
    _run(panel, 400)
    assert panel._cell_state[16:] == [rps.HIBERNATED] * 16 and panel._cell_state[:16].count(rps.HIBERNATED) == 0
    assert rp.CELL_COLORS[rps.HIBERNATED] in panel.cells[20].styleSheet() and "SOFT THROTTLE" in panel.status.text()
    panel.hot_btn.setChecked(False)
    _run(panel, 400)
    assert rps.HIBERNATED not in panel._cell_state


def test_forced_latchup_shows_grey_and_logs(window, panel):
    win, app = window
    _run(panel, 160)
    panel.sel_btn.click()
    _run(panel, 160)
    assert "latch-up forced" in panel.log.toPlainText()
    assert sum(b.latchups for b in panel.orch.protected.blades) == 1


def test_hand_injected_exponent_strike_flashes_red_then_heals(window, panel):
    win, app = window
    _run(panel, 160)
    panel.orch.inject(si.Upset(4, "SEU", 0.1, "weights", 0.3, 30.5 / 32))
    panel.set_time(panel.orch.t + 20.0)                                                   # exactly one training step
    assert panel._cell_state[4] == rps.QUARANTINED and rp.CELL_COLORS[rps.QUARANTINED] in panel.cells[4].styleSheet()
    _run(panel, 800)
    assert panel._cell_state[4] == rps.HEALTHY and np.array_equal(panel.orch.protected.weights_np(4), panel.orch.protected.obc.reference())


def test_overlay_paints_on_map_and_globe_and_can_be_switched_off(window, panel):
    win, app = window
    win.clock.seek(3600.0)
    panel.overlay_cb.setChecked(True)
    panel.set_time(3700.0)
    app.processEvents()
    assert win.map.rad_overlay and win.globe.rad_overlay
    on_map, on_globe = _grab(win.map), _grab(win.globe)
    panel.overlay_cb.setChecked(False)
    app.processEvents()
    assert not win.map.rad_overlay and not win.globe.rad_overlay
    off_map, off_globe = _grab(win.map), _grab(win.globe)
    assert (on_map != off_map).any(axis=-1).mean() > 0.01                                    # the SAA box and the auroral bands changed many pixels
    assert (on_globe != off_globe).any(axis=-1).mean() > 0.002
    panel.overlay_cb.setChecked(True)


def test_overlay_glows_brighter_when_the_flux_is_higher():
    from ui import radiation_overlay as ro
    assert ro.pulse(1.0) < ro.pulse(100.0) < ro.pulse(1000.0) <= 1.0 and ro.pulse(1.0) >= 0.4
    lon, lat = ro.saa_outline()
    assert lon.min() == si.SAA_LON_DEG[0] and lon.max() == si.SAA_LON_DEG[1] and lat.min() == si.SAA_LAT_DEG[0] and lat.max() == si.SAA_LAT_DEG[1]


def _wait(win, app, seconds=90):
    t0 = time.time()
    while win.payload._thread is not None and time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert win.payload._thread is None


def test_export_feeds_the_16qam_optical_downlink(window, panel, tmp_path):
    win, app = window
    _run(panel, 800)
    ext = ".pt" if rps.TORCH_AVAILABLE else ".npz"
    out = panel.export_to(tmp_path / f"alpha_sigma_recovered{ext}")
    assert out.exists() and out.stat().st_size < 64_000
    assert win.payload._file_name == out.name and win.payload._file_bytes == out.read_bytes()
    assert win.payload.mod.currentText() == "16QAM"
    _wait(win, app)                                                                            # the dock's worker thread transmits (or reports no line of sight)
    assert win.payload.status.text() != ""
    meta = rps.load_checkpoint_meta(out)
    assert meta["sha256"] == rps.sha256_hex(panel.orch.protected.consensus_weights()) and meta["step"] == panel.orch.protected.step_no


def test_export_without_downlink_leaves_the_payload_dock_alone(window, panel, tmp_path):
    win, app = window
    _run(panel, 160)
    win.payload._file_bytes, win.payload._file_name = None, ""
    panel.downlink_cb.setChecked(False)
    panel.export_to(tmp_path / "w.npz")
    assert win.payload._file_bytes is None
    panel.downlink_cb.setChecked(True)


def test_restart_resets_training_but_keeps_the_chaos_settings(window, panel):
    win, app = window
    panel.flare_btn.setChecked(True)
    _run(panel, 400)
    assert panel.orch.step_no > 0
    panel.restart()
    assert panel.orch.step_no == 0 and panel.orch.flare == si.FLARE_MULT
    panel.flare_btn.setChecked(False)


def test_seek_backwards_or_far_does_not_train_and_does_not_crash(window, panel):
    win, app = window
    _run(panel, 400)
    n = panel.orch.step_no
    win.clock.seek(10.0)
    win.clock.seek(30000.0)
    assert panel.orch.step_no == n


def test_screenshot_with_the_radiation_tab(tmp_path):
    out = tmp_path / "r.png"
    assert main_app.main(["--screenshot", str(out), "--show-radiation", "--rad-flare", "--at-hours", "1.0", "--rad-minutes", "40", "--hours", "6", "--size", "1800x1000"]) == 0
    assert out.stat().st_size > 50_000
