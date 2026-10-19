"""Build the real window offscreen, drive the clock, and check that every view actually paints something."""
import pytest

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402

import main_app  # noqa: E402
import simulation as sim  # noqa: E402
from ui import geo, map_view  # noqa: E402


@pytest.fixture(scope="module")
def window():
    QApplication, MainWindow, theme = main_app.build_window_classes()
    app = QApplication.instance() or QApplication([])
    theme.apply_dark_theme(app)
    win = MainWindow(sim.SimulationConfig(duration_h=6, step_s=10))
    win.resize(1500, 900)
    win.start_simulation(win.cfg, synchronous=True)
    win.show()
    app.processEvents()
    yield win, app
    win.close()


def _colors(widget):
    img = widget.grab().toImage()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4]
    return len(np.unique(arr.reshape(-1, 4), axis=0))


def test_all_four_views_paint(window):
    win, app = window
    win.clock.seek(3600.0)
    app.processEvents()
    for w in (win.globe, win.map, win.timeline, win.telemetry):
        assert _colors(w) > 40, f"{type(w).__name__} looks blank"


def test_clock_step_play_and_views_stay_in_sync(window):
    win, app = window
    win.clock.seek(0.0)
    win.clock.step(1)
    assert win.clock.t == pytest.approx(10.0) and win.globe.idx == win.map.idx == 1
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


def test_telemetry_follows_link_state(window):
    win, app = window
    link = win.result.primary_link
    i_vis, i_blk = int(np.argmax(link.visible)), int(np.argmax(~link.visible))
    win.telemetry.set_index(i_vis)
    assert win.telemetry._link_labels["status"].text() == "LOCKED"
    win.telemetry.set_index(i_blk)
    assert win.telemetry._link_labels["status"].text() == "OCCLUDED"
    assert win.telemetry._link_labels["rx"].text() == "—"


def test_modulation_switch_updates_readout(window):
    win, app = window
    i = int(np.argmax(win.result.primary_link.visible))
    win.telemetry.set_index(i)
    win.telemetry._mod.setCurrentText("BPSK")
    ber_bpsk = win.telemetry._link_labels["ber"].text()
    win.telemetry._mod.setCurrentText("16QAM")
    assert win.telemetry._link_labels["ber"].text() != ber_bpsk


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
    assert main_app.main(["--screenshot", str(out), "--hours", "3", "--at-hours", "1", "--size", "1200x760"]) == 0
    assert out.stat().st_size > 20_000


def test_headless_cli(tmp_path, capsys):
    csv_path = tmp_path / "o.csv"
    assert main_app.main(["--headless", "--hours", "2", "--csv", str(csv_path)]) == 0
    assert "access windows" in capsys.readouterr().out and csv_path.exists()
