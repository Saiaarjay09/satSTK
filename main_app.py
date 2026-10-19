#!/usr/bin/env python3
"""
satSTK - Orbital Mission & Laser Link Simulator.  Launch with:

    python main_app.py                          the desktop app: first place the beam origin on the map, then simulate
    python main_app.py --lat 28.3 --lon -16.5   skip the placement window and start with this ground station
    python main_app.py --headless               no window: run the scenario and print the summary
    python main_app.py --headless --csv out.csv also write the beam's time series (only when you ask)
    python main_app.py --screenshot shot.png    draw the window once and save a picture (used for the README and CI)

Everything runs locally with NumPy/SciPy/Skyfield and Qt. No Ansys STK, no Windows COM, no network, no account.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

__version__ = "2.0.0"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="satSTK: native orbital mission and 1550 nm laser-link simulator (ground station to satellite)")
    ap.add_argument("--headless", action="store_true", help="run the scenario without a window and print a summary")
    ap.add_argument("--csv", metavar="FILE", help="with --headless: also write the beam's time series to this CSV")
    ap.add_argument("--screenshot", metavar="PNG", help="open the window, draw it once, save a picture and exit")
    ap.add_argument("--at-hours", type=float, default=None, help="with --screenshot: simulation time to show (default: the middle of the first pass)")
    ap.add_argument("--size", default="1700x960", help="with --screenshot: window size, e.g. 1700x960")
    ap.add_argument("--selftest", action="store_true", help="build the window offscreen, render every pane, print SELFTEST OK (used to test the packaged app)")
    ap.add_argument("--selftest-file", metavar="FILE", help="with --selftest: also write the result to this file")
    g = ap.add_argument_group("scenario")
    g.add_argument("--lat", type=float, help="ground station latitude, degrees north (skips the placement window)")
    g.add_argument("--lon", type=float, help="ground station longitude, degrees east")
    g.add_argument("--alt-m", type=float, default=None, help="ground station altitude in metres")
    g.add_argument("--min-elevation", type=float, default=None, help="lowest usable elevation angle in degrees (default 20)")
    g.add_argument("--weather", default=None, help="weather preset: Clear, Hazy, 'Thin cirrus', 'Light fog', 'Dense fog', Overcast, Rain, Thunderstorm, 'Hot afternoon'")
    g.add_argument("--events", action="store_true", help="switch random events on")
    g.add_argument("--event-seed", type=int, default=1, help="random seed for events (same seed = same events)")
    g.add_argument("--event-rate", type=float, default=1.0, help="event frequency multiplier (1 = typical, 100 = everything goes wrong)")
    g.add_argument("--no-geo-relay", action="store_true", help="switch the GEO optical relay network OFF (direct ground-to-satellite passes only)")
    g.add_argument("--two-satellites", action="store_true", help="add the second satellite, Orbital-Node-Beta")
    g.add_argument("--hours", type=float, default=24.0, help="scenario length in hours (default 24)")
    g.add_argument("--step", type=float, default=10.0, help="time step in seconds: 1 or 10 (default 10)")
    g.add_argument("--start", default="2026-12-21T00:00:00", help="scenario start, UTC, ISO format (default 2026-12-21T00:00:00)")
    ap.add_argument("--version", action="version", version=f"satSTK {__version__}")
    return ap.parse_args(argv)


def config_from_args(args):
    import orbital_engine as oe
    from atmosphere import WEATHER_PRESETS
    from events import EventsConfig
    from simulation import SimulationConfig, two_satellites, default_satellites
    start = datetime.fromisoformat(args.start)
    start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start.astimezone(timezone.utc)
    cfg = SimulationConfig(start=start, duration_h=args.hours, step_s=args.step,
                           satellites=two_satellites() if args.two_satellites else default_satellites(),
                           events=EventsConfig(enabled=args.events, seed=args.event_seed, rate_scale=args.event_rate))
    gs = cfg.ground
    if args.lat is not None or args.lon is not None:
        gs = oe.GroundStation("Ground station", args.lat if args.lat is not None else gs.lat_deg, args.lon if args.lon is not None else gs.lon_deg, gs.alt_m, gs.min_elevation_deg)
    if args.alt_m is not None:
        gs.alt_m = args.alt_m
    if args.min_elevation is not None:
        gs.min_elevation_deg = args.min_elevation
    cfg.ground = gs
    cfg.relay.enabled = not args.no_geo_relay
    if args.weather:
        match = {k.lower(): v for k, v in WEATHER_PRESETS.items()}.get(args.weather.lower())
        if match is None:
            raise SystemExit(f"unknown weather preset {args.weather!r}; choose from {list(WEATHER_PRESETS)}")
        cfg.atmosphere = match
    return cfg


def run_headless(args) -> int:
    from simulation import run_simulation
    result = run_simulation(config_from_args(args))
    print(result.summary_text())
    if args.csv:
        result.write_csv(args.csv)
        print(f"Wrote {args.csv}")
    return 0


def build_window_classes():
    """Qt classes are created lazily so `--headless` and the physics tests never need a display or PySide6."""
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                                   QMessageBox, QProgressBar, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget)

    import orbital_engine as oe
    from simulation import SimulationConfig, run_simulation
    from ui import theme
    from ui.clock import SPEEDS, PlaybackClock
    from ui.conditions_panel import ConditionsPanel
    from ui.events_dialog import EventsDialog
    from ui.globe_view import GlobeView
    from ui.map_view import MapView
    from ui.placement_dialog import PlacementDialog
    from ui.telemetry_panel import TelemetryPanel
    from ui.timeline_view import TimelineView

    class SimWorker(QObject):
        progress = Signal(str, float)
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        @Slot()
        def run(self):
            try:
                self.finished.emit(run_simulation(self.cfg, lambda m, f: self.progress.emit(m, f)))
            except Exception as exc:    # shown to the user instead of crashing the app
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class MainWindow(QMainWindow):
        def __init__(self, cfg: SimulationConfig | None = None):
            super().__init__()
            self.setWindowTitle(f"satSTK {__version__} - Orbital Mission & Laser Link Simulator")
            self.resize(1700, 960)
            self.cfg = cfg or SimulationConfig()
            self.result = None
            self._thread = None
            self._pending = None
            self._keep_time = None
            self.clock = PlaybackClock(self)

            self.globe, self.map, self.timeline = GlobeView(), MapView(), TimelineView()
            self.telemetry = TelemetryPanel()
            self.conditions = ConditionsPanel(self.cfg)

            top = QSplitter(Qt.Orientation.Horizontal)
            top.addWidget(self._titled("3D VIEW  ·  inertial", self.globe))
            top.addWidget(self._titled("2D MAP  ·  ground tracks  ·  Ctrl/⌘+click moves the ground station", self.map))
            top.setSizes([560, 720])
            bottom = QWidget()
            bl = QVBoxLayout(bottom)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(0)
            bl.addWidget(self._transport_bar())
            bl.addWidget(self.timeline)
            vs = QSplitter(Qt.Orientation.Vertical)
            vs.addWidget(top)
            vs.addWidget(bottom)
            vs.setStretchFactor(0, 5)
            vs.setStretchFactor(1, 0)
            vs.setSizes([600, 320])
            self.setCentralWidget(vs)

            self.tele_dock = self._dock("TELEMETRY  ·  link budget", self.telemetry, Qt.DockWidgetArea.RightDockWidgetArea, 470)
            self.cond_dock = self._dock("CONDITIONS  ·  tinker here", self.conditions, Qt.DockWidgetArea.LeftDockWidgetArea, 400)

            self._build_menus()
            self.progress = QProgressBar()
            self.progress.setMaximumWidth(220)
            self.progress.setRange(0, 100)
            self.progress.hide()
            self.statusBar().addPermanentWidget(self.progress)

            self._debounce = QTimer(self)
            self._debounce.setSingleShot(True)
            self._debounce.setInterval(450)
            self._debounce.timeout.connect(self.recompute_from_panel)

            self.clock.time_changed.connect(self._on_time)
            self.clock.state_changed.connect(self._on_state)
            self.timeline.seek_requested.connect(self.clock.seek)
            self.conditions.changed.connect(self._conditions_changed)
            self.conditions.apply_btn.clicked.connect(self.recompute_from_panel)
            self.conditions.place_requested.connect(self.place_station)
            self.conditions.events_requested.connect(self.edit_events)
            self.map.station_move_requested.connect(self._station_clicked)

        # ---------------------------------------------------------------- layout helpers
        @staticmethod
        def _titled(title: str, widget: QWidget) -> QWidget:
            box = QWidget()
            lay = QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lab = QLabel(title)
            lab.setStyleSheet(f"background: {theme.PANEL_2}; color: {theme.ACCENT}; font-weight: bold; padding: 4px 8px;")
            lay.addWidget(lab)
            lay.addWidget(widget, 1)
            return box

        def _dock(self, title: str, widget: QWidget, area, min_width: int) -> QDockWidget:
            dock = QDockWidget(title, self)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            scroll.setMinimumWidth(min_width)
            dock.setWidget(scroll)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                             | QDockWidget.DockWidgetFeature.DockWidgetClosable)
            self.addDockWidget(area, dock)
            return dock

        def _transport_bar(self) -> QWidget:
            bar = QWidget()
            bar.setStyleSheet(f"background: {theme.PANEL_2};")
            lay = QHBoxLayout(bar)
            lay.setContentsMargins(8, 4, 8, 4)
            self.btn_reset = QPushButton("⏮ Start")
            self.btn_back = QPushButton("◀ Step")
            self.btn_play = QPushButton("▶ Play")
            self.btn_play.setMinimumWidth(90)
            self.btn_step = QPushButton("Step ▶")
            self.btn_pass = QPushButton("Next pass ⏭")
            self.geo_btn = QPushButton("GEO relay: ON")
            self.geo_btn.setCheckable(True)
            self.geo_btn.setChecked(self.cfg.relay.enabled)
            self.geo_btn.setToolTip("Switch the geostationary optical relay network on or off and compare the contact uptime")
            self.geo_btn.toggled.connect(self._geo_button_toggled)
            self.speed = QComboBox()
            for label, value in SPEEDS:
                self.speed.addItem(label, value)
            self.speed.setCurrentIndex(3)
            self.loop = QCheckBox("Loop")
            self.loop.setChecked(True)
            self.time_label = QLabel("—")
            self.time_label.setFont(theme.mono_font(11))
            self.btn_reset.clicked.connect(lambda: (self.clock.pause(), self.clock.seek(0.0)))
            self.btn_back.clicked.connect(lambda: self.clock.step(-1))
            self.btn_step.clicked.connect(lambda: self.clock.step(1))
            self.btn_play.clicked.connect(self.clock.toggle)
            self.btn_pass.clicked.connect(self.next_pass)
            self.speed.currentIndexChanged.connect(lambda: self.clock.set_speed(self.speed.currentData()))
            self.loop.toggled.connect(lambda v: setattr(self.clock, "loop", v))
            self.clock.set_speed(self.speed.currentData())
            for w in (self.btn_reset, self.btn_back, self.btn_play, self.btn_step, self.btn_pass):
                lay.addWidget(w)
            lay.addSpacing(12)
            lay.addWidget(self.geo_btn)
            lay.addSpacing(12)
            lay.addWidget(QLabel("Speed"))
            lay.addWidget(self.speed)
            lay.addWidget(self.loop)
            lay.addStretch(1)
            lay.addWidget(self.time_label)
            return bar

        def _build_menus(self) -> None:
            m = self.menuBar().addMenu("&File")
            for text, key, fn in (("Place ground station (beam origin) ...", "Ctrl+G", self.place_station), ("Random & rare events ...", "Ctrl+E", self.edit_events),
                                  ("Export beam data as CSV ...", None, self.export_csv)):
                act = QAction(text, self)
                if key:
                    act.setShortcut(QKeySequence(key))
                act.triggered.connect(fn)
                m.addAction(act)
            m.addSeparator()
            act = QAction("Quit", self)
            act.setShortcut(QKeySequence.StandardKey.Quit)
            act.triggered.connect(self.close)
            m.addAction(act)
            m = self.menuBar().addMenu("&View")
            m.addAction(self.cond_dock.toggleViewAction())
            m.addAction(self.tele_dock.toggleViewAction())
            m = self.menuBar().addMenu("&Playback")
            for text, key, fn in (("Play / Pause", "Space", self.clock.toggle), ("Step forward", "Right", lambda: self.clock.step(1)),
                                  ("Step back", "Left", lambda: self.clock.step(-1)), ("Next pass", "N", self.next_pass)):
                act = QAction(text, self)
                act.setShortcut(QKeySequence(key))
                act.triggered.connect(fn)
                m.addAction(act)
            m = self.menuBar().addMenu("&Help")
            act = QAction("About / limits of this model", self)
            act.triggered.connect(self.about)
            m.addAction(act)

        # ---------------------------------------------------------------- running scenarios
        def start_simulation(self, cfg: SimulationConfig, synchronous: bool = False, keep_time: bool = False) -> None:
            """Compute a scenario (in a background thread so the window stays responsive) and show it when done."""
            self.cfg = cfg
            self.clock.pause()
            self._keep_time = self.clock.t if keep_time and self.result is not None else None
            if synchronous:
                self.set_result(run_simulation(cfg))
                return
            if self._thread is not None:
                self._pending = (cfg, keep_time)       # busy: run this one as soon as the current one finishes
                return
            self.progress.show()
            self.progress.setValue(0)
            self._thread = QThread(self)
            self._worker = SimWorker(cfg)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._on_progress)       # a bound method: Qt delivers it on the GUI thread
            self._worker.finished.connect(self._finished)
            self._worker.failed.connect(self._failed)
            self._thread.start()

        @Slot(str, float)
        def _on_progress(self, message: str, fraction: float) -> None:
            self.statusBar().showMessage(message)
            self.progress.setValue(int(100 * fraction))

        def _cleanup_thread(self) -> None:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait()
                self._thread = None
            self.progress.hide()

        def _run_pending(self) -> bool:
            if self._pending is not None:
                cfg, keep = self._pending
                self._pending = None
                self.start_simulation(cfg, keep_time=keep)
                return True
            return False

        @Slot(object)
        def _finished(self, result) -> None:
            self._cleanup_thread()
            if not self._run_pending():
                self.set_result(result)

        @Slot(str)
        def _failed(self, message: str) -> None:
            self._cleanup_thread()
            self.statusBar().showMessage("Scenario failed")
            QMessageBox.warning(self, "Scenario failed", message)

        def set_result(self, result) -> None:
            self.result = result
            self.globe.set_result(result)
            self.map.set_result(result)
            self.timeline.set_result(result)
            self.telemetry.set_result(result)
            self.clock.configure(result.t_s[-1], result.config.step_s)
            t0 = self._keep_time if self._keep_time is not None else self._first_pass_time()
            self._keep_time = None
            self.clock.seek(min(t0, result.t_s[-1]))
            self._on_time(self.clock.t)
            gs = result.config.ground
            n_pass = len(result.primary_windows)
            rt = result.route
            self.statusBar().showMessage(f"Beam origin: {gs.name} ({gs.lat_deg:+.2f}°, {gs.lon_deg:+.2f}°).  GEO relay {'ON' if result.network_link is not None else 'OFF'}: "
                                         f"usable data {rt.uptime_pct:.2f} % of {result.config.duration_h:g} h (direct alone {rt.direct_uptime_pct:.2f} %), "
                                         f"{len(rt.handovers)} handover(s), {len(result.events)} event(s).  Space = play/pause, N = next pass, ←/→ = step.")

        def _first_pass_time(self) -> float:
            w = self.result.direct_windows.get(self.result.target, self.result.primary_windows)
            return float(max(0.0, w[0][0] - 120.0)) if len(w) else 0.0

        # ---------------------------------------------------------------- user actions
        def _geo_button_toggled(self, on: bool) -> None:
            self.geo_btn.setText("GEO relay: ON" if on else "GEO relay: OFF")
            if self.conditions.geo_on.isChecked() != on:
                self.conditions.geo_on.setChecked(on)         # this triggers the (debounced) recompute

        def _sync_geo_button(self, on: bool) -> None:
            self.geo_btn.blockSignals(True)
            self.geo_btn.setChecked(on)
            self.geo_btn.setText("GEO relay: ON" if on else "GEO relay: OFF")
            self.geo_btn.blockSignals(False)

        def _conditions_changed(self) -> None:
            self._sync_geo_button(self.conditions.geo_on.isChecked())
            if self.conditions.live.isChecked():
                self._debounce.start()

        def recompute_from_panel(self) -> None:
            try:
                cfg = self.conditions.config(self.cfg)
                cfg.validate()
            except ValueError as exc:
                QMessageBox.warning(self, "Conditions", str(exc))
                return
            self.start_simulation(cfg, keep_time=True)

        def place_station(self) -> None:
            dlg = PlacementDialog(self.cfg.ground, self.cfg.satellites[0].altitude_km, self)
            if dlg.exec():
                cfg = self.conditions.config(self.cfg)
                cfg.ground = dlg.station()
                self.conditions.load(cfg)
                self.start_simulation(cfg)

        def _station_clicked(self, lat: float, lon: float) -> None:
            cfg = self.conditions.config(self.cfg)
            cfg.ground = oe.GroundStation("Ground station", lat, lon, 0.0, cfg.ground.min_elevation_deg)
            self.conditions.load(cfg)
            self.start_simulation(cfg)

        def edit_events(self) -> None:
            cfg = self.conditions.config(self.cfg)
            dlg = EventsDialog(cfg.events, cfg.duration_h, self.result.events if self.result else (), self)
            if dlg.exec():
                self.conditions.set_events_config(dlg.config())
                self.recompute_from_panel()

        def next_pass(self) -> None:
            if self.result is None or not len(self.result.direct_windows.get(self.result.target, [])):
                return
            w = self.result.direct_windows[self.result.target]
            later = [s for s, e in w if s - 60.0 > self.clock.t]
            self.clock.seek(max(0.0, (later[0] if later else w[0][0]) - 60.0))

        def export_csv(self) -> None:
            if self.result is None:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export beam data", "satstk_beam.csv", "CSV (*.csv)")
            if path:
                self.result.write_csv(path)
                self.statusBar().showMessage(f"Wrote {path}")

        def about(self) -> None:
            QMessageBox.information(self, "About satSTK", (
                f"satSTK {__version__}\n\nA standalone simulator for a 1550 nm laser beam from a fixed ground station to a satellite.\n"
                "Runs locally; no licence, no network, nothing is stored unless you export a CSV.\n\n"
                "Limits: J2-only orbit model (no drag, no solar pressure, no third-body gravity); Sun position from a short analytic formula; "
                "the atmosphere, weather and turbulence models are engineering rules of thumb, and the random events are illustrative "
                "(rates and effects are plausible guesses, not statistics); the link budget is a textbook calculation, not a waveform "
                "simulation. Agreement with Ansys STK or measured hardware has NOT been validated.\n\nLand outlines: Natural Earth (public domain)."))

        # ---------------------------------------------------------------- clock reactions
        def _on_time(self, t: float) -> None:
            if self.result is None:
                return
            i = self.result.index_at(t)
            self.globe.set_index(i)
            self.map.set_index(i)
            self.timeline.set_time(t)
            self.telemetry.set_index(i)
            self.time_label.setText(f"{self.result.utc(t):%Y-%m-%d %H:%M:%S} UTC    T+ {t / 3600:6.3f} h")

        def _on_state(self, playing: bool) -> None:
            self.btn_play.setText("⏸ Pause" if playing else "▶ Play")

    return QApplication, MainWindow, theme


def run_gui(args) -> int:
    QApplication, MainWindow, theme = build_window_classes()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("satSTK")
    theme.apply_dark_theme(app)
    win = MainWindow(config_from_args(args))
    if args.screenshot:
        w, h = (int(v) for v in args.size.lower().split("x"))
        win.resize(w, h)
        win.start_simulation(win.cfg, synchronous=True)
        if args.at_hours is not None:
            win.clock.seek(args.at_hours * 3600.0)
        win.show()
        app.processEvents()
        if not win.grab().save(args.screenshot):
            print(f"Could not write {args.screenshot} (does the folder exist?)", file=sys.stderr)
            return 1
        print(f"Saved {args.screenshot}")
        return 0
    win.show()
    app.processEvents()
    if args.lat is None and args.lon is None:
        win.place_station()                    # first step: put the beam origin on the map; Cancel keeps the default station
        if win.result is None and win._thread is None:
            win.start_simulation(win.cfg)
    else:
        win.start_simulation(win.cfg)
    return app.exec()


def run_selftest(args) -> int:
    """Smoke test for the packaged app: bundled data, Qt, the engines and every pane must all work. Needs no display."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    lines, ok = [], True
    try:
        import numpy as np
        QApplication, MainWindow, theme = build_window_classes()
        from ui import geo
        app = QApplication.instance() or QApplication(sys.argv[:1])
        theme.apply_dark_theme(app)
        lines.append(f"land outlines: {len(geo.land_rings())} polygons (data file bundled: {len(geo.land_rings()) > 100})")
        cfg = config_from_args(parse_args(["--events", "--event-rate", "5", "--step", "30"]))
        win = MainWindow(cfg)
        win.resize(1500, 850)
        win.start_simulation(win.cfg, synchronous=True)
        res = win.result
        ok = ok and len(res.direct_windows[res.target]) > 0 and res.route is not None and res.network_link is not None
        win.clock.seek(float(np.mean(res.direct_windows[res.target][0])))
        win.show()
        app.processEvents()
        for name, widget in (("globe", win.globe), ("map", win.map), ("timeline", win.timeline), ("telemetry", win.telemetry), ("conditions", win.conditions)):
            img = widget.grab().toImage()
            arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4]
            colours = len(np.unique(arr.reshape(-1, 4), axis=0))
            lines.append(f"{name}: {colours} distinct colours")
            ok = ok and colours > 40
        ok = ok and len(geo.land_rings()) > 100
        lines.append(f"direct passes: {len(res.direct_windows[res.target])}, handovers: {len(res.route.handovers)}, uptime: {res.route.uptime_pct:.2f} %, events: {len(res.events)}, "
                     f"route now: {res.route.label(res.index_at(win.clock.t))}")
    except Exception as exc:
        ok = False
        lines.append(f"error: {type(exc).__name__}: {exc}")
    lines.append("SELFTEST OK" if ok else "SELFTEST FAILED")
    text = "\n".join(lines)
    if sys.stdout is not None:
        print(text)
    if args.selftest_file:
        with open(args.selftest_file, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest(args)
    return run_headless(args) if args.headless else run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
