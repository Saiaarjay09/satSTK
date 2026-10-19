#!/usr/bin/env python3
"""
satSTK - Orbital Mission & Laser Link Simulator.  Launch with:

    python main_app.py                          the desktop app (3D globe, 2D map, timeline, telemetry)
    python main_app.py --headless               no window: run the scenario and print the summary
    python main_app.py --headless --csv out.csv also write the link time series (only when you ask)
    python main_app.py --screenshot shot.png    draw the window once and save a picture (used for the README and CI)

Everything runs locally with NumPy/SciPy/Skyfield and Qt. No Ansys STK, no Windows COM, no network, no account.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

__version__ = "0.2.0"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="satSTK: native orbital mission and 1550 nm laser-link simulator")
    ap.add_argument("--headless", action="store_true", help="run the scenario without a window and print a summary")
    ap.add_argument("--csv", metavar="FILE", help="with --headless: also write the primary link's time series to this CSV")
    ap.add_argument("--screenshot", metavar="PNG", help="open the window, draw it once, save a picture and exit")
    ap.add_argument("--at-hours", type=float, default=2.0, help="with --screenshot: simulation time to show (default 2.0 h)")
    ap.add_argument("--size", default="1600x960", help="with --screenshot: window size, e.g. 1600x960")
    ap.add_argument("--hours", type=float, default=24.0, help="scenario length in hours (default 24)")
    ap.add_argument("--step", type=float, default=10.0, help="time step in seconds: 1 or 10 (default 10)")
    ap.add_argument("--start", default="2026-12-21T00:00:00", help="scenario start, UTC, ISO format (default 2026-12-21T00:00:00)")
    ap.add_argument("--selftest", action="store_true", help="build the window offscreen, render every pane, print SELFTEST OK (used to test the packaged app)")
    ap.add_argument("--selftest-file", metavar="FILE", help="with --selftest: also write the result to this file")
    ap.add_argument("--version", action="version", version=f"satSTK {__version__}")
    return ap.parse_args(argv)


def config_from_args(args):
    from simulation import SimulationConfig
    start = datetime.fromisoformat(args.start)
    start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start.astimezone(timezone.utc)
    return SimulationConfig(start=start, duration_h=args.hours, step_s=args.step)


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
    from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                                   QMessageBox, QProgressBar, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget)

    from simulation import SimulationConfig, run_simulation
    from ui import theme
    from ui.clock import SPEEDS, PlaybackClock
    from ui.globe_view import GlobeView
    from ui.map_view import MapView
    from ui.scenario_dialog import ScenarioDialog
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
            self.resize(1600, 960)
            self.cfg = cfg or SimulationConfig()
            self.result = None
            self._thread = None
            self.clock = PlaybackClock(self)

            self.globe, self.map, self.timeline = GlobeView(), MapView(), TimelineView()
            self.telemetry = TelemetryPanel()

            top = QSplitter(Qt.Orientation.Horizontal)
            top.addWidget(self._titled("3D VIEW  ·  inertial", self.globe))
            top.addWidget(self._titled("2D MAP  ·  ground tracks and footprints", self.map))
            top.setSizes([600, 800])
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
            vs.setSizes([700, 190])
            self.setCentralWidget(vs)

            dock = QDockWidget("TELEMETRY  ·  link budget", self)
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(self.telemetry)
            area.setMinimumWidth(470)
            dock.setWidget(area)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

            self._build_menus()
            self.progress = QProgressBar()
            self.progress.setMaximumWidth(220)
            self.progress.setRange(0, 100)
            self.progress.hide()
            self.statusBar().addPermanentWidget(self.progress)

            self.clock.time_changed.connect(self._on_time)
            self.clock.state_changed.connect(self._on_state)
            self.timeline.seek_requested.connect(self.clock.seek)

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
            self.speed.currentIndexChanged.connect(lambda: self.clock.set_speed(self.speed.currentData()))
            self.loop.toggled.connect(lambda v: setattr(self.clock, "loop", v))
            self.clock.set_speed(self.speed.currentData())
            for w in (self.btn_reset, self.btn_back, self.btn_play, self.btn_step):
                lay.addWidget(w)
            lay.addSpacing(12)
            lay.addWidget(QLabel("Speed"))
            lay.addWidget(self.speed)
            lay.addWidget(self.loop)
            lay.addStretch(1)
            lay.addWidget(self.time_label)
            return bar

        def _build_menus(self) -> None:
            m = self.menuBar().addMenu("&File")
            act = QAction("Scenario settings...", self)
            act.setShortcut(QKeySequence("Ctrl+N"))
            act.triggered.connect(self.edit_scenario)
            m.addAction(act)
            act = QAction("Export link data as CSV...", self)
            act.triggered.connect(self.export_csv)
            m.addAction(act)
            m.addSeparator()
            act = QAction("Quit", self)
            act.setShortcut(QKeySequence.StandardKey.Quit)
            act.triggered.connect(self.close)
            m.addAction(act)
            m = self.menuBar().addMenu("&Playback")
            for text, key, fn in (("Play / Pause", "Space", self.clock.toggle), ("Step forward", "Right", lambda: self.clock.step(1)),
                                  ("Step back", "Left", lambda: self.clock.step(-1))):
                act = QAction(text, self)
                act.setShortcut(QKeySequence(key))
                act.triggered.connect(fn)
                m.addAction(act)
            m = self.menuBar().addMenu("&Help")
            act = QAction("About / limits of this model", self)
            act.triggered.connect(self.about)
            m.addAction(act)

        # ---------------------------------------------------------------- running scenarios
        def start_simulation(self, cfg: SimulationConfig, synchronous: bool = False) -> None:
            """Compute a scenario (in a background thread so the window stays responsive) and show it when done."""
            self.cfg = cfg
            self.clock.pause()
            if synchronous:
                self.set_result(run_simulation(cfg))
                return
            if self._thread is not None:
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

        @Slot(object)
        def _finished(self, result) -> None:
            self._cleanup_thread()
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
            self._on_time(0.0)
            self.statusBar().showMessage(f"Scenario ready: {len(result.tracks)} satellites, {result.config.duration_h:g} h at "
                                         f"{result.config.step_s:g} s steps.  Space = play/pause, ←/→ = step.")

        def edit_scenario(self) -> None:
            dlg = ScenarioDialog(self.cfg, self)
            if dlg.exec():
                try:
                    cfg = dlg.config()
                    cfg.validate()
                except ValueError as exc:
                    QMessageBox.warning(self, "Scenario", str(exc))
                    return
                self.start_simulation(cfg)

        def export_csv(self) -> None:
            if self.result is None:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export link data", "satstk_link.csv", "CSV (*.csv)")
            if path:
                self.result.write_csv(path)
                self.statusBar().showMessage(f"Wrote {path}")

        def about(self) -> None:
            QMessageBox.information(self, "About satSTK", (
                f"satSTK {__version__}\n\nA standalone orbital mission and 1550 nm optical inter-satellite link simulator.\n"
                "Runs locally; no licence, no network, nothing is stored unless you export a CSV.\n\n"
                "Limits: J2-only orbit model (no drag, no solar pressure, no third-body gravity); Sun position from a short "
                "analytic formula (about 0.01°); Earth shadow ignores atmospheric refraction; link budget is a "
                "textbook calculation, not a waveform simulation; the radiator model ignores thermal mass. "
                "Agreement with Ansys STK or measured hardware has NOT been validated.\n\n"
                "Land outlines: Natural Earth (public domain)."))

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
        win.clock.seek(args.at_hours * 3600.0)
        win.show()
        app.processEvents()
        if not win.grab().save(args.screenshot):
            print(f"Could not write {args.screenshot} (does the folder exist?)", file=sys.stderr)
            return 1
        print(f"Saved {args.screenshot}")
        return 0
    win.show()
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
        win = MainWindow(config_from_args(parse_args(["--hours", "3"])))
        win.resize(1300, 800)
        win.start_simulation(win.cfg, synchronous=True)
        win.clock.seek(3600.0)
        win.show()
        app.processEvents()
        for name, widget in (("globe", win.globe), ("map", win.map), ("timeline", win.timeline), ("telemetry", win.telemetry)):
            img = widget.grab().toImage()
            arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4]
            colours = len(np.unique(arr.reshape(-1, 4), axis=0))
            lines.append(f"{name}: {colours} distinct colours")
            ok = ok and colours > 40
        ok = ok and len(geo.land_rings()) > 100
        link = win.result.primary_link
        lines.append(f"link windows: {len(win.result.access_windows[(0, 1)])}, locked now: {bool(link.visible[win.result.index_at(3600.0)])}")
        ok = ok and len(win.result.access_windows[(0, 1)]) > 0
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
