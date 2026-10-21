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
import math
import sys
from datetime import datetime, timezone

__version__ = "2.4.0"


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
    d = ap.add_argument_group("payload transmission (symbol-level optical DSP through the live simulation)")
    d.add_argument("--send-text", metavar="TEXT", help="transmit this text through the simulated optical link at --at-hours and print the result")
    d.add_argument("--send-file", metavar="FILE", help="transmit this file instead")
    d.add_argument("--send-test-payload", action="store_true", help="transmit the generated 63 KB self-checking test payload (with --screenshot: fire it into the dock first)")
    d.add_argument("--mod", default="Auto-Adaptive", help="BPSK, QPSK, 8PSK, 16QAM or Auto-Adaptive (default)")
    d.add_argument("--dsp-rate", type=float, default=10.0, help="DSP symbol rate in Gbaud (default 10; the budget's 250 Gbaud has no FEC here)")
    d.add_argument("--route", type=int, default=-1, help="force a route by candidate number instead of the active one (see the dock's list)")
    d.add_argument("--no-fsm", action="store_true", help="open-loop receiver: Fast Steering Mirror off")
    d.add_argument("--stream", action="store_true", help="stream the payload frame by frame over --stream-minutes of orbit instead of one shot")
    d.add_argument("--frame-bytes", type=int, default=1024, help="frame size for --stream")
    d.add_argument("--stream-minutes", type=float, default=120.0, help="simulated time the stream is spread over")
    t = ap.add_argument_group("digital twin: stream live telemetry to satCAD (UDP, loopback only, 30 Hz)")
    t.add_argument("--broadcast", action=argparse.BooleanOptionalAction, default=None,
                   help="stream telemetry to 127.0.0.1 for satCAD's digital twin. The window streams by default while it plays (--no-broadcast to stop it); "
                        "with --headless, --broadcast starts a real-time streaming run")
    t.add_argument("--broadcast-port", type=int, default=9876, help="UDP port to send telemetry to on 127.0.0.1 (default 9876)")
    t.add_argument("--status-port", type=int, default=9877, help="UDP port to listen on for satCAD's status datagrams (default 9877)")
    t.add_argument("--speed", type=float, default=None, help="playback speed (simulated seconds per real second); default 300 in the window, 60 headless")
    t.add_argument("--autoplay", action="store_true", help="start playing as soon as the scenario is computed (window)")
    t.add_argument("--no-placement", action="store_true", help="skip the 'place the beam origin' window and use the default station (for launchers)")
    t.add_argument("--wall-seconds", type=float, default=None, help="with --headless --broadcast: stop after this many real seconds (default: run until Ctrl-C)")
    r = ap.add_argument_group("radiation-tolerant AI scheduler (see radiation_pytorch_scheduler.py for its own command line)")
    r.add_argument("--show-radiation", action="store_true", help="with --screenshot: open the Radiation-Tolerant AI Scheduler tab after training up to --at-hours")
    r.add_argument("--rad-flare", action="store_true", help="with --show-radiation: a 1000x solar flare during the run")
    r.add_argument("--rad-minutes", type=float, default=90.0, help="with --show-radiation: minutes of mission time to train before the picture (default 90)")
    e = ap.add_argument_group("electrical power subsystem (see eps_power_engine.py for its own command line)")
    e.add_argument("--show-eps", action="store_true", help="with --screenshot: open the EPS Power, Battery & 48V Busbar tab")
    e.add_argument("--eps-wing-failure", action="store_true", help="with --show-eps: lose one of the 12 solar panels")
    e.add_argument("--eps-eclipse", action="store_true", help="with --show-eps: force every eclipse to last at least 45 minutes")
    e.add_argument("--eps-degradation", type=float, default=0.0, metavar="PCT", help="with --show-eps: battery capacity lost, percent")
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


def run_transmit(args) -> int:
    """Headless: run the scenario, then fire a payload through the route in force at --at-hours and print what came back."""
    from optical_dsp_bridge import OrbitalOpticalTransceiver, RouteUnavailable
    from dsp import payload as pl
    from simulation import run_simulation
    result = run_simulation(config_from_args(args))
    trx = OrbitalOpticalTransceiver(result)
    if args.send_file:
        data = open(args.send_file, "rb").read()
    elif args.send_text is not None:
        data = args.send_text.encode("utf-8")
    else:
        data = pl.generate_test_payload()
    idx = result.index_at((args.at_hours or 0.0) * 3600.0)
    forced = None if args.route < 0 else args.route
    rt = result.route
    print(f"Time T+{result.t_s[idx] / 3600:.3f} h; active route: {rt.label(idx)}; payload {len(data):,} bytes")
    try:
        if args.stream:
            s = trx.stream(data, idx, args.stream_minutes * 60.0, args.frame_bytes, args.mod, args.dsp_rate, closed_loop=not args.no_fsm)
            for r in s.frames:
                print(f"  frame {r.k + 1:3d} T+{r.time_s / 60:7.1f} min {r.route_label:42s} {r.modulation:6s} {'OK  ' if r.ok else 'LOST'}"
                      + ("  <handover" + (", hitless>" if r.hitless else ", NOT hitless>") if r.handover else ""))
            print(f"Stream: {s.frames_ok}/{len(s.frames)} frames delivered, {s.frame_loss_pct:.1f} % lost, {s.handovers} route changes "
                  f"({'all hitless' if s.all_handovers_hitless else 'some not hitless'}), whole-file CRC {'OK' if s.stream_crc_ok else 'MISMATCH'}")
            return 0 if s.frames_lost == 0 else 1
        out = trx.send(data, idx, args.mod, args.dsp_rate, closed_loop=not args.no_fsm, candidate=forced, seed=1)
    except RouteUnavailable as exc:
        print(f"Cannot transmit: {exc}")
        return 2
    print(f"Route {out.route_label}; modulation {out.modulation}" + (f" (auto: {out.adaptive_reason})" if out.adaptive_table else ""))
    print(f"CRC-32 {'PASS' if out.crc_ok else 'FAIL'}; BER {out.ber if out.ber is None else format(out.ber, '.3e')}; bytes wrong {out.bytes_wrong} ({out.bytes_wrong_pct:.2f} %); "
          f"frame loss {out.frame_loss_pct:.0f} %; status: {out.status}")
    print(f"Latency {out.latency_ms:.3f} ms (propagation {out.propagation_ms:.3f}, serialisation {out.serialization_ms:.4f}, processing {out.processing_ms:.3f}); "
          f"PAT lock {'-' if out.pat_ms is None else format(out.pat_ms, '.2f') + ' ms'}; Es/N0 predicted {out.esn0_predicted_db:.1f} dB")
    if out.blocks is not None:
        print(f"64-byte blocks bad: {out.blocks.n_bad} of {out.blocks.n_blocks}")
    return 0 if out.crc_ok else 1


def run_broadcast_headless(args) -> int:
    """Real-time telemetry without a window: step through the scenario at --speed and stream it to satCAD at 30 Hz until Ctrl-C (or --wall-seconds)."""
    import time
    from simulation import run_simulation
    from telemetry_broadcaster import DigitalTwinSource, TelemetryBroadcaster
    result = run_simulation(config_from_args(args))
    from eps_power_engine import simulate_eps
    source = DigitalTwinSource(result, eps_run=simulate_eps(result))              # thermal and eps_state come from the electrical-thermal closed loop
    speed = args.speed if args.speed else 60.0
    t_end = float(result.t_s[-1])
    t0 = (args.at_hours or 0.0) * 3600.0
    bc = TelemetryBroadcaster(port=args.broadcast_port, status_port=args.status_port)
    print(f"Streaming digital-twin telemetry to 127.0.0.1:{bc.port} at 30 Hz, simulated time x{speed:g} (Ctrl-C to stop).")
    if bc.listener_error:
        print(f"  note: {bc.listener_error}")
    wall0 = time.monotonic()
    next_print = wall0
    try:
        while True:
            now = time.monotonic()
            if args.wall_seconds is not None and now - wall0 >= args.wall_seconds:
                break
            t = (t0 + speed * (now - wall0)) % t_end if t_end > 0 else 0.0
            bc.tick(lambda seq, t=t: source.packet_at(t, seq), playing=True)
            if now >= next_print:
                pk = source.packet_at(t, bc.seq)
                print(f"  T+{t / 3600:6.3f} h  {pk['active_relay_id']:15s} core {pk['thermal_state']['core_temp_c']:5.1f} C  "
                      f"{pk['dsp_metrics']['modulation']:6s} {pk['dsp_metrics']['throughput_gbps']:7.1f} Gbps  sent {bc.sent}  {bc.status_text()}")
                next_print = now + 2.0
            time.sleep(0.004)
    except KeyboardInterrupt:
        pass
    finally:
        bc.close()
    print(f"Stopped after {bc.sent} packets ({bc.failed} not accepted by the operating system).")
    return 0


def build_window_classes():
    """Qt classes are created lazily so `--headless` and the physics tests never need a display or PySide6."""
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                                   QMessageBox, QProgressBar, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget)

    import orbital_engine as oe
    from optical_dsp_bridge import MAX_PAYLOAD_BYTES
    from pathlib import Path
    from simulation import SimulationConfig, run_simulation
    from ui import theme
    from ui.clock import SPEEDS, PlaybackClock
    from ui.conditions_panel import ConditionsPanel
    from ui.eps_panel import EPSPanel
    from ui.events_dialog import EventsDialog
    from ui.globe_view import GlobeView
    from ui.map_view import MapView
    from ui.payload_panel import PayloadPanel
    from ui.placement_dialog import PlacementDialog
    from ui.radiation_panel import RadiationPanel
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
        def __init__(self, cfg: SimulationConfig | None = None, broadcast: bool = False, broadcast_port: int = 9876, status_port: int = 9877, autoplay: bool = False):
            super().__init__()
            self._broadcast_wanted, self._bc_ports, self._autoplay = broadcast, (broadcast_port, status_port), autoplay
            self.broadcaster = None
            self.twin_source = None
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
            self.payload = PayloadPanel()
            self.radiation = RadiationPanel()
            self.eps = EPSPanel()

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
            self.payload_dock = self._dock("OPTICAL PAYLOAD TRANSMITTER  ·  fire real data", self.payload, Qt.DockWidgetArea.RightDockWidgetArea, 520)
            self.tabifyDockWidget(self.tele_dock, self.payload_dock)
            self.radiation_dock = self._dock("RADIATION-TOLERANT AI SCHEDULER  ·  32 blades", self.radiation, Qt.DockWidgetArea.RightDockWidgetArea, 600)
            self.tabifyDockWidget(self.payload_dock, self.radiation_dock)
            self.eps_dock = self._dock("EPS POWER, BATTERY && 48V BUSBAR", self.eps, Qt.DockWidgetArea.RightDockWidgetArea, 600)
            self.tabifyDockWidget(self.radiation_dock, self.eps_dock)
            self.tele_dock.raise_()
            self.cond_dock = self._dock("CONDITIONS  ·  tinker here", self.conditions, Qt.DockWidgetArea.LeftDockWidgetArea, 400)

            self._build_menus()
            self.progress = QProgressBar()
            self.progress.setMaximumWidth(220)
            self.progress.setRange(0, 100)
            self.progress.hide()
            self.statusBar().addPermanentWidget(self.progress)
            self.twin_label = QLabel("Digital twin: off")
            self.statusBar().addPermanentWidget(self.twin_label)
            self._twin_label_timer = QTimer(self)
            self._twin_label_timer.setInterval(1000)
            self._twin_label_timer.timeout.connect(self._update_twin_label)

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
            if self._broadcast_wanted:
                self.set_broadcast(True)
            self.map.station_move_requested.connect(self._station_clicked)
            self.radiation.overlay_changed.connect(self._radiation_overlay)
            self.eps.run_changed.connect(self._apply_eps)
            self.radiation.downlink_requested.connect(self.downlink_weights)

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
            m.addAction(self.payload_dock.toggleViewAction())
            m.addAction(self.radiation_dock.toggleViewAction())
            m.addAction(self.eps_dock.toggleViewAction())
            m = self.menuBar().addMenu("&Playback")
            for text, key, fn in (("Play / Pause", "Space", self.clock.toggle), ("Step forward", "Right", lambda: self.clock.step(1)),
                                  ("Step back", "Left", lambda: self.clock.step(-1)), ("Next pass", "N", self.next_pass)):
                act = QAction(text, self)
                act.setShortcut(QKeySequence(key))
                act.triggered.connect(fn)
                m.addAction(act)
            m.addSeparator()
            self.broadcast_action = QAction("Stream live telemetry to satCAD (digital twin)", self)
            self.broadcast_action.setCheckable(True)
            self.broadcast_action.setChecked(self._broadcast_wanted)
            self.broadcast_action.toggled.connect(self.set_broadcast)
            m.addAction(self.broadcast_action)
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
            self.payload.set_result(result)
            self.eps.set_result(result)                        # runs the electrical-thermal loop; _apply_eps then builds the twin source and restarts the AI scheduler
            self.clock.configure(result.t_s[-1], result.config.step_s)
            t0 = self._keep_time if self._keep_time is not None else self._first_pass_time()
            self._keep_time = None
            self.clock.seek(min(t0, result.t_s[-1]))
            self._on_time(self.clock.t)
            if self._autoplay:
                self._autoplay = False
                self.clock.play()
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
            self.payload.set_time(t)
            self.radiation.set_time(t)
            self.eps.set_time(t)
            self.time_label.setText(f"{self.result.utc(t):%Y-%m-%d %H:%M:%S} UTC    T+ {t / 3600:6.3f} h")

        # ---------------------------------------------------------------- electrical power subsystem
        def _apply_eps(self, run) -> None:
            """A new EPS run (a new scenario or a stress test): the digital twin streams its thermal and electrical state and the AI scheduler follows its breaker and DVFS."""
            if self.result is None:
                return
            try:
                from telemetry_broadcaster import DigitalTwinSource
                self.twin_source = DigitalTwinSource(self.result, eps_run=run)
            except Exception as exc:                           # the twin feed is optional: never let it break the simulation view
                self.twin_source = None
                self.twin_label.setText(f"Digital twin: cannot build telemetry ({exc})")
            self.radiation.set_result(self.result, run.thermal, run.history)

        # ---------------------------------------------------------------- radiation-tolerant AI scheduler
        def _radiation_overlay(self, on: bool, flux: float) -> None:
            self.globe.set_radiation(on, flux)
            self.map.set_radiation(on, flux)

        def downlink_weights(self, path: str) -> None:
            """Send an exported weights file through the existing 16QAM optical payload link (the payload dock does the transmission)."""
            data = Path(path).read_bytes()
            if len(data) > MAX_PAYLOAD_BYTES:
                self.statusBar().showMessage(f"{Path(path).name} is {len(data):,} bytes; the optical payload limit is {MAX_PAYLOAD_BYTES:,}")
                return
            self.payload._set_file(data, Path(path).name)
            self.payload.mod.setCurrentText("16QAM")
            self.payload_dock.show()
            self.payload_dock.raise_()
            self.statusBar().showMessage(f"{Path(path).name} ({len(data):,} bytes) loaded into the optical payload dock as 16QAM; firing the laser downlink")
            if self.payload.fire.isEnabled():
                self.payload.fire.click()

        # ---------------------------------------------------------------- digital twin telemetry
        def set_broadcast(self, on: bool) -> None:
            """Start or stop streaming telemetry to satCAD (127.0.0.1 only). Never raises: a problem is shown in the status bar."""
            if on and self.broadcaster is None:
                try:
                    from telemetry_broadcaster import TelemetryBroadcaster
                    self.broadcaster = TelemetryBroadcaster(port=self._bc_ports[0], status_port=self._bc_ports[1])
                except Exception as exc:
                    self.broadcaster = None
                    self.twin_label.setText(f"Digital twin: unavailable ({exc})")
                    self.twin_label.setStyleSheet(f"color: {theme.RED};")
                    return
                self._old_switch_interval = sys.getswitchinterval()
                sys.setswitchinterval(0.0005)                  # let the feed thread get the interpreter within 0.5 ms even while the window is busy painting
                self.broadcaster.start_feed(self._twin_packet, lambda: self.clock.playing)      # its own thread: the 30 Hz wall-clock rate holds while the window repaints
                self._twin_label_timer.start()
            elif not on and self.broadcaster is not None:
                sys.setswitchinterval(getattr(self, "_old_switch_interval", 0.005))
                self._twin_label_timer.stop()
                self.broadcaster.close()
                self.broadcaster = None
            if hasattr(self, "broadcast_action") and self.broadcast_action.isChecked() != on and (on == (self.broadcaster is not None)):
                self.broadcast_action.setChecked(on)
            self._update_twin_label()

        def _twin_packet(self, seq: int):
            """Called on the feed thread: only reads (the simulation arrays and the clock's current time), never touches a widget."""
            src = self.twin_source
            return None if src is None else src.packet_at(self.clock.t, seq)

        def _update_twin_label(self) -> None:
            bc = self.broadcaster
            if bc is None:
                self.twin_label.setText("Digital twin: off")
                self.twin_label.setStyleSheet(f"color: {theme.DIM};")
            else:
                st = bc.status
                self.twin_label.setText(bc.status_text())
                self.twin_label.setStyleSheet(f"color: {theme.GREEN if st.connected else theme.AMBER};")

        def set_speed_value(self, speed: float) -> None:
            i = self.speed.findData(float(speed))
            if i < 0:
                self.speed.addItem(f"{speed:g}x", float(speed))
                i = self.speed.count() - 1
            self.speed.setCurrentIndex(i)

        def closeEvent(self, ev) -> None:
            self.set_broadcast(False)
            super().closeEvent(ev)

        def _on_state(self, playing: bool) -> None:
            self.btn_play.setText("⏸ Pause" if playing else "▶ Play")

    return QApplication, MainWindow, theme


def run_gui(args) -> int:
    QApplication, MainWindow, theme = build_window_classes()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("satSTK")
    theme.apply_dark_theme(app)
    stream = args.broadcast is not False and not args.screenshot and not args.selftest
    win = MainWindow(config_from_args(args), broadcast=stream, broadcast_port=args.broadcast_port, status_port=args.status_port,
                     autoplay=args.autoplay and not args.screenshot)
    if args.speed:
        win.set_speed_value(args.speed)
    if args.screenshot:
        w, h = (int(v) for v in args.size.lower().split("x"))
        win.resize(w, h)
        win.start_simulation(win.cfg, synchronous=True)
        if args.at_hours is not None:
            win.clock.seek(args.at_hours * 3600.0)
        if args.show_eps:
            win.eps.wing_btn.setChecked(args.eps_wing_failure)
            win.eps.eclipse_btn.setChecked(args.eps_eclipse)
            win.eps.deg_slider.setValue(int(args.eps_degradation))
            win.eps.rerun()
            win.show()
            win.eps_dock.raise_()
        if args.show_radiation:
            t_end = (args.at_hours if args.at_hours is not None else 1.0) * 3600.0
            t = max(0.0, t_end - args.rad_minutes * 60.0)
            win.clock.seek(t)
            win.radiation.flare_btn.setChecked(bool(args.rad_flare))
            while t < t_end:
                t = min(t + 80.0, t_end)
                win.radiation.set_time(t)
            win.clock.seek(t_end)
            win.show()
            win.radiation_dock.raise_()
        if args.send_test_payload:
            from dsp import payload as pl
            data = pl.generate_test_payload()
            win.payload._set_file(data, "test_payload_63kb.txt (generated)")
            out = win.payload.trx.send(data, win.result.index_at(win.clock.t), args.mod, args.dsp_rate, closed_loop=not args.no_fsm, seed=1)
            win.show()
            win.payload.show_transmission(out)
            win.payload_dock.raise_()
        win.show()
        app.processEvents()
        if args.send_test_payload:
            win.payload.reveal_results()
            app.processEvents()
        if not win.grab().save(args.screenshot):
            print(f"Could not write {args.screenshot} (does the folder exist?)", file=sys.stderr)
            return 1
        print(f"Saved {args.screenshot}")
        return 0
    win.show()
    app.processEvents()
    if args.lat is None and args.lon is None and not args.no_placement:
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
        n_routes = win.payload.route_combo.count() - 1                    # the payload dock lists every candidate route (a dock tab behind another is not painted on every platform, so it is not pixel-checked)
        lines.append(f"payload dock: {n_routes} routes listed")
        ok = ok and n_routes == len(res.route.labels) > 1
        from optical_dsp_bridge import OrbitalOpticalTransceiver
        from dsp import payload as pl
        trx = OrbitalOpticalTransceiver(res)
        i3 = int(np.argmax(res.route.kind == 3)) if (res.route.kind == 3).any() else res.index_at(win.clock.t)
        probe = trx.send(b"selftest payload through the optical DSP chain " * 4, i3, "Auto-Adaptive", 10.0, seed=1)
        lines.append(f"optical DSP: {probe.route_label} as {probe.modulation}: CRC {'OK' if probe.crc_ok else 'FAIL'}, bundled generator makes {len(pl.generate_test_payload()):,} bytes")
        ok = ok and probe.crc_ok and len(pl.generate_test_payload()) == 64512
        eh = win.eps.run.history
        eps_ok = len(eh.mode) == res.n and float(eh.soc.min()) > 0.2 and eh.v_min.min() >= 45.5 and win.twin_source is not None and "eps_state" in win.twin_source.packet_at(win.clock.t, 0)
        lines.append(f"EPS: peak solar {eh.solar_w.max() / 1000:.1f} kW, lowest SoC {eh.soc.min() * 100:.1f} %, worst blade {eh.v_min.min():.3f} V, eps_state in the twin packet: {eps_ok}")
        ok = ok and eps_ok
        for k in range(1, 4):
            win.radiation.set_time(float(win.clock.t) + 400.0 * k)
        rr = win.radiation.orch.summary()
        lines.append(f"radiation scheduler ({rr['engine']} engine): {rr['steps']} training steps, {rr['protected_totals']['hits']} simulated bit flips, protected loss "
                     f"{win.radiation.orch.history['loss_protected'][-1]:.3f}")
        ok = ok and rr["steps"] >= 8 and math.isfinite(win.radiation.orch.history["loss_protected"][-1])
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
    if args.send_text is not None or args.send_file or (args.send_test_payload and not args.screenshot):
        return run_transmit(args)
    if args.headless and args.broadcast:
        return run_broadcast_headless(args)
    return run_headless(args) if args.headless else run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
