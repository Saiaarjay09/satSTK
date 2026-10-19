"""
Live Optical Payload Transmitter: type text or load a file, pick a route and a moment in the orbit, and fire it through the symbol-level DSP chain
(modulator -> every hop of the route -> PAT / Fast Steering Mirror -> carrier recovery -> CRC-32). Results: constellation, FSM trajectory, a
side-by-side diff of what was sent and what came back, and the numbers. A streaming mode sends a big file frame by frame while the satellite moves.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QScrollArea, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

import optical_dsp_bridge as ob
from dsp import payload as pl
from dsp import transmitter as dtx

from . import theme

MAX_SHOWN_BYTES = 4096
TEST_PAYLOAD_NAME = "test_payload_63kb.txt"


class PayloadWorker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = False

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(lambda m, f: self.progress.emit(m, f), lambda: self.cancel))
        except Exception as exc:                       # shown in the panel instead of crashing the app
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def printable(b: int) -> str:
    return chr(b) if 32 <= b < 127 or b == 10 else "·"


class DiffView(QPlainTextEdit):
    """Read-only view of the first bytes of the data, with every wrong byte highlighted."""

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(theme.mono_font(9))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    def show_bytes(self, data: bytes | None, wrong: np.ndarray | None, hexdump: bool, limit: int = MAX_SHOWN_BYTES) -> None:
        self.clear()
        if data is None:
            self.setPlainText("(nothing was recovered)")
            return
        cur = self.textCursor()
        bad, good = QTextCharFormat(), QTextCharFormat()
        bad.setBackground(QColor("#7a1f24"))
        bad.setForeground(QColor("#ffd6d8"))
        good.setForeground(QColor(theme.TEXT))
        data = data[:limit]
        runs: list[tuple[str, bool]] = []
        n = len(data)
        flags = [bool(wrong[i]) if wrong is not None and i < len(wrong) else False for i in range(n)]
        if hexdump:
            for i in range(0, n, 16):
                for j in range(i, min(i + 16, n)):
                    runs.append((f"{data[j]:02x} ", flags[j]))
                runs.append(("\n", False))
        else:
            for i in range(n):
                runs.append((printable(data[i]), flags[i]))
        text, cur_flag = "", None
        for ch, f in runs:
            if f != cur_flag and text:
                cur.insertText(text, bad if cur_flag else good)
                text = ""
            cur_flag = f
            text += ch
        if text:
            cur.insertText(text, bad if cur_flag else good)
        self.moveCursor(QTextCursor.MoveOperation.Start)


class PayloadPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.trx = None
        self.t = 0.0
        self._file_bytes: bytes | None = None
        self._file_name = ""
        self._thread: QThread | None = None
        self._worker: PayloadWorker | None = None
        self.last = None
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        intro = QLabel("Fire a real payload through the live simulation: the route, ranges, atmosphere, events and pointing jitter come from the "
                       "orbit at the clock time shown below. Move the clock (or pick a route) to change the conditions.")
        intro.setWordWrap(True)
        intro.setProperty("role", "dim")
        root.addWidget(intro)

        # ---- where in the orbit ----------------------------------------------------------------------------------------------
        g = self._group("ROUTE AND MOMENT")
        f = QFormLayout(g)
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.when = QLabel("—")
        self.when.setFont(theme.mono_font(10))
        f.addRow("Clock time", self.when)
        self.route_combo = QComboBox()
        self.route_combo.addItem("Active route at the clock time", -1)
        f.addRow("Route to use", self.route_combo)
        self.route_note = QLabel("")
        self.route_note.setWordWrap(True)
        f.addRow(self.route_note)
        root.addWidget(g)

        # ---- payload ----------------------------------------------------------------------------------------------------------
        g = self._group("PAYLOAD")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.rb_text, self.rb_file = QRadioButton("Text"), QRadioButton("File")
        self.rb_text.setChecked(True)
        row.addWidget(self.rb_text)
        row.addWidget(self.rb_file)
        row.addStretch(1)
        v.addLayout(row)
        self.text = QPlainTextEdit("Hello from GS-INDIA! This message crossed the real orbital geometry, atmosphere and relay chain of satSTK.")
        self.text.setFixedHeight(64)
        v.addWidget(self.text)
        row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse file ...")
        self.gen_btn = QPushButton("Generate 63 KB Test Payload")
        self.gen_btn.setToolTip("1,008 self-checking 64-byte lines (63 KB). Every line carries its own CRC-32, so damaged blocks can be located.")
        self.save_btn = QPushButton("Save payload ...")
        self.save_btn.setEnabled(False)
        row.addWidget(self.browse_btn)
        row.addWidget(self.gen_btn)
        row.addWidget(self.save_btn)
        v.addLayout(row)
        self.file_label = QLabel("No file chosen")
        self.file_label.setProperty("role", "dim")
        self.file_label.setWordWrap(True)
        v.addWidget(self.file_label)
        root.addWidget(g)

        # ---- hardware ---------------------------------------------------------------------------------------------------------
        g = self._group("MODULATION AND HARDWARE")
        f = QFormLayout(g)
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.mod = QComboBox()
        self.mod.addItems(ob.MODULATIONS + [ob.AUTO])
        self.mod.setCurrentText(ob.AUTO)
        self.mod.setToolTip("Auto-Adaptive picks the densest scheme that should still pass the CRC, per route and per frame.")
        f.addRow("Modulation", self.mod)
        self.rate = self._spin(f, "Symbol rate (Gbaud)", 10.0, 0.1, 500.0, 1, " Gbaud",
                               "The simulation's link budget assumes 250 Gbaud with FEC. This DSP has no FEC (every bit must be right for the CRC), "
                               "so ~10 Gbaud is a good starting point for the GEO paths.")
        self.tx_power = self._spin(f, "Transmit laser power (W)", 10.0, 0.01, 200.0, 2, " W")
        self.beacon = self._spin(f, "Beacon power (W)", 1.0, 0.001, 100.0, 3, " W")
        self.fsm = QCheckBox("FSM closed-loop ON  (off = open loop: no tracking)")
        self.fsm.setChecked(True)
        f.addRow(self.fsm)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setSpecialValueText("random")
        f.addRow("Random seed (0 = new each time)", self.seed)
        root.addWidget(g)

        # ---- streaming --------------------------------------------------------------------------------------------------------
        g = self._group("CONTINUOUS FILE STREAM")
        f = QFormLayout(g)
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.stream_on = QCheckBox("Stream the payload frame by frame while the satellite moves")
        f.addRow(self.stream_on)
        self.frame_bytes = QSpinBox()
        self.frame_bytes.setRange(64, 16384)
        self.frame_bytes.setValue(1024)
        self.frame_bytes.setSuffix(" bytes / frame")
        f.addRow("Frame size", self.frame_bytes)
        self.stream_min = self._spin(f, "Spread over (simulated minutes)", 120.0, 1.0, 1440.0, 0, " min",
                                     "Frame k leaves at start + (k + 0.5)/n x this duration, using whatever route is in force at that moment. 120 min covers several handovers.")
        root.addWidget(g)

        # ---- fire --------------------------------------------------------------------------------------------------------------
        row = QHBoxLayout()
        self.fire = QPushButton("🔥 Fire Laser Payload")
        self.fire.setStyleSheet(f"QPushButton {{ background: #7a2b12; border: 1px solid #ff7a45; font-weight: bold; padding: 8px 16px; }} QPushButton:hover {{ background: #9a3a1a; }}"
                                f"QPushButton:disabled {{ background: {theme.PANEL_2}; color: {theme.DIM}; border-color: {theme.BORDER}; }}")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        row.addWidget(self.fire, 1)
        row.addWidget(self.cancel_btn)
        root.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        root.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        # ---- results ------------------------------------------------------------------------------------------------------------
        root.addWidget(self._section("RESULT"))
        self.badge = QLabel("—")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFont(theme.mono_font(14))
        self.badge.setStyleSheet(f"background: {theme.PANEL_2}; border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 6px;")
        root.addWidget(self.badge)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(1)
        self.m = {}
        labels = [("route", "Route"), ("mod", "Modulation"), ("ber", "BER (receiver, vs sent bits)"), ("ber_o", "BER (oracle, single hop)"), ("bytes", "Bytes sent / recovered"),
                  ("wrong", "Bytes wrong"), ("blocks", "64-byte blocks bad"), ("floss", "Frame loss"), ("lat", "End-to-end latency"), ("lat_detail", "…propagation / serialisation / processing"),
                  ("pat", "PAT time to lock"), ("snr", "Es/N0 predicted / measured"), ("rx", "Received power (last hop)"), ("hops", "Hops")]
        for r, (key, name) in enumerate(labels):
            lab = QLabel(name)
            lab.setProperty("role", "dim")
            grid.addWidget(lab, r, 0)
            val = QLabel("—")
            val.setFont(theme.mono_font(10))
            val.setWordWrap(True)
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(val, r, 1)
            self.m[key] = val

        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(430)
        # constellation
        self.iq = pg.PlotWidget()
        self._style_plot(self.iq, "IQ constellation (carrier-tracked received symbols)")
        self.iq.setAspectLocked(True)
        self.iq.setLabel("bottom", "In-phase")
        self.iq.setLabel("left", "Quadrature")
        self.tabs.addTab(self.iq, "Constellation")
        # FSM
        fsm_box = QWidget()
        fl = QVBoxLayout(fsm_box)
        fl.setContentsMargins(0, 0, 0, 0)
        self.fsm_plot = pg.PlotWidget()
        self._style_plot(self.fsm_plot, "Quadrant detector: where the beam sits on the sensor (µrad)")
        self.fsm_plot.setAspectLocked(True)
        self.fsm_plot.setLabel("bottom", "x error (µrad)")
        self.fsm_plot.setLabel("left", "y error (µrad)")
        fl.addWidget(self.fsm_plot, 1)
        self.pat_log = QPlainTextEdit()
        self.pat_log.setReadOnly(True)
        self.pat_log.setFont(theme.mono_font(8))
        self.pat_log.setMaximumHeight(110)
        fl.addWidget(self.pat_log)
        self.tabs.addTab(fsm_box, "FSM / PAT")
        # diff
        diff_box = QWidget()
        dl = QVBoxLayout(diff_box)
        dl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.hex_check = QCheckBox("Hex view")
        row.addWidget(self.hex_check)
        row.addStretch(1)
        row.addWidget(QLabel(f"first {MAX_SHOWN_BYTES:,} bytes; wrong bytes in red"))
        dl.addLayout(row)
        cols = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(QLabel("Transmitted"))
        right.addWidget(QLabel("Recovered"))
        self.diff_tx, self.diff_rx = DiffView(), DiffView()
        left.addWidget(self.diff_tx)
        right.addWidget(self.diff_rx)
        cols.addLayout(left)
        cols.addLayout(right)
        dl.addLayout(cols)
        self.tabs.addTab(diff_box, "Payload diff")
        # stream log
        stream_box = QWidget()
        sl = QVBoxLayout(stream_box)
        sl.setContentsMargins(0, 0, 0, 0)
        self.stream_summary = QLabel("Run with the stream option on to fill this tab.")
        self.stream_summary.setWordWrap(True)
        sl.addWidget(self.stream_summary)
        self.stream_plot = pg.PlotWidget()
        self._style_plot(self.stream_plot, "Frames over the orbit: modulation (bits/symbol), lost frames, handovers")
        self.stream_plot.setLabel("bottom", "simulation time (min)")
        self.stream_plot.setLabel("left", "bits / symbol")
        self.stream_plot.setFixedHeight(180)
        sl.addWidget(self.stream_plot)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "T+ (min)", "Route", "Mod", "BER", "Result", "Note"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        sl.addWidget(self.table, 1)
        self.tabs.addTab(stream_box, "Stream log")
        root.addWidget(self.tabs)                                    # plots right under the badge ...
        root.addWidget(self._section("DETAILS"))
        root.addLayout(grid)                                         # ... numbers below them
        root.addStretch(1)

        self.browse_btn.clicked.connect(self._browse)
        self.gen_btn.clicked.connect(self._generate)
        self.save_btn.clicked.connect(self._save)
        self.fire.clicked.connect(self._fire)
        self.cancel_btn.clicked.connect(self._cancel)
        self.hex_check.toggled.connect(self._redraw_diff)
        self.route_combo.currentIndexChanged.connect(lambda *_: self._refresh_route_note())
        self._diff_data = None

    # ------------------------------------------------------------------ widget helpers
    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #35b6ff; border: 1px solid #263241; border-radius: 4px; margin-top: 8px; padding-top: 6px; }"
                        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")
        return g

    @staticmethod
    def _section(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "section")
        return lab

    def _spin(self, form, label, value, lo, hi, decimals, suffix, tip="") -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(value)
        sp.setSuffix(suffix)
        sp.setAlignment(Qt.AlignmentFlag.AlignRight)
        if tip:
            sp.setToolTip(tip)
        form.addRow(label, sp)
        return sp

    @staticmethod
    def _style_plot(pw: pg.PlotWidget, title: str) -> None:
        pw.setTitle(title, size="9pt", color=theme.TEXT)
        pw.showGrid(x=True, y=True, alpha=0.18)
        pw.setMenuEnabled(False)

    # ------------------------------------------------------------------ simulation state in
    def set_result(self, result) -> None:
        self.result = result
        self.trx = ob.OrbitalOpticalTransceiver(result)
        self.tx_power.setValue(result.config.link.tx_power_w)
        self.route_combo.blockSignals(True)
        keep = self.route_combo.currentData()
        self.route_combo.clear()
        self.route_combo.addItem("Active route at the clock time", -1)
        for k, label in enumerate(result.route.labels):
            self.route_combo.addItem(label, k)
        i = self.route_combo.findData(keep)
        self.route_combo.setCurrentIndex(max(i, 0))
        self.route_combo.blockSignals(False)
        self.set_time(self.t)

    def set_time(self, t: float) -> None:
        self.t = t
        if self.result is None:
            return
        i = self.result.index_at(t)
        self.when.setText(f"{self.result.utc(t):%Y-%m-%d %H:%M:%S} UTC   (T+ {t / 3600:.3f} h)")
        for k in range(1, self.route_combo.count()):
            cand = self.route_combo.itemData(k)
            up = bool(self.result.route.available[cand, i])
            self.route_combo.setItemText(k, f"{'✓' if up else '✗'} {self.result.route.labels[cand]}")
        self._refresh_route_note()

    def _refresh_route_note(self) -> None:
        if self.result is None:
            return
        i = self.result.index_at(self.t)
        rt = self.result.route
        forced = self.route_combo.currentData()
        if forced == -1:
            self.route_note.setText(f"Active route now: {rt.label(i)}")
            self.route_note.setStyleSheet(f"color: {theme.GREEN if rt.chosen[i] >= 0 else theme.RED};")
        else:
            up = bool(rt.available[forced, i])
            self.route_note.setText(f"{rt.labels[forced]}: " + ("has line of sight right now" if up else "NO line of sight at this moment - pick another time or route"))
            self.route_note.setStyleSheet(f"color: {theme.GREEN if up else theme.RED};")

    # ------------------------------------------------------------------ payload sources
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a payload file")
        if not path:
            return
        data = Path(path).read_bytes()
        if len(data) > ob.MAX_PAYLOAD_BYTES:
            QMessageBox.warning(self, "Payload", f"That file is {len(data):,} bytes; the limit is {ob.MAX_PAYLOAD_BYTES:,} bytes.")
            return
        self._set_file(data, Path(path).name)

    def _generate(self) -> None:
        self._set_file(pl.generate_test_payload(), TEST_PAYLOAD_NAME + " (generated)")

    def _set_file(self, data: bytes, name: str) -> None:
        self._file_bytes, self._file_name = data, name
        self.rb_file.setChecked(True)
        self.save_btn.setEnabled(True)
        self.file_label.setText(f"{name}: {len(data):,} bytes ({len(data) // 64} blocks of 64 B)" if len(data) % 64 == 0 else f"{name}: {len(data):,} bytes")

    def _save(self) -> None:
        if self._file_bytes is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save payload", TEST_PAYLOAD_NAME)
        if path:
            Path(path).write_bytes(self._file_bytes)
            self.status.setText(f"Saved {path}")

    def current_payload(self) -> tuple[bytes, str]:
        if self.rb_file.isChecked():
            if self._file_bytes is None:
                raise ValueError("choose a file or press 'Generate 63 KB Test Payload' first")
            return self._file_bytes, self._file_name
        text = self.text.toPlainText()
        if not text:
            raise ValueError("the text payload is empty")
        return text.encode("utf-8"), "typed text"

    # ------------------------------------------------------------------ firing
    def _fire(self) -> None:
        if self.result is None or self._thread is not None:
            return
        try:
            data, name = self.current_payload()
        except ValueError as exc:
            QMessageBox.warning(self, "Payload", str(exc))
            return
        trx, idx = self.trx, self.result.index_at(self.t)
        forced = self.route_combo.currentData()
        forced = None if forced == -1 else int(forced)
        args = dict(modulation=self.mod.currentText(), symbol_rate_gbaud=self.rate.value(), tx_power_w=self.tx_power.value(), beacon_power_w=self.beacon.value(),
                    closed_loop=self.fsm.isChecked(), seed=self.seed.value() or None)
        streaming = self.stream_on.isChecked()
        if streaming:
            frame, span = self.frame_bytes.value(), self.stream_min.value() * 60.0

            def job(progress, cancelled):
                return trx.stream(data, idx, span, frame, progress=progress, should_cancel=cancelled, **args)
        else:
            def job(progress, cancelled):
                return trx.send(data, idx, candidate=forced, progress=progress, **args)
        self._streaming, self._payload_name = streaming, name
        self._start_worker(job)

    def _start_worker(self, job) -> None:
        self.fire.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setStyleSheet(f"color: {theme.DIM};")
        self.status.setText("Starting ...")
        self._thread = QThread(self)
        self._worker = PayloadWorker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)             # bound methods: delivered on the GUI thread
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel = True
            self.status.setText("Cancelling after the current frame ...")

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
        self.fire.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.hide()

    @Slot(str, float)
    def _on_progress(self, msg: str, frac: float) -> None:
        self.status.setText(msg)
        self.progress.setValue(int(100 * frac))

    @Slot(str)
    def _on_failed(self, msg: str) -> None:
        self._stop_thread()
        self.status.setStyleSheet(f"color: {theme.RED};")
        self.status.setText(msg)
        self._set_badge(None, msg)

    @Slot(object)
    def _on_finished(self, out) -> None:
        self._stop_thread()
        self.status.setStyleSheet(f"color: {theme.DIM};")
        if isinstance(out, ob.StreamResult):
            self.show_stream(out)
        else:
            self.show_transmission(out)
        QTimer.singleShot(0, self.reveal_results)

    # ------------------------------------------------------------------ showing results
    def reveal_results(self) -> None:
        """Scroll the dock (if it sits in a scroll area) so the result badge and the plots are in view."""
        w = self.parentWidget()
        while w is not None and not isinstance(w, QScrollArea):
            w = w.parentWidget()
        if w is not None:
            w.ensureWidgetVisible(self.badge, 0, 20)
            bar = w.verticalScrollBar()
            bar.setValue(min(bar.maximum(), max(bar.value(), self.badge.mapTo(self, self.badge.rect().topLeft()).y() - 10)))

    def _set_badge(self, ok, text: str) -> None:
        color = theme.DIM if ok is None else (theme.GREEN if ok else theme.RED)
        self.badge.setText(text)
        self.badge.setStyleSheet(f"background: {theme.PANEL_2}; color: {color}; border: 1px solid {color}; border-radius: 4px; padding: 6px; font-weight: bold;")

    def show_transmission(self, out: ob.TransmissionResult) -> None:
        self.last = out
        self._set_badge(out.crc_ok, "CRC-32  PASS  ✔  payload delivered intact" if out.crc_ok else "CRC-32  FAIL  ✘  " + (out.status if len(out.status) < 70 else out.status[:67] + "..."))
        m = self.m
        m["route"].setText(out.route_label)
        m["mod"].setText(out.modulation + ("  (auto: " + out.adaptive_reason + ")" if out.adaptive_table is not None else ""))
        m["ber"].setText("—" if out.ber is None else (f"{out.ber:.3e}  ({out.n_bit_errors} / {out.n_bits} bits)" if out.n_bit_errors else f"0  (< {1 / out.n_bits:.1e}; 0 / {out.n_bits} bits)"))
        m["ber_o"].setText("n/a (several hops)" if out.ber_oracle is None else (f"{out.ber_oracle:.3e}" if out.ber_oracle else "0"))
        m["bytes"].setText(f"{len(out.payload):,} / {0 if out.recovered is None else len(out.recovered):,}")
        m["wrong"].setText(f"{out.bytes_wrong:,}  ({out.bytes_wrong_pct:.2f} %)")
        m["blocks"].setText("—" if out.blocks is None else f"{out.blocks.n_bad} of {out.blocks.n_blocks}" + (f"  (first: {out.blocks.bad_blocks[:6]})" if out.blocks.n_bad else ""))
        m["floss"].setText(f"{out.frame_loss_pct:.0f} %  (1 frame)")
        m["lat"].setText(f"{out.latency_ms:.3f} ms")
        m["lat_detail"].setText(f"{out.propagation_ms:.3f} / {out.serialization_ms:.4f} / {out.processing_ms:.3f} ms")
        m["pat"].setText("did not lock" if out.pat_ms is None and not out.pre_locked and out.closed_loop else ("open loop (FSM off)" if not out.closed_loop else
                         ("already tracking" if out.pre_locked else f"{out.pat_ms:.2f} ms")))
        m["snr"].setText(f"{out.esn0_predicted_db:.1f} dB / " + ("—" if out.esn0_est_db is None else f"{out.esn0_est_db:.1f} dB"))
        m["rx"].setText(f"{out.rx_power_dbm:.2f} dBm" + ("   (BEAM BLOCKED BY AN EVENT)" if out.blocked_by_event else ""))
        m["hops"].setText("   ".join(f"{h['src']}→{h['dst']} {h['range_km']:,.0f} km" + (f" (el {h['elevation_deg']:.0f}°, +{h['extra_loss_db']:.1f} dB air)" if h["elevation_deg"] is not None else "")
                                    for h in out.hop_summaries))
        self._draw_constellation(out)
        self._draw_fsm(out)
        self._diff_data = (out.payload, out.recovered)
        self._redraw_diff()
        self.status.setText(f"Done at T+{out.time_s / 60:.1f} min over {out.route_label}")

    def _draw_constellation(self, out: ob.TransmissionResult) -> None:
        self.iq.clear()
        pts = dtx.constellation(out.modulation)
        if out.rx_symbols is not None and len(out.rx_symbols):
            y = out.rx_symbols
            step = max(1, len(y) // 5000)
            y = y[::step]
            self.iq.addItem(pg.ScatterPlotItem(y.real, y.imag, size=3, pen=None, brush=pg.mkBrush(0, 229, 255, 90)))
        self.iq.addItem(pg.ScatterPlotItem(pts.real, pts.imag, size=11, symbol="+", pen=pg.mkPen("w", width=1.6), brush=None))
        lim = 1.8 if out.modulation != "16QAM" else 1.6
        self.iq.setRange(xRange=(-lim, lim), yRange=(-lim, lim))
        self.iq.setTitle(f"IQ constellation: {out.modulation}, {out.route_label}"
                         + ("" if out.esn0_est_db is None else f"  (measured Es/N0 {out.esn0_est_db:.1f} dB)"), size="9pt", color=theme.TEXT)

    def _draw_fsm(self, out: ob.TransmissionResult) -> None:
        self.fsm_plot.clear()
        det = 150.0
        box = pg.PlotCurveItem([-det, det, det, -det, -det], [-det, -det, det, det, -det], pen=pg.mkPen(theme.DIM, style=Qt.PenStyle.DashLine))
        self.fsm_plot.addItem(box)
        self.fsm_plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(theme.BORDER)))
        self.fsm_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.BORDER)))
        tr = out.pat_trace
        if tr is not None and len(tr["err_urad"]):
            e, st = tr["err_urad"], tr["state"]
            step = max(1, len(e) // 3000)
            e, st = e[::step], st[::step]
            if np.any(st == 0):
                s = e[st == 0]
                self.fsm_plot.addItem(pg.PlotCurveItem(s[:, 0], s[:, 1], pen=pg.mkPen(theme.AMBER, width=1.2)))
            t = e[st == 1] if np.any(st == 1) else e
            self.fsm_plot.addItem(pg.PlotCurveItem(t[:, 0], t[:, 1], pen=pg.mkPen(theme.GREEN, width=1.6)))
            self.fsm_plot.addItem(pg.ScatterPlotItem([e[0, 0]], [e[0, 1]], size=9, symbol="o", brush=pg.mkBrush(theme.AMBER), pen=pg.mkPen("w")))
            self.fsm_plot.addItem(pg.ScatterPlotItem([e[-1, 0]], [e[-1, 1]], size=10, symbol="t", brush=pg.mkBrush(theme.GREEN), pen=pg.mkPen("w")))
            span = max(300.0, float(np.max(np.abs(e))) * 1.1)
        else:
            span = 300.0
        self.fsm_plot.addItem(pg.ScatterPlotItem([0], [0], size=14, symbol="+", pen=pg.mkPen(theme.RED, width=2), brush=None))
        self.fsm_plot.setRange(xRange=(-span, span), yRange=(-span, span))
        self.fsm_plot.setTitle("Quadrant detector: spiral search (amber) then PI tracking (green) onto (0, 0)" if out.closed_loop else
                               "FSM OFF: the beam stays wherever the platform points it (offset in µrad)", size="9pt", color=theme.TEXT)
        self.pat_log.setPlainText("\n".join(f"[{t * 1e3:9.3f} ms] {txt}" for t, txt in out.pat_events))

    def _redraw_diff(self) -> None:
        if self._diff_data is None:
            return
        sent, got = self._diff_data
        wrong = pl.byte_diff(sent, got)
        hexd = self.hex_check.isChecked()
        self.diff_tx.show_bytes(sent, wrong, hexd)
        self.diff_rx.show_bytes(got, wrong, hexd)

    def show_stream(self, s: ob.StreamResult) -> None:
        self._diff_data = (s.payload, s.recovered)
        self._redraw_diff()
        if s.last is not None:
            self._draw_constellation(s.last)
            self._draw_fsm(s.last)
        ok = s.frames_lost == 0 and s.stream_crc_ok and not s.cancelled
        self._set_badge(ok, f"STREAM  {s.frames_ok}/{len(s.frames)} frames delivered  ·  {s.frame_loss_pct:.1f} % lost" + ("  ·  CANCELLED" if s.cancelled else ""))
        n_wrong = int(pl.byte_diff(s.payload, s.recovered).sum())
        summary = (f"{len(s.frames)} frames ({len(s.payload):,} bytes): {s.frames_ok} delivered, {s.frames_lost} lost ({s.frame_loss_pct:.1f} %).  "
                   f"Route changes during the stream: {s.handovers}" + ("  (all make-before-break: no frame dropped by a handover)" if s.handovers and s.all_handovers_hitless else
                                                                           "  (some were NOT hitless)" if s.handovers else "") +
                   f".  Whole-file CRC-32 {'MATCHES' if s.stream_crc_ok else 'does NOT match'}; {n_wrong:,} bytes wrong.")
        if s.blocks is not None:
            summary += f"  64-byte blocks bad: {s.blocks.n_bad} of {s.blocks.n_blocks}."
        self.stream_summary.setText(summary)
        m = self.m
        m["route"].setText(f"{len(set(r.route_label for r in s.frames))} different routes over the stream")
        m["mod"].setText(", ".join(sorted({r.modulation for r in s.frames})))
        m["bytes"].setText(f"{len(s.payload):,} / {s.bytes_ok:,} (frames passing CRC)")
        m["wrong"].setText(f"{n_wrong:,}  ({100.0 * n_wrong / max(len(s.payload), 1):.2f} %)")
        m["blocks"].setText("—" if s.blocks is None else f"{s.blocks.n_bad} of {s.blocks.n_blocks}")
        m["floss"].setText(f"{s.frame_loss_pct:.1f} %  ({s.frames_lost} of {len(s.frames)} frames)")
        lat = [r.latency_ms for r in s.frames if math.isfinite(r.latency_ms)]
        m["lat"].setText(f"{min(lat):.1f} – {max(lat):.1f} ms" if lat else "—")
        for key in ("ber", "ber_o", "lat_detail", "pat", "snr", "rx", "hops"):
            m[key].setText("see the stream log")
        # table
        self.table.setRowCount(len(s.frames))
        for row, r in enumerate(s.frames):
            vals = [str(r.k + 1), f"{r.time_s / 60:.1f}", r.route_label, r.modulation, "—" if r.ber is None else f"{r.ber:.1e}", "OK" if r.ok else "LOST",
                    ("handover" + (" (hitless)" if r.hitless else " (NOT hitless)") if r.handover else "") + (" pre-locked" if r.pre_locked else "") +
                    ("" if r.ok else f" {r.reason}")]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if col == 5:
                    it.setForeground(QColor(theme.GREEN if r.ok else theme.RED))
                self.table.setItem(row, col, it)
        self.table.resizeColumnsToContents()
        # plot
        self.stream_plot.clear()
        bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}
        xs = np.array([r.time_s / 60 for r in s.frames])
        ys = np.array([bps.get(r.modulation, 0) for r in s.frames], dtype=float)
        if len(xs):
            self.stream_plot.addItem(pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#00e5ff", width=1.5), stepMode=False))
            self.stream_plot.addItem(pg.ScatterPlotItem(xs[[r.ok for r in s.frames]], ys[[r.ok for r in s.frames]], size=7, brush=pg.mkBrush(theme.GREEN), pen=None))
            lost = [not r.ok for r in s.frames]
            if any(lost):
                self.stream_plot.addItem(pg.ScatterPlotItem(xs[lost], np.zeros(sum(lost)), size=10, symbol="x", pen=pg.mkPen(theme.RED, width=2), brush=None))
            for r in s.frames:
                if r.handover:
                    self.stream_plot.addItem(pg.InfiniteLine(pos=r.time_s / 60, angle=90, pen=pg.mkPen(theme.AMBER, style=Qt.PenStyle.DashLine)))
        self.tabs.setCurrentWidget(self.tabs.widget(3))
        self.status.setText("Stream finished." if not s.cancelled else "Stream cancelled.")
