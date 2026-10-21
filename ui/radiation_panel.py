"""
The "Radiation-Tolerant AI Scheduler" tab: a live 2 x 16 blade-health grid, the protected-vs-unprotected training-loss chart, the chaos controls
(solar flare, protection on/off, forced latch-up, hot core, export) and an event log. The training runs in `radiation_pytorch_scheduler.py`, driven by
satSTK's playback clock: every time the clock moves, the panel advances the mission time and the 32-blade trainer runs the steps that time calls for.

All upset rates, zone boxes and the 15 s latchup recovery are SIMULATION ASSUMPTIONS (see seu_injector.py), not measurements.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QGroupBox, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

import radiation_pytorch_scheduler as rps
import seu_injector as si

from . import theme

CELL_COLORS = {rps.HEALTHY: "#1f9d55", rps.SCRUBBED: "#f2d024", rps.QUARANTINED: "#e5383b", rps.TMR_STATE: "#8e44d6", rps.HIBERNATED: "#5d6672", rps.RESET: "#5d6672"}
LEGEND = (("healthy", rps.HEALTHY), ("mantissa flip scrubbed by ECC", rps.SCRUBBED), ("exponent / NaN strike caught", rps.QUARANTINED),
          ("TMR shielding (SAA)", rps.TMR_STATE), ("hibernated / power-cycling", rps.HIBERNATED))
HOT_CORE_OFFSET_C = 35.0
CHART_POINTS = 3000
LOSS_CEILING = 3.0


def _cell_style(state: str) -> str:
    c = CELL_COLORS[state]
    fg = "#101010" if state == rps.SCRUBBED else "#ffffff"
    border = "#ffffff" if state in (rps.SCRUBBED, rps.QUARANTINED) else "#0a0e13"
    return f"background: {c}; color: {fg}; border: 1px solid {border}; border-radius: 3px; font-weight: bold;"


class RadiationPanel(QWidget):
    overlay_changed = Signal(bool, float)            # (show the overlay?, current flux multiplier)
    downlink_requested = Signal(str)                 # path of an exported weights file to send through the 16QAM optical link

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.orch: rps.RadiationOrchestrator | None = None
        self._thermal = None
        self._cell_state = [None] * rps.N_BLADES
        self._regions: list = []
        self._region_key = None
        self._engine_pref = "auto"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        self.status.setFont(theme.mono_font(10))
        lay.addWidget(self.status)
        note = QLabel("Upset rates (1e-5 SEU/blade/s baseline, x15 polar, x100 South Atlantic Anomaly), the zone boxes, the 15 s latch-up recovery and the "
                      "100 Gbps backplane are simulation assumptions, not measurements. The 'blades' are simulated replicas in one process.")
        note.setWordWrap(True)
        note.setProperty("role", "dim")
        lay.addWidget(note)

        g = self._group("BLADE HEALTH  ·  Alpha-Sigma, 2 banks x 16 trays")
        gl = QGridLayout(g)
        gl.setSpacing(3)
        self.cells: list = []
        for bank, base in (("A", 0), ("B", 16)):
            lab = QLabel(f"BANK {bank}")
            lab.setProperty("role", "dim")
            gl.addWidget(lab, 0 if bank == "A" else 1, 0)
            for k in range(16):
                cell = QLabel(f"{k + 1:02d}")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setMinimumSize(22, 22)
                cell.setStyleSheet(_cell_style(rps.HEALTHY))
                gl.addWidget(cell, 0 if bank == "A" else 1, k + 1)
                self.cells.append(cell)
        for k, (text, state) in enumerate(LEGEND):                                 # three rows of two, so the legend never widens the panel
            t = QLabel(f'<span style="color:{CELL_COLORS[state]}">■</span> {text}')
            t.setProperty("role", "dim")
            gl.addWidget(t, 2 + k // 2, 1 + (k % 2) * 8, 1, 8)
        lay.addWidget(g)

        g = self._group("TRAINING LOSS  ·  protected vs unprotected")
        gv = QVBoxLayout(g)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(190)
        self.plot.setBackground(theme.BG)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "training step")
        self.plot.setLabel("left", "loss")
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setYRange(0.0, LOSS_CEILING, padding=0.02)
        self.curve_p = self.plot.plot([], [], pen=pg.mkPen(theme.GREEN, width=2), name="protected (sanitizer + TMR + ECC/OBC)")
        self.curve_u = self.plot.plot([], [], pen=pg.mkPen(theme.RED, width=2), name="unprotected baseline", connect="finite")
        self.nan_mark = pg.ScatterPlotItem([], [], size=14, symbol="x", pen=pg.mkPen(theme.RED, width=3), brush=None)
        self.plot.addItem(self.nan_mark)
        self.nan_text = pg.TextItem("", color=theme.RED, anchor=(0, 0))
        self.plot.addItem(self.nan_text)
        gv.addWidget(self.plot)
        self.stats = QLabel("—")
        self.stats.setWordWrap(True)
        self.stats.setFont(theme.mono_font(9))
        gv.addWidget(self.stats)
        lay.addWidget(g)

        g = self._group("CHAOS CONTROLS")
        cl = QGridLayout(g)
        self.flare_btn = QPushButton("Force Solar Flare / CME Event (1000x Flux)")
        self.flare_btn.setCheckable(True)
        self.flare_btn.setToolTip("Multiplies the radiation flux 1000-fold everywhere until you click it again.")
        self.protect_btn = QPushButton("Toggle ECC & TMR Protection ON/OFF")
        self.protect_btn.setCheckable(True)
        self.protect_btn.setChecked(True)
        self.protect_btn.setToolTip("Switch the sanitizer, TMR voting, ECC parity and OBC checkpointing off for the 'protected' line (it then behaves like the baseline).")
        self.export_btn = QPushButton("Export Recovered .pt Weights")
        self.export_btn.setToolTip("Save the OBC-verified weights (a PyTorch .pt file; a NumPy .npz when torch is not installed) and optionally send them over the 16QAM optical link.")
        self.sel_btn = QPushButton("Force a latch-up (SEL) on a random blade")
        self.hot_btn = QPushButton(f"Simulate a hot core (+{HOT_CORE_OFFSET_C:g} °C) -> soft throttle")
        self.hot_btn.setCheckable(True)
        self.reset_btn = QPushButton("Restart training")
        self.overlay_cb = QCheckBox("SAA / auroral overlay on the map and globe")
        self.overlay_cb.setChecked(True)
        self.downlink_cb = QCheckBox("After export, downlink the file over the 16QAM link")
        self.downlink_cb.setChecked(True)
        for k, w in enumerate((self.flare_btn, self.protect_btn, self.export_btn, self.sel_btn, self.hot_btn, self.reset_btn)):
            cl.addWidget(w, k, 0)
        cl.addWidget(self.overlay_cb, 6, 0)
        cl.addWidget(self.downlink_cb, 7, 0)
        lay.addWidget(g)

        g = self._group("EVENT LOG")
        el = QVBoxLayout(g)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setFont(theme.mono_font(9))
        self.log.setMinimumHeight(120)
        el.addWidget(self.log)
        lay.addWidget(g)
        lay.addStretch(1)

        self.flare_btn.toggled.connect(self._flare)
        self.protect_btn.toggled.connect(self._protect)
        self.export_btn.clicked.connect(self.export_dialog)
        self.sel_btn.clicked.connect(self._latchup)
        self.hot_btn.toggled.connect(self._hot)
        self.reset_btn.clicked.connect(self.restart)
        self.overlay_cb.toggled.connect(lambda _on: self._emit_overlay())

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #35b6ff; border: 1px solid #263241; border-radius: 4px; margin-top: 8px; padding-top: 6px; }"
                        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")
        return g

    def _emit_overlay(self) -> None:
        self.overlay_changed.emit(self.overlay_cb.isChecked(), self.flux_now())

    def flux_now(self) -> float:
        rp = self.orch.last[0] if self.orch else None
        return rp.flux_mult if rp is not None else 1.0

    # ------------------------------------------------------------------ simulation state in
    def set_result(self, result, thermal_history=None) -> None:
        self.result = result
        self._thermal = rps.CoreTemp(result, thermal_history)
        self.restart()

    def restart(self) -> None:
        if self.result is None:
            return
        flare, prot, hot = self.flare_btn.isChecked(), self.protect_btn.isChecked(), self.hot_btn.isChecked()
        orbit = rps.TrackOrbit.from_result(self.result)
        self.orch = rps.RadiationOrchestrator(orbit, self._thermal, rps.OrchestratorConfig(engine=self._engine_pref))
        self.orch.set_flare(flare)
        self.orch.set_protection(prot)
        self.orch.temp_offset_c = HOT_CORE_OFFSET_C if hot else 0.0
        self.orch.resync(0.0)
        self.log.clear()
        self._region_key = None
        self._cell_state = [None] * rps.N_BLADES
        self._refresh()

    def set_time(self, t: float) -> None:
        if self.orch is None:
            return
        done = self.orch.advance_to(t)
        if done:
            self._refresh()

    # ------------------------------------------------------------------ chaos controls
    def _flare(self, on: bool) -> None:
        if self.orch:
            self.orch.set_flare(on)
            self.orch.note("SOLAR FLARE / CME begins: flux x1000" if on else "flare over: flux back to the orbit zone value")
            self._flush_log()

    def _protect(self, on: bool) -> None:
        if self.orch:
            self.orch.set_protection(on)
            self.orch.note(f"ECC & TMR protection switched {'ON' if on else 'OFF'}")
            self._flush_log()

    def _hot(self, on: bool) -> None:
        if self.orch:
            self.orch.temp_offset_c = HOT_CORE_OFFSET_C if on else 0.0

    def _latchup(self) -> None:
        if self.orch:
            b = self.orch.force_latchup()
            self.orch.note(f"[chaos] latch-up forced on blade {rps.blade_name(b)}; it goes offline at the next training step")
            self._flush_log()

    def export_dialog(self) -> None:
        if self.orch is None:
            return
        ext = ".pt" if rps.TORCH_AVAILABLE else ".npz"
        path, _ = QFileDialog.getSaveFileName(self, "Export recovered weights", f"alpha_sigma_recovered{ext}", "Weights (*.pt *.npz)")
        if path:
            self.export_to(path)

    def export_to(self, path) -> Path:
        out = self.orch.export_weights(path)
        self.orch.note(f"[export] recovered weights written to {out} ({out.stat().st_size:,} bytes)")
        self._flush_log()
        if self.downlink_cb.isChecked():
            self.downlink_requested.emit(str(out))
        return out

    # ------------------------------------------------------------------ drawing
    def _refresh(self) -> None:
        o = self.orch
        rp, rb = o.last
        for i, (state, tip) in enumerate(o.blade_grid()):
            if self._cell_state[i] != state:
                self.cells[i].setStyleSheet(_cell_style(state))
                self._cell_state[i] = state
            self.cells[i].setToolTip(tip)
        h = o.history
        n = len(h["step"])
        if rp is not None:
            self.status.setText(
                f"zone {si.ZONE_LABEL[rp.zone]}   flux x{rp.flux_mult:,.1f}   mode {rp.mode}" + ("   [SAA ahead]" if rp.mode == rps.TMR and rp.zone != si.SAA else "")
                + f"\nprotection {'ON' if o.protected.protected else 'OFF'}   flare {'ACTIVE x1000' if o.flare > 1 else 'none'}   thermal {'SOFT THROTTLE: Bank B hibernating, Bank A carries the batch' if rp.throttled else 'nominal'}   "
                f"throughput {100 * rp.rel_throughput:.0f} % of the 32-batch maximum   engine {o.engine.name}")
            self._emit_overlay()
        if n:
            xs = np.asarray(h["step"][-CHART_POINTS:], float)
            yp = np.asarray(h["loss_protected"][-CHART_POINTS:], float)
            yu = np.asarray(h["loss_baseline"][-CHART_POINTS:], float)
            bad = ~np.isfinite(yu) | (yu > 1e6)
            yu = np.where(bad, np.nan, np.minimum(yu, LOSS_CEILING))
            self.curve_p.setData(xs, np.minimum(yp, LOSS_CEILING))
            self.curve_u.setData(xs, yu)
            if bad.any():
                first = xs[bad][0]
                self.nan_mark.setData([first], [LOSS_CEILING * 0.95])
                self.nan_text.setText(f"unprotected: NaN at step {int(first)}")
                self.nan_text.setPos(first, LOSS_CEILING * 0.9)
            else:
                self.nan_mark.setData([], [])
                self.nan_text.setText("")
            self._shade_saa(xs, h["zone"][-CHART_POINTS:])
        self._stats()
        self._flush_log()

    def _shade_saa(self, xs, zones) -> None:
        runs, start = [], None
        for x, z in zip(xs, zones):
            if z == si.SAA and start is None:
                start = x
            elif z != si.SAA and start is not None:
                runs.append((start, x))
                start = None
        if start is not None:
            runs.append((start, xs[-1]))
        runs = runs[-20:]
        key = tuple(runs[:-1])
        if key != self._region_key or len(self._regions) != len(runs):
            for r in self._regions:
                self.plot.removeItem(r)
            self._regions = []
            for a, b in runs:
                r = pg.LinearRegionItem((a, b), movable=False, brush=pg.mkBrush(255, 120, 40, 40), pen=pg.mkPen(None))
                r.setZValue(-10)
                self.plot.addItem(r)
                self._regions.append(r)
            self._region_key = key
        elif self._regions:
            self._regions[-1].setRegion(runs[-1])

    def _stats(self) -> None:
        o = self.orch
        t, b = o.protected.totals, o.baseline.totals
        w = o.protected.consensus_weights()
        vl, va = o.protected.evaluate(w)
        wb = o.baseline.consensus_weights()
        bl, ba = o.baseline.evaluate(wb)
        budget = "within" if t["max_restore_ms"] < rps.RESTORE_BUDGET_MS else "OVER"
        self.stats.setText(
            f"upsets {t['hits']}: mantissa {t['mantissa']}, exponent {t['exponent']}, sign {t['sign']}; blow-ups {t['blowups']}; latch-ups {t['latchups']}\n"
            f"PROTECTED   quarantined {t['quarantines']}, outvoted {t['voted_out']}, scrubbed {t['scrubs']}, restores {t['restores']} "
            f"(slowest {t['max_restore_ms']:.3f} ms, {budget} the {rps.RESTORE_BUDGET_MS:g} ms budget), TMR steps {t['tmr_steps']}  | val loss {vl:.3f}, acc {100 * va:.0f} %\n"
            f"UNPROTECTED {'NaN/Inf - training destroyed' if o.baseline.poisoned else 'still alive'}  | val loss {bl:.3f}, acc {100 * ba:.0f} %"
            + ("" if o.protected.protected else "\n(protection is OFF: the green line now suffers the same fate as the red one)"))

    def _flush_log(self) -> None:
        for line in self.orch.drain_log()[-40:]:
            self.log.appendPlainText(line)
