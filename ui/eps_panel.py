"""
The "EPS Power, Battery & 48V Busbar" tab: a live power-flow schematic (solar -> BCDU <-> battery -> 48 V busbars -> Bank A / Bank B / pumps and lasers), the
battery-SoC / solar / load sawtooth over the whole scenario, the delivered voltage at every one of the 32 blades, and the stress-test controls.

The electrical solver is `eps_power_engine.py` (a closed loop with the cooling loop); this panel only runs it and shows the result. Everything except the array
area, efficiency, battery size, bus voltage and busbar geometry is a placeholder (see that module).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget, QHBoxLayout

import eps_power_engine as ep

from . import theme

MODE_COLORS = {ep.SUNLIT_PEAK: "#2ecc71", ep.SUNLIT_RECHARGE: "#f2d024", ep.ECLIPSE_CRUISE: "#35b6ff", ep.ECLIPSE_SURVIVAL: "#e5383b"}
SOLAR, LOAD, BAT_IN, BAT_OUT, BUS = QColor("#ffc233"), QColor("#35e0ff"), QColor("#2ecc71"), QColor("#ff8a3d"), QColor("#7aa2ff")
MAX_POINTS = 3500
DEGRADATION_MAX_PCT = 95


class PowerFlowWidget(QWidget):
    """A Sankey-style schematic: box per component, arrows whose width follows the power."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(330)
        self.s: dict | None = None

    def set_state(self, state: dict | None) -> None:
        self.s = state
        self.update()

    def _box(self, p: QPainter, r: QRectF, title: str, lines: list, color: QColor, dim: bool = False) -> None:
        c = QColor(color)
        c.setAlpha(70 if dim else 200)
        p.setPen(QPen(c, 1.6))
        p.setBrush(QColor(theme.PANEL_2))
        p.drawRoundedRect(r, 5, 5)
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(c)
        p.drawText(QRectF(r.left() + 4, r.top() + 3, r.width() - 8, 16), Qt.AlignmentFlag.AlignHCenter, title)
        p.setFont(theme.mono_font(8))
        p.setPen(QColor(theme.DIM if dim else theme.TEXT))
        for k, line in enumerate(lines):
            p.drawText(QRectF(r.left() + 4, r.top() + 21 + 15 * k, r.width() - 8, 15), Qt.AlignmentFlag.AlignHCenter, line)

    def _arrow(self, p: QPainter, a: QPointF, b: QPointF, power_kw: float, color: QColor, label: str = "") -> None:
        if abs(power_kw) < 0.01:
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.PenStyle.DashLine))
            p.drawLine(a, b)
            return
        w = 2.0 + min(abs(power_kw), 90.0) / 90.0 * 14.0
        c = QColor(color)
        c.setAlpha(190)
        pen = QPen(c, w)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.drawLine(a, b)
        d = b - a
        n = (d.x() ** 2 + d.y() ** 2) ** 0.5 or 1.0
        ux, uy = d.x() / n, d.y() / n
        tip = QPointF(b.x() + ux * 8, b.y() + uy * 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPolygon(QPolygonF([tip, QPointF(b.x() - uy * (w / 2 + 3), b.y() + ux * (w / 2 + 3)), QPointF(b.x() + uy * (w / 2 + 3), b.y() - ux * (w / 2 + 3))]))
        if label:
            p.setPen(QColor(theme.TEXT))
            p.setFont(theme.mono_font(8))
            p.drawText(QPointF((a.x() + b.x()) / 2 - 20, (a.y() + b.y()) / 2 - w / 2 - 3), label)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))
        s = self.s
        if s is None:
            p.setPen(QColor(theme.DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Computing the power system ...")
            return
        w, h = self.width(), self.height()
        bh, row = 84.0, 92.0
        cols = [(0.01, 0.245), (0.30, 0.18), (0.52, 0.19), (0.75, 0.24)]           # (left, width) as fractions of the widget
        def box(c: int, r: int) -> QRectF:
            return QRectF(w * cols[c][0], 30 + row * r, w * cols[c][1], bh)
        mc = QColor(MODE_COLORS.get(s["mode"], theme.TEXT))
        p.setPen(mc)
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSize(10)
        p.setFont(f)
        badges = s["mode"] + ("   ·   DVFS eco-clock" if s["dvfs"] else "") + ("   ·   BANK B BREAKER OPEN" if not s["bank_b_closed"] else "") \
            + ("   ·   Bank B asleep (thermal)" if s["bank_b_hib"] else "") + ("   ·   BROWNOUT" if s["brownout"] else "")
        p.drawText(QRectF(8, 4, w - 16, 20), Qt.AlignmentFlag.AlignLeft, badges)
        solar, bcdu, batt, bus = box(0, 0), box(1, 1), box(0, 2), box(2, 1)
        bank_a, bank_b, misc = box(3, 0), box(3, 1), box(3, 2)
        sk, ld = s["solar_kw"], s["load_kw"]
        bat_kw = s["bat_kw"]                                                           # + discharging
        self._arrow(p, QPointF(solar.right(), solar.center().y()), QPointF(bcdu.left(), bcdu.top() + 16), sk, SOLAR, f"{sk:.1f} kW")
        if bat_kw >= 0:
            self._arrow(p, QPointF(batt.right(), batt.center().y()), QPointF(bcdu.left(), bcdu.bottom() - 16), bat_kw, BAT_OUT, f"{bat_kw:.1f} kW out")
        else:
            self._arrow(p, QPointF(bcdu.left(), bcdu.bottom() - 16), QPointF(batt.right(), batt.center().y()), -bat_kw, BAT_IN, f"{-bat_kw:.1f} kW in")
        self._arrow(p, QPointF(bcdu.right(), bcdu.center().y()), QPointF(bus.left(), bus.center().y()), ld, BUS, f"{ld:.1f} kW")
        self._arrow(p, QPointF(bus.right(), bus.center().y() - 14), QPointF(bank_a.left(), bank_a.center().y()), s["bank_a_kw"], LOAD)
        self._arrow(p, QPointF(bus.right(), bus.center().y()), QPointF(bank_b.left(), bank_b.center().y()), s["bank_b_kw"], LOAD)
        self._arrow(p, QPointF(bus.right(), bus.center().y() + 14), QPointF(misc.left(), misc.center().y()), s["fixed_kw"], LOAD)
        self._box(p, solar, "SOLAR ARRAY", [f"{sk:.1f} kW", f"{s['area_m2']:.0f} m²", f"cell {s['t_array_c']:.0f} °C", f"cos {s['cos']:.2f} sun {s['illum'] * 100:.0f}%"], SOLAR)
        self._box(p, bcdu, "BCDU", ["regulators", f"bat {s['bat_kw']:+.1f} kW", f"shunt {s['shunt_kw']:.1f} kW"], BUS)
        self._box(p, batt, "8x Li-Ion PACK", [f"SoC {s['soc_pct']:.1f}% DoD {100 - s['soc_pct']:.1f}%", f"{s['i_bat_a']:+.0f} A  {s['c_rate']:.2f} C", f"{s['pack_v']:.1f} V", f"R {s['r_mohm']:.1f} mΩ"],
                  BAT_IN if bat_kw < 0 else BAT_OUT)
        self._box(p, bus, "48 V BUSBARS", [f"{s['i_bus_a']:.0f} A (4 bars)", f"drop {s['drop_mv']:.0f} mV", f"loss {s['bus_loss_w']:.0f} W", f"{s['t_bus_c']:.0f} °C"], BUS)
        self._box(p, bank_a, "BANK A  1-16", [f"{s['bank_a_kw']:.1f} kW", f"{s['bank_a_a']:.0f} A"], LOAD)
        self._box(p, bank_b, "BANK B  17-32", [f"{s['bank_b_kw']:.1f} kW" if s["bank_b_closed"] else "OFF: breaker open", f"{s['bank_b_a']:.0f} A" if s["bank_b_closed"] else ""], LOAD,
                  dim=not s["bank_b_closed"] or s["bank_b_hib"])
        self._box(p, misc, "PUMPS·OISL·OBC", [f"{s['fixed_kw']:.1f} kW", f"pumps {s['rpm'] * 100:.0f}% RPM", "lasers low" if s["oisl_low"] else "lasers full"], LOAD)


class EPSPanel(QWidget):
    run_changed = Signal(object)            # an eps_power_engine.EPSRun, whenever the electrical-thermal run is (re)computed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.run: ep.EPSRun | None = None
        self._idx = 0
        self._regions: list = []
        self._lay_ready = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.status = QLabel("—")
        self.status.setWordWrap(True)
        self.status.setFont(theme.mono_font(10))
        lay.addWidget(self.status)
        note = QLabel("Array 182.4 m², 32 % cells, 320 kWh, 48 V bus and the busbar geometry are the design's numbers. The blade power (1.2 kW each), converter and battery "
                      "resistances, array and bus temperatures and all thresholds' hysteresis are placeholders. Not a flight-qualified power budget.")
        note.setWordWrap(True)
        note.setProperty("role", "dim")
        lay.addWidget(note)

        g = self._group("POWER FLOW")
        gl = QVBoxLayout(g)
        self.flow = PowerFlowWidget()
        gl.addWidget(self.flow)
        lay.addWidget(g)

        g = self._group("BATTERY SoC AND ECLIPSE SAWTOOTH  ·  whole scenario")
        gl = QVBoxLayout(g)
        self.soc_plot = pg.PlotWidget()
        self.soc_plot.setMinimumHeight(150)
        self.power_plot = pg.PlotWidget()
        self.power_plot.setMinimumHeight(150)
        for pw, ylab in ((self.soc_plot, "SoC %"), (self.power_plot, "kW")):
            pw.setBackground(theme.BG)
            pw.showGrid(x=True, y=True, alpha=0.18)
            pw.setMenuEnabled(False)
            pw.setLabel("left", ylab)
        self.soc_plot.setYRange(0, 105)
        self.soc_plot.addItem(pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen(theme.RED, style=Qt.PenStyle.DashLine), label="20 % floor", labelOpts=dict(position=0.08, color=theme.RED)))
        self.soc_plot.addItem(pg.InfiniteLine(pos=35, angle=0, pen=pg.mkPen(theme.AMBER, style=Qt.PenStyle.DashLine), label="35 % survival", labelOpts=dict(position=0.08, color=theme.AMBER)))
        self.soc_curve = self.soc_plot.plot([], [], pen=pg.mkPen(theme.GREEN, width=2))
        self.power_plot.setLabel("bottom", "hours")
        self.power_plot.addLegend(offset=(-10, 10))
        self.solar_curve = self.power_plot.plot([], [], pen=pg.mkPen("#ffc233", width=2), name="solar generation")
        self.load_curve = self.power_plot.plot([], [], pen=pg.mkPen("#35e0ff", width=2), name="total load")
        self.cursors = [pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("w", width=1)) for _ in range(2)]
        self.soc_plot.addItem(self.cursors[0])
        self.power_plot.addItem(self.cursors[1])
        self.power_plot.setXLink(self.soc_plot)
        gl.addWidget(self.soc_plot)
        gl.addWidget(self.power_plot)
        self.eclipse_note = QLabel("—")
        self.eclipse_note.setWordWrap(True)
        self.eclipse_note.setFont(theme.mono_font(9))
        gl.addWidget(self.eclipse_note)
        lay.addWidget(g)

        g = self._group("32-BLADE BUSBAR VOLTAGE GRADIENT  ·  delivered volts, Blade 1 (base) to Blade 32 (top), axis zoomed")
        gl = QVBoxLayout(g)
        self.v_plot = pg.PlotWidget()
        self.v_plot.setMinimumHeight(170)
        self.v_plot.setBackground(theme.BG)
        self.v_plot.showGrid(x=False, y=True, alpha=0.18)
        self.v_plot.setMenuEnabled(False)
        self.v_plot.setLabel("left", "V at the blade")
        self.v_plot.setLabel("bottom", "blade number")
        self.v_plot.setXRange(0, 33)
        self.bars_a = pg.BarGraphItem(x=np.arange(1, 17), height=np.zeros(16), width=0.8, y0=45.0, brush=pg.mkBrush(53, 182, 255, 200))
        self.bars_b = pg.BarGraphItem(x=np.arange(17, 33), height=np.zeros(16), width=0.8, y0=45.0, brush=pg.mkBrush(199, 125, 255, 200))
        self.v_plot.addItem(self.bars_a)
        self.v_plot.addItem(self.bars_b)
        self.setpoint = pg.InfiniteLine(pos=48.0, angle=0, pen=pg.mkPen(theme.GREEN, style=Qt.PenStyle.DashLine), label="48.0 V BCDU set-point", labelOpts=dict(position=0.5, color=theme.GREEN))
        self.v_plot.addItem(self.setpoint)
        self.v_plot.setYRange(47.8, 48.02, padding=0)
        gl.addWidget(self.v_plot)
        self.v_note = QLabel("—")
        self.v_note.setWordWrap(True)
        self.v_note.setFont(theme.mono_font(9))
        gl.addWidget(self.v_note)
        lay.addWidget(g)

        g = self._group("STRESS TESTS")
        sl = QGridLayout(g)
        self.wing_btn = QPushButton("Simulate 1-Panel Solar Wing Failure")
        self.wing_btn.setCheckable(True)
        self.wing_btn.setToolTip("Loses 1 of the 12 panels (15.2 m² of 182.4 m²).")
        self.eclipse_btn = QPushButton("Force Extended 45-Min Eclipse")
        self.eclipse_btn.setCheckable(True)
        self.eclipse_btn.setToolTip("Lengthens every eclipse that is shorter than 45 minutes (in the power model only: the map and the timeline still show the real shadow).")
        self.deg_label = QLabel("Battery Cell Degradation: capacity lost 0 %")
        self.deg_slider = QSlider(Qt.Orientation.Horizontal)
        self.deg_slider.setRange(0, DEGRADATION_MAX_PCT)
        self.deg_slider.setToolTip("Capacity fade (and the matching rise in internal resistance). Real end-of-life is around 20-30 %; the slider goes far beyond that so that the "
                                   "low-battery modes can be seen.")
        self.reset_btn = QPushButton("Reset stress tests")
        sl.addWidget(self.wing_btn, 0, 0)
        sl.addWidget(self.eclipse_btn, 0, 1)
        sl.addWidget(self.deg_label, 1, 0, 1, 2)
        sl.addWidget(self.deg_slider, 2, 0, 1, 2)
        sl.addWidget(self.reset_btn, 3, 0, 1, 2)
        lay.addWidget(g)
        lay.addStretch(1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self.rerun)
        self.wing_btn.toggled.connect(lambda _v: self.rerun())
        self.eclipse_btn.toggled.connect(lambda _v: self.rerun())
        self.deg_slider.valueChanged.connect(self._slider)
        self.reset_btn.clicked.connect(self.reset_stress)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #35b6ff; border: 1px solid #263241; border-radius: 4px; margin-top: 8px; padding-top: 6px; }"
                        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")
        return g

    def config(self) -> ep.EPSConfig:
        return ep.EPSConfig(panels_failed=1 if self.wing_btn.isChecked() else 0, force_eclipse_min=45.0 if self.eclipse_btn.isChecked() else 0.0,
                            soh=1.0 - self.deg_slider.value() / 100.0)

    def _slider(self, v: int) -> None:
        self.deg_label.setText(f"Battery Cell Degradation: capacity lost {v} %  ({ep.EPSConfig(soh=1.0 - v / 100.0).capacity_kwh:.0f} kWh left)")
        self._debounce.start()

    def reset_stress(self) -> None:
        for w in (self.wing_btn, self.eclipse_btn):
            w.blockSignals(True)
            w.setChecked(False)
            w.blockSignals(False)
        self.deg_slider.blockSignals(True)
        self.deg_slider.setValue(0)
        self.deg_slider.blockSignals(False)
        self._slider(0)
        self._debounce.stop()
        self.rerun()

    # ------------------------------------------------------------------ run and show
    def set_result(self, result) -> None:
        self.result = result
        self.rerun()

    def rerun(self) -> None:
        self._debounce.stop()
        if self.result is None:
            return
        self.run = ep.simulate_eps(self.result, self.config())
        self._draw_curves()
        self._idx = min(self._idx, self.run.history.n - 1)
        self.set_index(self._idx)
        self.run_changed.emit(self.run)

    def set_time(self, t: float) -> None:
        if self.run is not None:
            self.set_index(self.run.history.index_at(t))

    def _draw_curves(self) -> None:
        h = self.run.history
        stride = max(1, h.n // MAX_POINTS)
        x = h.t_s[::stride] / 3600.0
        self.soc_curve.setData(x, h.soc[::stride] * 100.0)
        self.solar_curve.setData(x, h.solar_w[::stride] / 1000.0)
        self.load_curve.setData(x, h.load_w[::stride] / 1000.0)
        self.soc_plot.setXRange(0.0, float(h.t_s[-1]) / 3600.0, padding=0.01)
        self.power_plot.setYRange(0.0, max(float(h.solar_w.max()), float(h.load_w.max())) / 1000.0 * 1.12, padding=0)
        for r in self._regions:
            r.scene().removeItem(r) if r.scene() is not None else None
        self._regions = []
        for pw in (self.soc_plot, self.power_plot):
            for a, b in ep.eclipse_runs(h.illumination, h.cfg.sunlit_threshold)[:60]:
                r = pg.LinearRegionItem((h.t_s[a] / 3600.0, h.t_s[min(b, h.n - 1)] / 3600.0), movable=False, brush=pg.mkBrush(91, 79, 207, 45), pen=pg.mkPen(None))
                r.setZValue(-10)
                pw.addItem(r)
                self._regions.append(r)
        s = h.summary()
        ec = s["eclipses"]
        if ec:
            e = ec[min(1, len(ec) - 1)]
            self.eclipse_note.setText(f"{len(ec)} eclipses, {np.mean([x['duration_min'] for x in ec]):.1f} min each on average. Typical one: SoC {e['soc_entry_pct']:.1f} % -> {e['soc_min_pct']:.1f} % "
                                      f"(DoD {e['dod_pct']:.1f} %, {e['energy_kwh']:.1f} kWh).  Lowest SoC {s['soc_min_pct']:.1f} % "
                                      f"({'20 % FLOOR VIOLATED in ' + str(s['floor_violation_steps']) + ' steps' if s['floor_violation_steps'] else 'above the 20 % floor'}); "
                                      f"{s['flushes']} survival checkpoint flush(es); Bank B open {s['bank_b_open_min']:.0f} min.")
        else:
            self.eclipse_note.setText("No eclipse in this scenario.")

    def set_index(self, i: int) -> None:
        if self.run is None:
            return
        h = self.run.history
        i = int(min(max(i, 0), h.n - 1))
        self._idx = i
        t_h = h.t_s[i] / 3600.0
        for c in self.cursors:
            c.setValue(t_h)
        prof = h.blades(i)
        v = prof["v_blade"]
        lo = float(np.floor(v.min() * 200.0) / 200.0 - 0.01)                                # zoomed axis: the whole drop is only a tenth of a volt
        hi = max(float(h.cfg.bus_v), float(v.max())) + 0.01
        self.bars_a.setOpts(height=v[:16] - lo, y0=lo)
        self.bars_b.setOpts(height=v[16:] - lo, y0=lo)
        self.v_plot.setYRange(lo, hi, padding=0)
        self.setpoint.setValue(h.cfg.bus_v)
        ok = prof["in_band"]
        self.v_note.setText(f"Blade 1 {v[0]:.3f} V · Blade 16 {v[15]:.3f} V · Blade 32 {v[31]:.3f} V · worst drop {prof['drop_max_v'] * 1000:.0f} mV over the {h.cfg.bus_length_m:g} m bars "
                            f"({h.cfg.bar_width_m * 1000:.0f} × {h.cfg.bar_thickness_m * 1000:.0f} mm copper at {h.t_bus_c[i]:.0f} °C) · {prof['i_total']:.0f} A · "
                            f"{'inside' if ok else 'OUTSIDE'} the 45.5-50.0 V server VRM band "
                            f"(the axis is zoomed to show the drop; the band is {v.min() - ep.VRM_MIN_V:.1f} V away).")
        a_kw, b_kw = h.p_bank_a_w[i] / 1000.0, h.p_bank_b_w[i] / 1000.0
        self.flow.set_state(dict(mode=h.mode[i], dvfs=bool(h.dvfs[i]), bank_b_closed=bool(h.bank_b_closed[i]), bank_b_hib=bool(h.bank_b_hibernated[i]), brownout=bool(h.brownout[i]),
                                 solar_kw=h.solar_w[i] / 1000.0, load_kw=h.load_w[i] / 1000.0, bat_kw=h.bat_power_w[i] / 1000.0, shunt_kw=h.shunt_w[i] / 1000.0, soc_pct=h.soc[i] * 100.0,
                                 i_bat_a=h.bat_current_a[i], c_rate=h.c_rate[i], pack_v=h.pack_v[i], r_mohm=h.r_int_mohm[i], i_bus_a=prof["i_total"], drop_mv=prof["drop_max_v"] * 1000.0,
                                 bus_loss_w=prof["loss_w"], t_bus_c=h.t_bus_c[i], bank_a_kw=a_kw, bank_b_kw=b_kw, bank_a_a=prof["i_bank_a"], bank_b_a=prof["i_bank_b"],
                                 fixed_kw=h.p_fixed_w[i] / 1000.0, rpm=h.pump_rpm[i], oisl_low=bool(h.oisl_low_power[i]), area_m2=h.cfg.usable_area_m2, t_array_c=h.t_array_c[i],
                                 cos=h.cos_incidence[i], illum=h.illumination[i]))
        th = self.run.thermal
        self.status.setText(f"{h.mode[i]}   SoC {h.soc[i] * 100:.1f} %   solar {h.solar_w[i] / 1000:.1f} kW   load {h.load_w[i] / 1000:.1f} kW   "
                            f"battery {h.bat_power_w[i] / 1000:+.1f} kW\nbattery heat {h.bat_heat_w[i]:.0f} W + busbar {h.bus_loss_w[i]:.0f} W -> cooling loop "
                            f"(total {h.heat_w[i] / 1000:.1f} kW), core {th.core_temp_c[i]:.1f} °C   training throughput {100 * h.throughput()[i]:.0f} %"
                            + ("   [stress: " + ", ".join(x for x in (f"{h.cfg.panels_failed} panel failed" if h.cfg.panels_failed else "", "eclipse ≥ 45 min" if h.cfg.force_eclipse_min else "",
                                                                       f"capacity -{(1 - h.cfg.soh) * 100:.0f} %" if h.cfg.soh < 1 else "") if x) + "]"
                               if (h.cfg.panels_failed or h.cfg.force_eclipse_min or h.cfg.soh < 1) else ""))
