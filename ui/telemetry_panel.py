"""Right-hand dock: numeric readouts for the link and for each satellite, and live plots of the key link/thermal quantities."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import orbital_engine as oe
from laser_link_engine import MODULATIONS

from . import theme

pg.setConfigOptions(antialias=True, background=theme.PANEL, foreground=theme.DIM)
ECLIPSE_COLORS = {oe.ECLIPSE_SUNLIT: "#ffd93d", oe.ECLIPSE_PENUMBRA: theme.AMBER, oe.ECLIPSE_UMBRA: "#9d8cff"}
MAX_PLOT_POINTS = 4000


def _fmt(x, spec: str, unit: str = "") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{format(x, spec)}{(' ' + unit) if unit else ''}"


def _fmt_ber(x: float) -> str:
    return "—" if not math.isfinite(x) else ("< 1e-30" if x <= 1.0001e-30 else f"{x:.2e}")


class TelemetryPanel(QWidget):
    modulation_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.modulation = "QPSK"
        self._plots = {}
        self._lines = []
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        top = QHBoxLayout()
        top.addWidget(QLabel("Modulation for BER / throughput:"))
        self._mod = QComboBox()
        self._mod.addItems(list(MODULATIONS))
        self._mod.setCurrentText(self.modulation)
        self._mod.currentTextChanged.connect(self._modulation_changed)
        top.addWidget(self._mod)
        top.addStretch(1)
        root.addLayout(top)

        self._link_labels = {}
        root.addWidget(self._section("LASER LINK  (1550 nm OISL)"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(1)
        fields = [("status", "Status"), ("range", "Range"), ("rate", "Range rate"), ("latency", "Latency"), ("doppler", "Doppler"),
                  ("rx", "Received power"), ("snr", "SNR (Es/N0)"), ("ebn0", "Eb/N0"), ("ber", "BER"), ("thr", "Throughput"),
                  ("adaptive", "Best modulation"), ("fspl", "Path loss (FSPL)"), ("div", "Divergence loss"), ("point", "Pointing loss"),
                  ("clear", "Grazing clearance")]
        for n, (key, label) in enumerate(fields):
            r, c = divmod(n, 2)
            grid.addWidget(self._dim(label), r, c * 2)
            lab = self._value()
            grid.addWidget(lab, r, c * 2 + 1)
            self._link_labels[key] = lab
        root.addLayout(grid)

        self._sat_labels = []
        self._sat_grid = QGridLayout()
        self._sat_grid.setHorizontalSpacing(10)
        self._sat_grid.setVerticalSpacing(1)
        root.addWidget(self._section("SATELLITES"))
        root.addLayout(self._sat_grid)

        root.addWidget(self._section("LIVE PLOTS  (whole scenario, red line = now)"))
        self._plot_box = QVBoxLayout()
        self._plot_box.setSpacing(4)
        root.addLayout(self._plot_box)
        root.addStretch(1)

    # ------------------------------------------------------------------ small widget helpers
    @staticmethod
    def _section(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "section")
        return lab

    @staticmethod
    def _dim(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "dim")
        return lab

    @staticmethod
    def _value() -> QLabel:
        lab = QLabel("—")
        lab.setFont(theme.mono_font(10))
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return lab

    # ------------------------------------------------------------------ data in
    def set_result(self, result) -> None:
        self.result = result
        # per-satellite grid
        while self._sat_grid.count():
            w = self._sat_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._sat_labels = []
        rows = [("lat", "Latitude"), ("lon", "Longitude"), ("alt", "Altitude"), ("speed", "Speed"), ("ecl", "Sun / eclipse"),
                ("beta", "Beta angle"), ("temp", "Radiator temp"), ("pwr", "Solar array"), ("az", "Turret azimuth"),
                ("el", "Turret elevation"), ("slew", "Turret slew rate"), ("azr", "Az / El rate")]
        for k, tr in enumerate(result.tracks):
            head = QLabel(tr.name)
            head.setStyleSheet(f"color: {theme.sat_color(k).name()}; font-weight: bold;")
            self._sat_grid.addWidget(head, 0, 1 + k)
        labels = []
        for r, (key, name) in enumerate(rows, start=1):
            self._sat_grid.addWidget(self._dim(name), r, 0)
        for k in range(len(result.tracks)):
            d = {}
            for r, (key, name) in enumerate(rows, start=1):
                d[key] = self._value()
                self._sat_grid.addWidget(d[key], r, 1 + k)
            labels.append(d)
        self._sat_labels = labels
        self._build_plots()
        self.set_index(0)

    def _build_plots(self) -> None:
        while self._plot_box.count():
            w = self._plot_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._lines = []
        self._plots = {}
        res = self.result
        link = res.primary_link
        stride = max(1, math.ceil(res.n / MAX_PLOT_POINTS))
        x = (res.t_s / 3600.0)[::stride]

        def new_plot(key, title, ylabel, height=140):
            pw = pg.PlotWidget()
            pw.setFixedHeight(height)
            pw.setTitle(title, size="9pt", color=theme.TEXT)
            pw.setLabel("left", ylabel)
            pw.getAxis("left").enableAutoSIPrefix(False)
            pw.getAxis("left").setWidth(52)
            pw.showGrid(x=True, y=True, alpha=0.18)
            pw.setMenuEnabled(False)
            pw.setMouseEnabled(x=False, y=False)
            pw.getPlotItem().hideButtons()
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.RED, width=1.6))
            pw.addItem(line)
            self._lines.append(line)
            self._plot_box.addWidget(pw)
            self._plots[key] = pw
            return pw

        def curve(pw, y, color, width=1.6, style=None, name=None):
            pen = pg.mkPen(color, width=width, style=style or Qt.PenStyle.SolidLine)
            return pw.plot(x, np.asarray(y)[::stride], pen=pen, connect="finite", name=name)

        pw = new_plot("range", "Range (km)", "km")
        curve(pw, link.range_km, theme.ACCENT)
        pw = new_plot("rx", "Received optical power (dBm)", "dBm")
        curve(pw, link.rx_power_dbm, theme.GREEN)
        self._ber_plot = new_plot("ber", "Bit error rate (selected modulation, log scale)", "BER")
        self._ber_plot.setLogMode(False, True)
        self._ber_curve = curve(self._ber_plot, link.ber[self.modulation], "#ff7ab6")
        self._thr_plot = new_plot("thr", "Throughput (Gbps)  - solid: selected, dashed: best of four", "Gbps")
        self._thr_curve = curve(self._thr_plot, link.throughput_gbps[self.modulation], theme.AMBER)
        curve(self._thr_plot, link.adaptive_throughput_gbps, "#ffffff", 1.0, Qt.PenStyle.DashLine)
        pw = new_plot("temp", "Radiator equilibrium temperature (°C)", "°C")
        for k, tr in enumerate(res.tracks):
            curve(pw, tr.radiator_temp_c, theme.sat_color(k).name())
        for line in self._lines:
            line.setValue(0.0)

    def _modulation_changed(self, name: str) -> None:
        self.modulation = name
        if self.result is not None:
            link = self.result.primary_link
            stride = max(1, math.ceil(self.result.n / MAX_PLOT_POINTS))
            x = (self.result.t_s / 3600.0)[::stride]
            self._ber_curve.setData(x, link.ber[name][::stride], connect="finite")
            self._thr_curve.setData(x, link.throughput_gbps[name][::stride], connect="finite")
            self.set_index(self._idx)
        self.modulation_changed.emit(name)

    _idx = 0

    # ------------------------------------------------------------------ per-frame update
    def set_index(self, i: int) -> None:
        res = self.result
        if res is None:
            return
        self._idx = i
        t_h = res.t_s[i] / 3600.0
        for line in self._lines:
            line.setValue(t_h)
        link, m = res.primary_link, self.modulation
        L = self._link_labels
        locked = bool(link.visible[i])
        L["status"].setText("LOCKED" if locked else "OCCLUDED")
        L["status"].setStyleSheet(f"color: {theme.GREEN if locked else theme.RED}; font-weight: bold;")
        L["range"].setText(_fmt(float(link.range_km[i]), ",.1f", "km"))
        L["rate"].setText(_fmt(float(link.range_rate_km_s[i]), "+.3f", "km/s"))
        L["latency"].setText(_fmt(float(link.latency_ms[i]), ".3f", "ms"))
        L["doppler"].setText(_fmt(float(link.doppler_ghz[i]), "+.3f", "GHz"))
        L["rx"].setText(_fmt(float(link.rx_power_dbm[i]), ".2f", "dBm"))
        L["snr"].setText(_fmt(float(link.snr_db[i]), ".2f", "dB"))
        L["ebn0"].setText(_fmt(float(link.eb_n0_db[m][i]), ".2f", "dB"))
        L["ber"].setText(_fmt_ber(float(link.ber[m][i])))
        L["thr"].setText(_fmt(float(link.throughput_gbps[m][i]), ".1f", "Gbps") if locked else "—")
        best = int(link.adaptive_modulation[i])
        names = list(MODULATIONS)
        L["adaptive"].setText(f"{names[best]} @ {link.adaptive_throughput_gbps[i]:.0f} Gbps" if best >= 0 else "—")
        L["fspl"].setText(_fmt(float(link.fspl_db[i]), ".2f", "dB"))
        L["div"].setText(_fmt(float(link.divergence_loss_db[i]), ".2f", "dB"))
        L["point"].setText(_fmt(link.pointing_loss_db, ".2f", "dB"))
        L["clear"].setText(_fmt(float(link.clearance_km[i]), "+,.0f", "km"))
        a, b = res.primary_pair
        turrets = {a: link.pointing_a, b: link.pointing_b}
        for k, tr in enumerate(res.tracks):
            d = self._sat_labels[k]
            d["lat"].setText(f"{tr.lat_deg[i]:+.3f}°")
            d["lon"].setText(f"{tr.lon_deg[i]:+.3f}°")
            d["alt"].setText(f"{tr.alt_km[i]:.1f} km")
            d["speed"].setText(f"{np.linalg.norm(tr.v_eci[i]):.3f} km/s")
            state = int(tr.eclipse[i])
            d["ecl"].setText(f"{oe.ECLIPSE_NAMES[state]} {100 * tr.sun_visibility[i]:.0f}%")
            d["ecl"].setStyleSheet(f"color: {ECLIPSE_COLORS[state]};")
            d["beta"].setText(f"{tr.beta_deg[i]:+.1f}°")
            d["temp"].setText(f"{tr.radiator_temp_c[i]:.1f} °C")
            d["pwr"].setText(f"{tr.solar_power_w[i] / 1000:.1f} kW")
            tp = turrets.get(k)
            if tp is not None:
                d["az"].setText(f"{tp.azimuth_deg[i]:.2f}°")
                d["el"].setText(f"{tp.elevation_deg[i]:+.2f}°")
                d["slew"].setText(f"{tp.slew_rate_deg_s[i]:.4f} °/s")
                d["azr"].setText(f"{tp.azimuth_rate_deg_s[i]:+.4f} / {tp.elevation_rate_deg_s[i]:+.4f}")
            else:
                for key in ("az", "el", "slew", "azr"):
                    d[key].setText("n/a")
