"""Bottom Gantt chart: access (line-of-sight) windows and eclipse intervals over the whole scenario, with a draggable playhead."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

import orbital_engine as oe
from events import CATEGORIES, CATEGORY_COLORS

from . import theme

LABEL_W = 300
ROW_H = 20
AXIS_H = 24


class TimelineView(QWidget):
    seek_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.t = 0.0
        self._rows = []        # (label, [(start, end, color)], accent)
        self._drag = False
        self.setMouseTracking(True)
        self.setMinimumHeight(AXIS_H + ROW_H * 3 + 8)

    def set_result(self, result) -> None:
        self.result = result
        self.t = 0.0
        rows = []
        tgt = result.target
        link = result.primary_link
        above = oe.mask_to_intervals(link.visible, result.t_s)
        rows.append((f"Above {result.config.ground.min_elevation_deg:g}° mask  ({result.config.ground.name})", [(s_, e_, QColor("#1f7a8c")) for s_, e_ in above], QColor("#4cc9f0"), None))
        rows.append((f"Beam on {result.tracks[tgt].name}", [(s_, e_, QColor(theme.GREEN)) for s_, e_ in result.primary_windows], QColor(theme.GREEN), None))
        for k, tr in enumerate(result.tracks):
            ecl = result.eclipse_windows[k]
            bars = [(s_, e_, QColor(theme.AMBER)) for s_, e_ in ecl["penumbra"]] + [(s_, e_, QColor(theme.UMBRA)) for s_, e_ in ecl["umbra"]]
            rows.append((f"Eclipse  {tr.name}", bars, theme.sat_color(k), None))
        for cat in CATEGORIES:
            evs = [e for e in result.events if e.category == cat]
            if evs:
                rows.append((f"Events: {cat}", [(e.start_s, e.end_s, QColor(CATEGORY_COLORS[cat])) for e in evs], QColor(CATEGORY_COLORS[cat]), evs))
        self._rows = rows
        self.setMinimumHeight(AXIS_H + ROW_H * len(rows) + 8)
        self.update()

    def set_time(self, t: float) -> None:
        self.t = t
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(LABEL_W, AXIS_H, max(10, self.width() - LABEL_W - 12), ROW_H * len(self._rows))

    def _time_at(self, x: float) -> float:
        r = self._plot_rect()
        frac = min(max((x - r.left()) / r.width(), 0.0), 1.0)
        return frac * self.result.t_s[-1]

    def mousePressEvent(self, ev) -> None:
        if self.result is not None and ev.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self.seek_requested.emit(self._time_at(ev.position().x()))

    def mouseMoveEvent(self, ev) -> None:
        if self.result is None:
            return
        t = self._time_at(ev.position().x())
        if self._drag:
            self.seek_requested.emit(t)
        else:
            text = f"{self.result.utc(t):%H:%M:%S} UTC  (T+{t/3600:.2f} h)\nclick or drag to jump"
            slack = self.result.t_s[-1] / max(1.0, self._plot_rect().width()) * 3.0          # about 3 pixels either side
            hits = [e for e in self.result.events if e.start_s - slack <= t <= e.end_s + slack]
            if hits:
                text += "\n" + "\n".join(f"* {e.name}: {e.duration_s:.0f} s" + (f", +{e.loss_db:.0f} dB" if e.loss_db else "") + (", beam lost" if e.outage else "") for e in hits[:5])
            QToolTip.showText(ev.globalPosition().toPoint(), text, self)

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = False

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        if self.result is None:
            return
        r = self._plot_rect()
        total = float(self.result.t_s[-1])
        px = lambda t: r.left() + t / total * r.width()

        # hour grid and axis labels
        p.setFont(theme.mono_font(9))
        hours = total / 3600.0
        tick_h = 1 if hours <= 30 else 6 if hours <= 100 else 24
        h = 0.0
        while h <= hours + 1e-9:
            x = px(h * 3600.0)
            p.setPen(QPen(QColor(255, 255, 255, 22), 1))
            p.drawLine(QPointF(x, AXIS_H - 4), QPointF(x, r.bottom()))
            if h % (tick_h * (3 if hours <= 30 else 1)) < 1e-9:
                p.setPen(QColor(theme.DIM))
                p.drawText(QPointF(x - 14, AXIS_H - 8), f"{self.result.utc(h * 3600.0):%H:%M}")
            h += tick_h

        for n, (label, bars, accent, _evs) in enumerate(self._rows):
            y = r.top() + n * ROW_H
            p.fillRect(QRectF(r.left(), y + 1, r.width(), ROW_H - 2), QColor(255, 255, 255, 10 if n % 2 == 0 else 5))
            p.setPen(accent)
            p.drawText(QRectF(8, y, LABEL_W - 12, ROW_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            for s, e, c in bars:
                x0, x1 = px(s), px(e)
                p.fillRect(QRectF(x0, y + 4, max(1.5, x1 - x0), ROW_H - 8), c)
            if not bars:
                p.setPen(QColor(theme.DIM))
                p.drawText(QRectF(r.left() + 6, y, 200, ROW_H), Qt.AlignmentFlag.AlignVCenter, "none in this scenario")

        # legend
        p.setFont(theme.mono_font(8))
        lx = 8.0
        for text, col, w in (("beam", theme.GREEN, 50), ("penumbra", theme.AMBER, 76), ("umbra", theme.UMBRA, 60)):
            p.fillRect(QRectF(lx, 8, 9, 9), QColor(col))
            p.setPen(QColor(theme.DIM))
            p.drawText(QPointF(lx + 13, 16), text)
            lx += w

        # playhead
        x = px(self.t)
        p.setPen(QPen(QColor(theme.RED), 2))
        p.drawLine(QPointF(x, AXIS_H - 6), QPointF(x, r.bottom()))
        p.setBrush(QColor(theme.RED))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon([QPointF(x - 5, AXIS_H - 9), QPointF(x + 5, AXIS_H - 9), QPointF(x, AXIS_H - 2)])
