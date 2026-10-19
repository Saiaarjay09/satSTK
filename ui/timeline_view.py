"""Bottom Gantt chart: access (line-of-sight) windows and eclipse intervals over the whole scenario, with a draggable playhead."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from . import theme

LABEL_W = 300
ROW_H = 22
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
        a, b = result.primary_pair
        na, nb = result.tracks[a].name, result.tracks[b].name
        rows.append((f"Access  {na} ↔ {nb}", [(s, e, QColor(theme.GREEN)) for s, e in result.access_windows[(a, b)]], QColor(theme.GREEN)))
        for k, tr in enumerate(result.tracks):
            ecl = result.eclipse_windows[k]
            bars = [(s, e, QColor(theme.AMBER)) for s, e in ecl["penumbra"]] + [(s, e, QColor(theme.UMBRA)) for s, e in ecl["umbra"]]
            rows.append((f"Eclipse  {tr.name}", bars, theme.sat_color(k)))
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
            QToolTip.showText(ev.globalPosition().toPoint(), f"{self.result.utc(t):%H:%M:%S} UTC  (T+{t/3600:.2f} h)\nclick or drag to jump", self)

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

        for n, (label, bars, accent) in enumerate(self._rows):
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
        for text, col, w in (("access", theme.GREEN, 62), ("penumbra", theme.AMBER, 76), ("umbra", theme.UMBRA, 60)):
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
