"""
3D globe view, drawn in software (NumPy ray-casting + QPainter) so it works on every machine and in headless tests - no OpenGL.

* The Earth is a shaded sphere: land from the Natural Earth outlines, a smooth day/night terminator computed from the Sun
  direction, and the planet turns underneath the orbits because the camera is fixed in inertial space (like STK's ECI view).
* Orbit paths, satellite markers and the laser link line are projected with an orthographic camera. Anything behind the Earth's
  disc is hidden. The link line is green while the line of sight is clear and dashed red while the Earth blocks it.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QComboBox, QWidget

import orbital_engine as oe

from . import geo, theme

BACKGROUND = QColor("#04070b")
OCEAN_DAY = np.array([22, 78, 150], dtype=np.float32)
LAND_DAY = np.array([84, 146, 84], dtype=np.float32)
OCEAN_NIGHT = np.array([4, 9, 24], dtype=np.float32)
LAND_NIGHT = np.array([16, 26, 22], dtype=np.float32)
TWILIGHT = np.array([150, 70, 20], dtype=np.float32)


class GlobeView(QWidget):
    camera_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.result = None
        self.idx = 0
        self.az = math.radians(-60.0)
        self.el = math.radians(28.0)
        self.zoom = 1.0
        self.follow: int | None = None            # satellite index to centre on, or None for the free inertial camera
        self._drag = None
        self._grid_key = None
        self._grid = None
        self._img_arr = None
        self._mask = None
        self._stars = np.random.default_rng(7).random((140, 3))
        self._combo = QComboBox(self)
        self._combo.addItem("Inertial camera (drag to rotate)")
        self._combo.currentIndexChanged.connect(self._camera_mode_changed)
        self._combo.setFixedWidth(250)

    # ------------------------------------------------------------------ public API
    def set_result(self, result) -> None:
        self.result = result
        self.idx = 0
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("Inertial camera (drag to rotate)")
        for tr in result.tracks:
            self._combo.addItem(f"Follow {tr.name}")
        self._combo.blockSignals(False)
        self.follow = None
        self.update()

    def set_index(self, idx: int) -> None:
        self.idx = idx
        self.update()

    # ------------------------------------------------------------------ camera
    def _camera_mode_changed(self, i: int) -> None:
        self.follow = None if i <= 0 else i - 1
        self.update()

    def resizeEvent(self, ev) -> None:
        self._combo.move(self.width() - self._combo.width() - 10, 10)

    def _camera_angles(self):
        if self.follow is not None and self.result is not None:
            r = self.result.tracks[self.follow].r_eci[self.idx]
            return math.atan2(r[1], r[0]), math.asin(r[2] / np.linalg.norm(r))
        return self.az, self.el

    def _basis(self):
        az, el = self._camera_angles()
        eye = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        right = np.cross([0.0, 0.0, 1.0], eye)
        right /= np.linalg.norm(right)
        up = np.cross(eye, right)
        return eye, right, up

    def _scale(self) -> float:
        """Pixels per kilometre: the default fits the satellites' orbit with some margin."""
        return 0.5 * min(self.width(), self.height()) / (1.28 * (oe.R_EARTH + 700.0)) * self.zoom

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self.follow is None:
            self._drag = ev.position()

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is not None:
            d = ev.position() - self._drag
            self._drag = ev.position()
            self.az -= math.radians(d.x() * 0.4)
            self.el = max(-1.5, min(1.5, self.el + math.radians(d.y() * 0.4)))
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, ev) -> None:
        self.az, self.el, self.zoom = math.radians(-60.0), math.radians(28.0), 1.0
        self.update()

    def wheelEvent(self, ev) -> None:
        self.zoom = max(0.4, min(8.0, self.zoom * 1.1 ** (ev.angleDelta().y() / 120.0)))
        self.update()

    # ------------------------------------------------------------------ Earth shading
    def _earth_grid(self, eye, right, up, d_px: int, r_px: float):
        """Per-pixel sphere normals for the current camera. Cached until the camera or the size changes."""
        key = (round(float(eye[0]), 4), round(float(eye[1]), 4), round(float(eye[2]), 4), d_px)
        if key == self._grid_key:
            return self._grid
        axis = (np.arange(d_px) + 0.5 - d_px / 2.0) / r_px
        x, y = np.meshgrid(axis, -axis)
        rr2 = x * x + y * y
        inside = rr2 <= 1.0
        z = np.sqrt(np.clip(1.0 - rr2, 0.0, None))
        flat = np.flatnonzero(inside.ravel())
        xs, ys, zs = x.ravel()[flat], y.ravel()[flat], z.ravel()[flat]
        n = xs[:, None] * right + ys[:, None] * up + zs[:, None] * eye          # unit normals in the inertial frame
        h, w = (self._mask.shape if self._mask is not None else (720, 1440))
        lat = np.arcsin(np.clip(n[:, 2], -1, 1))
        row = np.clip(((math.pi / 2 - lat) / math.pi * h).astype(np.int32), 0, h - 1)
        self._grid = (flat, n, np.arctan2(n[:, 1], n[:, 0]), row, zs)
        self._grid_key = key
        return self._grid

    def _render_earth(self, eye, right, up, d_px: int, r_px: float):
        if self._mask is None:
            self._mask = geo.land_mask()
        flat, n, lon0, row, zs = self._earth_grid(eye, right, up, d_px, r_px)
        h, w = self._mask.shape
        sun = self.result.sun_eci[self.idx]
        sun = sun / np.linalg.norm(sun)
        gmst = self.result.gmst[self.idx]
        col = (((lon0 - gmst + math.pi) / (2 * math.pi)) % 1.0 * w).astype(np.int32) % w
        land = self._mask[row, col][:, None]
        s = n @ sun
        day = np.clip((s + 0.10) / 0.20, 0.0, 1.0)
        day = (day * day * (3.0 - 2.0 * day))[:, None]
        rgb_day = OCEAN_DAY + land * (LAND_DAY - OCEAN_DAY)
        rgb_day = rgb_day * (0.70 + 0.30 * np.clip(s, 0.0, 1.0))[:, None]
        rgb_night = OCEAN_NIGHT + land * (LAND_NIGHT - OCEAN_NIGHT)
        rgb = rgb_night + day * (rgb_day - rgb_night)
        rgb += (np.exp(-(s / 0.06) ** 2) * 0.35)[:, None] * TWILIGHT                  # warm band along the terminator
        rgb *= (0.45 + 0.55 * np.sqrt(zs))[:, None]                                   # darken toward the limb
        arr = np.zeros((d_px * d_px, 4), dtype=np.uint8)
        arr[flat, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        arr[flat, 3] = 255
        self._img_arr = arr.reshape(d_px, d_px, 4)
        return QImage(self._img_arr.data, d_px, d_px, d_px * 4, QImage.Format.Format_RGBA8888)

    # ------------------------------------------------------------------ painting
    def _project(self, pts, eye, right, up, sc):
        pts = np.atleast_2d(pts)
        x, y, d = pts @ right, pts @ up, pts @ eye
        hidden = (d < 0) & (x * x + y * y < oe.R_EARTH ** 2)
        return self.width() / 2 + x * sc, self.height() / 2 - y * sc, hidden

    def _polyline(self, p: QPainter, pts, eye, right, up, sc, pen: QPen) -> None:
        sx, sy, hidden = self._project(pts, eye, right, up, sc)
        path, pen_down = QPainterPath(), False
        for k in range(len(sx)):
            if hidden[k]:
                pen_down = False
                continue
            if pen_down:
                path.lineTo(sx[k], sy[k])
            else:
                path.moveTo(sx[k], sy[k])
                pen_down = True
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), BACKGROUND)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        for fx, fy, fb in self._stars:
            p.setBrush(QColor(255, 255, 255, int(40 + 90 * fb)))
            p.drawEllipse(QPointF(fx * w, fy * h), 0.9, 0.9)
        if self.result is None:
            p.setPen(QColor(theme.DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Computing scenario ...")
            return

        res, i = self.result, self.idx
        eye, right, up = self._basis()
        sc = self._scale()
        cx, cy = w / 2.0, h / 2.0
        r_px = oe.R_EARTH * sc

        # atmosphere glow, then the shaded planet
        glow = QRadialGradient(QPointF(cx, cy), r_px * 1.10)
        glow.setColorAt(0.0, QColor(0, 0, 0, 0))
        glow.setColorAt(0.88, QColor(0, 0, 0, 0))
        glow.setColorAt(0.915, QColor(70, 160, 255, 110))
        glow.setColorAt(1.0, QColor(70, 160, 255, 0))
        p.setBrush(glow)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r_px * 1.10, r_px * 1.10)
        render_scale = min(1.0, 520.0 / (2.0 * r_px)) if r_px > 0 else 1.0
        d_px = max(8, int(2 * r_px * render_scale) + 2)
        img = self._render_earth(eye, right, up, d_px, r_px * render_scale)
        half = d_px / render_scale / 2.0
        p.drawImage(QRectF(cx - half, cy - half, 2 * half, 2 * half), img)

        # orbit paths: dim path ahead for one revolution, bright trail behind
        step = res.config.step_s
        n_rev = max(2, int(oe.orbital_period_s(res.tracks[0].elements0[0]) / step))
        stride = max(1, n_rev // 220)
        for k, tr in enumerate(res.tracks):
            c = theme.sat_color(k)
            j1 = min(res.n, i + n_rev + 1)
            ahead = QColor(c)
            ahead.setAlpha(95)
            self._polyline(p, tr.r_eci[i:j1:stride], eye, right, up, sc, QPen(ahead, 1.2))
            j0 = max(0, i - n_rev // 4)
            trail = QColor(c)
            trail.setAlpha(235)
            self._polyline(p, tr.r_eci[j0:i + 1:max(1, stride // 2)], eye, right, up, sc, QPen(trail, 2.2))

        # laser link line
        link = res.primary_link
        a, b = res.primary_pair
        pa, pb = res.tracks[a].r_eci[i], res.tracks[b].r_eci[i]
        locked = bool(link.visible[i])
        (xa, xb), (ya, yb), _ = self._project(np.array([pa, pb]), eye, right, up, sc)
        if locked:
            glow_pen = QPen(QColor(46, 204, 113, 70), 7)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(glow_pen)
            p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
            p.setPen(QPen(QColor(theme.GREEN), 2.0))
        else:
            pen = QPen(QColor(theme.RED), 1.6, Qt.PenStyle.DashLine)
            p.setPen(pen)
        p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
        mid = QPointF((xa + xb) / 2, (ya + yb) / 2)
        p.setFont(theme.mono_font(9))
        p.setPen(QColor(theme.GREEN if locked else theme.RED))
        p.drawText(mid + QPointF(6, -6), f"{link.range_km[i]:,.0f} km" + ("" if locked else "  (blocked)"))

        # satellites
        label_font = QFont(self.font())
        label_font.setPointSize(10)
        label_font.setBold(True)
        for k, tr in enumerate(res.tracks):
            sx, sy, hidden = self._project(tr.r_eci[i], eye, right, up, sc)
            c = theme.sat_color(k)
            if hidden[0]:
                c.setAlpha(80)
            p.setBrush(c)
            p.setPen(QPen(QColor("white"), 1.2) if not hidden[0] else Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(sx[0], sy[0]), 5.5, 5.5)
            p.setFont(label_font)
            p.setPen(c)
            p.drawText(QPointF(sx[0] + 9, sy[0] - 8), tr.name)

        # Sun marker on the edge of the view, in the direction the Sun lies
        sun = res.sun_eci[i] / np.linalg.norm(res.sun_eci[i])
        sx, sy = float(sun @ right), float(sun @ up)
        mag = math.hypot(sx, sy)
        ux, uy = (sx / mag, sy / mag) if mag > 0.05 else (0.7, 0.7)
        rad = min(w, h) * 0.46
        sp = QPointF(cx + ux * rad, cy - uy * rad)
        behind = float(sun @ eye) < 0
        p.setBrush(QColor(255, 210, 60, 110 if behind else 255))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(sp, 7, 7)
        p.setPen(QColor(255, 210, 60, 140 if behind else 255))
        p.setFont(theme.mono_font(9))
        p.drawText(sp + QPointF(11, 4), "Sun" + (" (far side)" if behind else ""))

        # heads-up text
        p.setFont(theme.mono_font(10))
        p.setPen(QColor(theme.TEXT))
        p.drawText(QPointF(12, 22), f"{res.utc(res.t_s[i]):%Y-%m-%d %H:%M:%S} UTC")
        p.setPen(QColor(theme.GREEN))
        p.drawText(QPointF(12, 42), "━ laser link locked")
        p.setPen(QColor(theme.RED))
        p.drawText(QPointF(12, 58), "┅ blocked by Earth")
        p.setPen(QColor(theme.DIM))
        p.setFont(theme.mono_font(8))
        p.drawText(QPointF(12, h - 10), "drag: rotate · wheel: zoom · double-click: reset · inertial view, Earth turns")
