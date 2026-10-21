"""
2D map: equirectangular world map with the day/night shading, sub-satellite ground tracks, coverage footprints, the sub-solar
point and the laser link between the two sub-satellite points .
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

import orbital_engine as oe

from . import geo, radiation_overlay, theme

OCEAN = QColor("#0d2238")
LAND = QColor("#2c4a3a")
LAND_EDGE = QColor("#4d7a62")


def footprint_polygon(lat_deg: float, lon_deg: float, rho_rad: float, n: int = 90):
    """
    Outline (lon, lat in degrees) of the circle of angular radius `rho_rad` around a sub-satellite point.
    Longitudes are unwrapped so the outline is continuous; if the circle encloses a pole the outline is closed through the pole.
    """
    lat0, lon0 = math.radians(lat_deg), math.radians(lon_deg)
    az = np.linspace(0.0, 2.0 * math.pi, n + 1)
    lat = np.arcsin(np.clip(math.sin(lat0) * math.cos(rho_rad) + math.cos(lat0) * math.sin(rho_rad) * np.cos(az), -1, 1))
    dlon = np.unwrap(np.arctan2(np.sin(az) * math.sin(rho_rad) * math.cos(lat0),
                                math.cos(rho_rad) - math.sin(lat0) * np.sin(lat)))
    lon = np.degrees(lon0 + dlon)
    lat = np.degrees(lat)
    if abs(dlon[-1] - dlon[0]) > math.pi:                      # the circle contains a pole
        pole = 90.0 if lat_deg >= 0 else -90.0
        lon = np.append(lon, [lon[-1], lon[0]])
        lat = np.append(lat, [pole, pole])
    return lon, lat


class MapView(QWidget):
    station_move_requested = Signal(float, float)     # Ctrl/Cmd + click on the map

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.result = None
        self.idx = 0
        self._cursor = None
        self._land_cache = (None, None)
        self._track_cache = (None, None)
        self._night_arr = None
        self.rad_overlay = False                    # SAA / auroral overlay (set by the Radiation-Tolerant AI Scheduler tab)
        self.rad_strength = 0.6
        self._lat_c = np.radians(89.5 - np.arange(180))
        self._lon_c = np.radians(-179.5 + np.arange(360))

    def set_result(self, result) -> None:
        self.result = result
        self.idx = 0
        self._track_cache = (None, None)
        self.update()

    def set_index(self, idx: int) -> None:
        self.idx = idx
        self.update()

    def set_radiation(self, on: bool, flux_mult: float = 1.0) -> None:
        self.rad_overlay, self.rad_strength = bool(on), radiation_overlay.pulse(flux_mult)
        self.update()

    # ------------------------------------------------------------------ geometry helpers
    def _box(self) -> QRectF:
        w, h = self.width(), self.height()
        mw = min(w - 8, 2.0 * (h - 8))
        return QRectF((w - mw) / 2.0, (h - mw / 2.0) / 2.0, mw, mw / 2.0)

    @staticmethod
    def _xy(box: QRectF, lon, lat):
        return box.left() + (np.asarray(lon) + 180.0) / 360.0 * box.width(), box.top() + (90.0 - np.asarray(lat)) / 180.0 * box.height()

    def mouseMoveEvent(self, ev) -> None:
        box = self._box()
        pos = ev.position()
        if box.contains(pos):
            self._cursor = (90.0 - (pos.y() - box.top()) / box.height() * 180.0, (pos.x() - box.left()) / box.width() * 360.0 - 180.0)
        else:
            self._cursor = None
        self.update()

    def mousePressEvent(self, ev) -> None:
        mods = ev.modifiers()
        if ev.button() == Qt.MouseButton.LeftButton and self._cursor and (mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)):
            self.station_move_requested.emit(round(self._cursor[0], 3), round(self._cursor[1], 3))

    def leaveEvent(self, ev) -> None:
        self._cursor = None
        self.update()

    def _night_image(self, box: QRectF) -> QImage:
        res, i = self.result, self.idx
        sun = res.sun_eci[i] / np.linalg.norm(res.sun_eci[i])
        g = res.gmst[i]
        sun_e = np.array([math.cos(g) * sun[0] + math.sin(g) * sun[1], -math.sin(g) * sun[0] + math.cos(g) * sun[1], sun[2]])
        lat, lon = np.meshgrid(self._lat_c, self._lon_c, indexing="ij")
        s = np.cos(lat) * np.cos(lon) * sun_e[0] + np.cos(lat) * np.sin(lon) * sun_e[1] + np.sin(lat) * sun_e[2]
        day = np.clip((s + 0.10) / 0.20, 0.0, 1.0)
        day = day * day * (3 - 2 * day)
        arr = np.zeros((180, 360, 4), dtype=np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = 0, 4, 18
        arr[..., 3] = ((1.0 - day) * 165).astype(np.uint8)
        self._night_arr = arr
        return QImage(arr.data, 360, 180, 360 * 4, QImage.Format.Format_RGBA8888)

    def _path_from_lonlat(self, box: QRectF, lon, lat) -> QPainterPath:
        """Polyline in map pixels; the pen lifts at the antimeridian so a track does not streak across the whole map."""
        x, y = self._xy(box, lon, lat)
        path = QPainterPath()
        if len(x) == 0:
            return path
        path.moveTo(x[0], y[0])
        jumps = np.flatnonzero(np.abs(np.diff(lon)) > 180.0)
        last = 0
        for j in jumps:
            for k in range(last + 1, j + 1):
                path.lineTo(x[k], y[k])
            path.moveTo(x[j + 1], y[j + 1])
            last = j + 1
        for k in range(last + 1, len(x)):
            path.lineTo(x[k], y[k])
        return path

    # ------------------------------------------------------------------ painting
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(theme.BG))
        if self.result is None:
            p.setPen(QColor(theme.DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Computing scenario ...")
            return
        res, i = self.result, self.idx
        box = self._box()
        p.fillRect(box, OCEAN)

        key = (round(box.width()), round(box.height()))
        if self._land_cache[0] != key:
            lp = geo.land_path(box.width(), box.height())
            lp.translate(box.left(), box.top())
            self._land_cache = (key, lp)
        p.setPen(QPen(LAND_EDGE, 0.8))
        p.setBrush(LAND)
        p.drawPath(self._land_cache[1])
        p.drawImage(box, self._night_image(box))
        if self.rad_overlay:
            radiation_overlay.draw_on_map(p, box, self.rad_strength, theme.mono_font(8))

        # graticule
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for lon in range(-150, 180, 30):
            x, _ = self._xy(box, lon, 0)
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for lat in range(-60, 90, 30):
            _, y = self._xy(box, 0, lat)
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        p.drawLine(QPointF(box.left(), box.center().y()), QPointF(box.right(), box.center().y()))
        p.setPen(QColor(theme.DIM))
        p.setFont(theme.mono_font(8))
        for lon in range(-180, 181, 60):
            x, _ = self._xy(box, lon, 0)
            p.drawText(QPointF(x + 2, box.bottom() - 3), f"{lon}°")
        for lat in range(-60, 91, 30):
            _, y = self._xy(box, 0, lat)
            p.drawText(QPointF(box.left() + 3, y - 2), f"{lat}°")

        # ground tracks (whole scenario faint, last orbit bright)
        step = res.config.step_s
        stride = max(1, int(round(30.0 / step)))
        n_rev = int(oe.orbital_period_s(res.tracks[0].elements0[0]) / step)
        if self._track_cache[0] != key:
            full = [self._path_from_lonlat(box, tr.lon_deg[::stride], tr.lat_deg[::stride]) for tr in res.tracks]
            self._track_cache = (key, full)
        for k, tr in enumerate(res.tracks):
            c = theme.sat_color(k)
            faint = QColor(c)
            faint.setAlpha(38)
            p.setPen(QPen(faint, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._track_cache[1][k])
            j0 = max(0, i - n_rev)
            p.setPen(QPen(c, 1.8))
            p.drawPath(self._path_from_lonlat(box, tr.lon_deg[j0:i + 1], tr.lat_deg[j0:i + 1]))

        # footprints (coverage circle at 0 degrees elevation) and markers
        for k, tr in enumerate(res.tracks):
            c = theme.sat_color(k)
            r = float(np.linalg.norm(tr.r_eci[i]))
            rho = math.acos(min(1.0, oe.R_EARTH / r))
            lon, lat = footprint_polygon(float(tr.lat_deg[i]), float(tr.lon_deg[i]), rho)
            fill = QColor(c)
            fill.setAlpha(38)
            p.setBrush(fill)
            p.setPen(QPen(c, 1.2))
            p.save()
            p.setClipRect(box)
            for shift in (-360.0, 0.0, 360.0):
                x, y = self._xy(box, lon + shift, lat)
                p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in zip(x, y)]))
            p.restore()

        # sub-solar point
        sun = res.sun_eci[i]
        dec = math.degrees(math.asin(sun[2] / np.linalg.norm(sun)))
        ra = math.degrees(math.atan2(sun[1], sun[0]))
        slon = (ra - math.degrees(res.gmst[i]) + 180.0) % 360.0 - 180.0
        sx, sy = self._xy(box, slon, dec)
        p.setBrush(QColor(255, 210, 60))
        p.setPen(QPen(QColor(255, 240, 160), 1.2))
        p.drawEllipse(QPointF(float(sx), float(sy)), 6, 6)
        p.setPen(QColor(255, 210, 60))
        p.drawText(QPointF(float(sx) + 9, float(sy) + 4), "Sun")

        # ground station, its elevation-mask coverage circle, and the beam to the sub-satellite point of the target
        gs = res.config.ground
        tgt = res.target
        state, reason = res.beam_state(i)
        route = res.route
        n_hops = int(route.kind[i]) if route is not None else 1
        if state == "blocked":
            beam_color = QColor(theme.RED)
        elif state == "degraded":
            beam_color = QColor(theme.AMBER)
        else:
            beam_color = QColor("#00e5ff") if n_hops >= 2 else QColor("#39ff14")
        alt = float(np.mean(res.tracks[tgt].alt_km))
        mask_lon, mask_lat = footprint_polygon(gs.lat_deg, gs.lon_deg, math.radians(oe.horizon_cap_radius_deg(alt, gs.min_elevation_deg)))
        p.save()
        p.setClipRect(box)
        p.setPen(QPen(QColor(theme.GREEN), 1.2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(46, 204, 113, 22))
        for shift in (-360.0, 0.0, 360.0):
            x, y = self._xy(box, mask_lon + shift, mask_lat)
            p.drawPolygon(QPolygonF([QPointF(a_, b_) for a_, b_ in zip(x, y)]))
        p.restore()
        tl, tn = res.tracks[tgt].lat_deg[i], res.tracks[tgt].lon_deg[i]
        gx, gy = self._xy(box, gs.lon_deg, gs.lat_deg)

        # GEO relay sub-points (fixed over the equator)
        for rt in res.relay_tracks:
            rx_, ry_ = self._xy(box, rt.lon_deg, 0.0)
            rx_, ry_ = float(rx_), float(ry_)
            p.setBrush(QColor("#ffd166"))
            p.setPen(QPen(QColor("white"), 1.1))
            p.drawPolygon(QPolygonF([QPointF(rx_, ry_ - 6), QPointF(rx_ + 6, ry_), QPointF(rx_, ry_ + 6), QPointF(rx_ - 6, ry_)]))
            p.setPen(QColor("#ffd166"))
            p.setFont(theme.mono_font(8))
            p.drawText(QPointF(rx_ + 8, ry_ - 6), rt.short)

        # the beam along its route (lines take the short way round the globe; drawn three times to cover the map edge)
        chain = [(gs.lon_deg, gs.lat_deg)]
        if state != "blocked" and route is not None:
            chain += [(res.relay_tracks[k].lon_deg, 0.0) for k in route.nodes(i)]
        chain.append((float(tn), float(tl)))
        p.save()
        p.setClipRect(box)
        p.setPen(QPen(beam_color, 1.8, Qt.PenStyle.DashLine if state == "blocked" else Qt.PenStyle.SolidLine))
        for (lo1, la1), (lo2, la2) in zip(chain[:-1], chain[1:]):
            dlon = (lo2 - lo1 + 180.0) % 360.0 - 180.0
            for shift in (-360.0, 0.0, 360.0):
                x1, y1 = self._xy(box, lo1 + shift, la1)
                x2, y2 = self._xy(box, lo1 + dlon + shift, la2)
                p.drawLine(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))
        p.restore()
        p.setBrush(QColor(theme.GREEN))
        p.setPen(QPen(QColor("white"), 1.4))
        gx, gy = float(gx), float(gy)
        p.drawPolygon(QPolygonF([QPointF(gx, gy - 8), QPointF(gx - 6.5, gy + 4.5), QPointF(gx + 6.5, gy + 4.5)]))
        p.setPen(QColor(theme.GREEN))
        p.setFont(theme.mono_font(9))
        p.drawText(QPointF(gx + 9, gy + 14), gs.name)

        label_font = p.font()
        label_font.setBold(True)
        label_font.setPointSize(10)
        for k, tr in enumerate(res.tracks):
            x, y = self._xy(box, tr.lon_deg[i], tr.lat_deg[i])
            c = theme.sat_color(k)
            p.setBrush(c)
            p.setPen(QPen(QColor("white"), 1.2))
            p.drawEllipse(QPointF(float(x), float(y)), 5, 5)
            p.setFont(label_font)
            p.setPen(c)
            p.drawText(QPointF(float(x) + 9, float(y) - 7), tr.name)

        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        p.setFont(theme.mono_font(9))
        p.setPen(QColor(theme.TEXT))
        txt = f"{res.utc(res.t_s[i]):%H:%M:%S} UTC"
        if self._cursor:
            txt += f"    cursor {self._cursor[0]:+6.2f}°, {self._cursor[1]:+7.2f}°"
        p.drawText(QRectF(box.left() + 6, box.bottom() - 22, box.width() - 12, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, txt)
