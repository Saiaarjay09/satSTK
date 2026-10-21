"""
The radiation overlay for the 2D map and the 3D globe: a glowing outline and hatch over the South Atlantic Anomaly box and two auroral rings at
+/-60 degrees latitude. The zones are the simulation assumptions in seu_injector.py (a box and two latitude lines), drawn exactly as defined.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen, QPolygonF

import seu_injector as si

SAA_COLOR = (255, 120, 40)
POLAR_COLOR = (150, 90, 255)


def pulse(flux_mult: float, phase: float = 0.0) -> float:
    """0.45 .. 1.0 brightness: brighter the higher the current flux (x1 -> dim, x1000 -> full), with a slow breathing."""
    base = 0.45 + 0.55 * min(1.0, math.log10(max(flux_mult, 1.0)) / 3.0)
    return base * (0.9 + 0.1 * math.sin(phase))


def saa_outline(n_per_edge: int = 40):
    """The SAA box boundary as (lon, lat) arrays, closed."""
    (la0, la1), (lo0, lo1) = si.SAA_LAT_DEG, si.SAA_LON_DEG
    lon = np.concatenate([np.linspace(lo0, lo1, n_per_edge), np.full(n_per_edge, lo1), np.linspace(lo1, lo0, n_per_edge), np.full(n_per_edge, lo0)])
    lat = np.concatenate([np.full(n_per_edge, la0), np.linspace(la0, la1, n_per_edge), np.full(n_per_edge, la1), np.linspace(la1, la0, n_per_edge)])
    return np.append(lon, lon[0]), np.append(lat, lat[0])


def glow_pens(rgb, strength: float):
    """Three strokes, wide and faint to thin and bright, that read as a glow."""
    return [QPen(QColor(*rgb, int(a * strength)), w) for w, a in ((9.0, 36), (4.5, 80), (1.6, 235))]


def draw_on_map(p, box: QRectF, strength: float, label_font=None) -> None:
    """Draw the zones on the equirectangular map (`box` is the map rectangle)."""
    def x_of(lon): return box.left() + (lon + 180.0) / 360.0 * box.width()
    def y_of(lat): return box.top() + (90.0 - lat) / 180.0 * box.height()
    p.save()
    p.setClipRect(box)
    # auroral zones
    for lo, hi in ((si.POLAR_LAT_DEG, 90.0), (-90.0, -si.POLAR_LAT_DEG)):
        r = QRectF(box.left(), y_of(hi), box.width(), y_of(lo) - y_of(hi))
        p.fillRect(r, QColor(*POLAR_COLOR, int(46 * strength)))
    for lat in (si.POLAR_LAT_DEG, -si.POLAR_LAT_DEG):
        for pen in glow_pens(POLAR_COLOR, strength):
            pen.setStyle(Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawLine(QPointF(box.left(), y_of(lat)), QPointF(box.right(), y_of(lat)))
    # the SAA
    (la0, la1), (lo0, lo1) = si.SAA_LAT_DEG, si.SAA_LON_DEG
    r = QRectF(x_of(lo0), y_of(la1), x_of(lo1) - x_of(lo0), y_of(la0) - y_of(la1))
    p.fillRect(r, QColor(*SAA_COLOR, int(70 * strength)))
    p.setPen(QPen(QColor(*SAA_COLOR, int(55 * strength)), 1))
    for lat in np.arange(la0 + 5.0, la1, 5.0):
        p.drawLine(QPointF(r.left(), y_of(lat)), QPointF(r.right(), y_of(lat)))
    for pen in glow_pens(SAA_COLOR, strength):
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
    p.setPen(QColor(*SAA_COLOR, 255))
    if label_font is not None:
        p.setFont(label_font)
    p.drawText(QPointF(r.left() + 6, r.top() + 14), f"SAA x{si.SAA_MULT:g}")
    p.setPen(QColor(*POLAR_COLOR, 255))
    p.drawText(QPointF(box.left() + 6, y_of(90.0) + 14), f"auroral zone x{si.POLAR_MULT:g}")
    p.drawText(QPointF(box.left() + 6, y_of(-si.POLAR_LAT_DEG) + 14), f"auroral zone x{si.POLAR_MULT:g}")
    p.restore()


def _xyz(lon_deg, lat_deg, radius_km: float, gmst: float) -> np.ndarray:
    """Points on a sphere (Earth-fixed lon/lat) in the inertial frame at the given sidereal angle."""
    lo, la = np.radians(np.asarray(lon_deg, float)) + gmst, np.radians(np.asarray(lat_deg, float))
    return radius_km * np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) + 0 * lo], axis=1)


def draw_on_globe(globe, p, eye, right, up, sc, gmst: float, strength: float, radius_km: float) -> None:
    """Draw the zones on the 3D globe using its own polyline helper (hidden far-side points are skipped by it)."""
    r = radius_km * 1.003
    lon, lat = saa_outline()
    for pen in glow_pens(SAA_COLOR, strength):
        globe._polyline(p, _xyz(lon, lat, r, gmst), eye, right, up, sc, pen)
    (la0, la1), (lo0, lo1) = si.SAA_LAT_DEG, si.SAA_LON_DEG
    hatch = QPen(QColor(*SAA_COLOR, int(80 * strength)), 1.0)
    for la in np.arange(la0 + 5.0, la1, 5.0):
        globe._polyline(p, _xyz(np.linspace(lo0, lo1, 30), np.full(30, la), r, gmst), eye, right, up, sc, hatch)
    ring = np.linspace(-180.0, 180.0, 181)
    for la in (si.POLAR_LAT_DEG, -si.POLAR_LAT_DEG):
        for pen in glow_pens(POLAR_COLOR, strength):
            globe._polyline(p, _xyz(ring, np.full_like(ring, la), r, gmst), eye, right, up, sc, pen)
