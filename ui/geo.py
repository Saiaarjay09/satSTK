"""Land outline data (Natural Earth 110 m, public domain) and a land/sea raster built from it with Qt's own painter."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPolygonF

# In a PyInstaller build the data folder sits next to the bundled modules (sys._MEIPASS); from source it is next to this package.
_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
LAND_FILE = _ROOT / "data" / "land_110m.json"


@lru_cache(maxsize=1)
def land_rings() -> tuple:
    """Outer rings of the land polygons as (n,2) arrays of [longitude, latitude] degrees. Empty if the data file is missing."""
    try:
        with open(LAND_FILE) as fh:
            return tuple(np.array(r, dtype=float) for r in json.load(fh)["rings"])
    except (OSError, ValueError, KeyError):
        return ()


def land_path(width: float, height: float) -> QPainterPath:
    """All land polygons as one painter path in an equirectangular box of the given size (lon -180..180, lat 90..-90)."""
    path = QPainterPath()
    for ring in land_rings():
        pts = [QPointF((lon + 180.0) / 360.0 * width, (90.0 - lat) / 180.0 * height) for lon, lat in ring]
        if len(pts) > 2:
            path.addPolygon(QPolygonF(pts))
            path.closeSubpath()
    return path


@lru_cache(maxsize=2)
def land_mask(width: int = 1440, height: int = 720) -> np.ndarray:
    """(height,width) float32 array, 1 on land and 0 at sea, row 0 = 90 N, column 0 = 180 W."""
    img = QImage(width, height, QImage.Format.Format_Grayscale8)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(255, 255, 255))
    p.setBrush(QColor(255, 255, 255))
    p.drawPath(land_path(width, height))
    p.end()
    stride = img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(height, stride)[:, :width]
    return (arr.astype(np.float32) / 255.0).copy()
