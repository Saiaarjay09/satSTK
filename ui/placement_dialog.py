"""'Place the beam origin' dialog: click a map (or type coordinates, or pick a known optical ground station), then start the simulation."""
from __future__ import annotations

import copy
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QVBoxLayout, QWidget)

import orbital_engine as oe
from simulation import GROUND_STATION_PRESETS

from . import geo, theme
from .map_view import footprint_polygon


class MapPicker(QWidget):
    """A small world map. Click (or drag) to choose a point; the circle shows how far from the station a 600 km satellite is above the elevation mask."""
    picked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 330)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.lat, self.lon, self.radius_deg = 28.3, -16.5, 10.8

    def set_point(self, lat: float, lon: float, radius_deg: float) -> None:
        self.lat, self.lon, self.radius_deg = lat, lon, radius_deg
        self.update()

    def _box(self) -> QRectF:
        w, h = self.width(), self.height()
        mw = min(w - 8, 2.0 * (h - 8))
        return QRectF((w - mw) / 2.0, (h - mw / 2.0) / 2.0, mw, mw / 2.0)

    def _pick(self, pos) -> None:
        box = self._box()
        if not box.contains(pos):
            return
        lat = 90.0 - (pos.y() - box.top()) / box.height() * 180.0
        lon = (pos.x() - box.left()) / box.width() * 360.0 - 180.0
        self.picked.emit(round(lat, 3), round(lon, 3))

    def mousePressEvent(self, ev) -> None:
        self._pick(ev.position())

    def mouseMoveEvent(self, ev) -> None:
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._pick(ev.position())

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))
        box = self._box()
        p.fillRect(box, QColor("#0d2238"))
        path = geo.land_path(box.width(), box.height())
        path.translate(box.left(), box.top())
        p.setPen(QPen(QColor("#4d7a62"), 0.8))
        p.setBrush(QColor("#2c4a3a"))
        p.drawPath(path)
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for lon in range(-150, 180, 30):
            x = box.left() + (lon + 180) / 360 * box.width()
            p.drawLine(QPointF(x, box.top()), QPointF(x, box.bottom()))
        for lat in range(-60, 90, 30):
            y = box.top() + (90 - lat) / 180 * box.height()
            p.drawLine(QPointF(box.left(), y), QPointF(box.right(), y))
        x = box.left() + (self.lon + 180) / 360 * box.width()
        y = box.top() + (90 - self.lat) / 180 * box.height()
        p.setPen(QPen(QColor(theme.GREEN), 1.4, Qt.PenStyle.DashLine))
        p.setBrush(QColor(46, 204, 113, 30))
        p.save()
        p.setClipRect(box)
        flon, flat = footprint_polygon(self.lat, self.lon, math.radians(self.radius_deg))
        for shift in (-360.0, 0.0, 360.0):
            p.drawPolygon(QPolygonF([QPointF(box.left() + (lo + shift + 180) / 360 * box.width(), box.top() + (90 - la) / 180 * box.height())
                                     for lo, la in zip(flon, flat)]))
        p.restore()
        p.setBrush(QColor(theme.GREEN))
        p.setPen(QPen(QColor("white"), 1.5))
        p.drawPolygon([QPointF(x, y - 9), QPointF(x - 7, y + 5), QPointF(x + 7, y + 5)])
        p.setPen(QColor(theme.DIM))
        p.setFont(theme.mono_font(8))
        p.drawText(QPointF(box.left() + 4, box.bottom() - 4), "click anywhere to place the beam origin")


class PlacementDialog(QDialog):
    def __init__(self, station: oe.GroundStation, satellite_altitude_km: float = 600.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Place the laser ground station (beam origin)")
        self.altitude_km = satellite_altitude_km
        root = QVBoxLayout(self)
        root.addWidget(QLabel("The laser beam starts at one fixed place on the ground and is aimed at the satellite. Click the map to place it, "
                              "type coordinates, or pick a known optical ground station. The green circle is the area in which a satellite is "
                              "above the minimum elevation.", wordWrap=True))
        self.map = MapPicker()
        root.addWidget(self.map, 1)

        form = QFormLayout()
        self.preset = QComboBox()
        self.preset.addItem("Custom location")
        self.preset.addItems(list(GROUND_STATION_PRESETS))
        form.addRow("Known site", self.preset)
        self.name = QLineEdit(station.name)
        form.addRow("Name", self.name)
        row = QHBoxLayout()
        self.lat, self.lon, self.alt, self.minel = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sp, lo, hi, dec, suf, val in ((self.lat, -90, 90, 3, "° N", station.lat_deg), (self.lon, -180, 180, 3, "° E", station.lon_deg),
                                          (self.alt, -400, 9000, 0, " m", station.alt_m), (self.minel, 0, 80, 0, "°", station.min_elevation_deg)):
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSuffix(suf)
            sp.setValue(val)
        for label, sp in (("Latitude", self.lat), ("Longitude", self.lon), ("Altitude", self.alt), ("Min. elevation", self.minel)):
            row.addWidget(QLabel(label))
            row.addWidget(sp)
        form.addRow(row)
        root.addLayout(form)

        bb = QDialogButtonBox()
        self.ok = bb.addButton("Start simulation", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self.map.picked.connect(self._from_map)
        self.preset.currentTextChanged.connect(self._from_preset)
        for sp in (self.lat, self.lon, self.minel):
            sp.valueChanged.connect(self._sync_map)
        self._sync_map()
        self.resize(900, 640)

    def _sync_map(self) -> None:
        self.map.set_point(self.lat.value(), self.lon.value(), oe.horizon_cap_radius_deg(self.altitude_km, self.minel.value()))

    def _from_map(self, lat: float, lon: float) -> None:
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(0)
        self.preset.blockSignals(False)
        self.lat.setValue(lat)
        self.lon.setValue(lon)
        if self.name.text() in ("", "Ground station") or self.name.text() in [g.name for g in GROUND_STATION_PRESETS.values()]:
            self.name.setText("Ground station")

    def _from_preset(self, text: str) -> None:
        gs = GROUND_STATION_PRESETS.get(text)
        if gs is None:
            return
        self.name.setText(gs.name)
        self.lat.setValue(gs.lat_deg)
        self.lon.setValue(gs.lon_deg)
        self.alt.setValue(gs.alt_m)

    def station(self) -> oe.GroundStation:
        return oe.GroundStation(self.name.text().strip() or "Ground station", self.lat.value(), self.lon.value(), self.alt.value(), self.minel.value())
