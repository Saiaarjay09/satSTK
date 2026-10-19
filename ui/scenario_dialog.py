"""Scenario dialog: the handful of numbers most people want to change. Everything else lives in simulation.SimulationConfig."""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import (QComboBox, QDateTimeEdit, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QVBoxLayout)

import orbital_engine as oe
from simulation import SimulationConfig


class ScenarioDialog(QDialog):
    def __init__(self, cfg: SimulationConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scenario")
        self.base = cfg
        self._w = {}
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Orbits are propagated with J2 from these starting conditions. Click OK to recompute (about a second).",
                              wordWrap=True))
        row = QHBoxLayout()
        root.addLayout(row)

        left, right = QVBoxLayout(), QVBoxLayout()
        row.addLayout(left)
        row.addLayout(right)

        g = QGroupBox("Time")
        f = QFormLayout(g)
        s = oe.as_utc(cfg.start)
        self._start = QDateTimeEdit(QDateTime(s.year, s.month, s.day, s.hour, s.minute, s.second))
        self._start.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
        self._start.setCalendarPopup(True)
        f.addRow("Start", self._start)
        self._spin(f, "duration", "Duration (hours)", cfg.duration_h, 0.5, 168, 1, " h")
        self._step = QComboBox()
        self._step.addItems(["1 s", "10 s", "30 s"])
        self._step.setCurrentText(f"{cfg.step_s:g} s" if f"{cfg.step_s:g} s" in ("1 s", "10 s", "30 s") else "10 s")
        f.addRow("Time step", self._step)
        self._shadow = QComboBox()
        self._shadow.addItems(["conical", "cylindrical"])
        self._shadow.setCurrentText(cfg.shadow_model)
        f.addRow("Earth shadow model", self._shadow)
        self._spin(f, "buffer", "Atmosphere buffer (km)", cfg.los_buffer_km, 0, 500, 0, " km")
        left.addWidget(g)

        for k in range(2):
            sat = cfg.satellites[k]
            g = QGroupBox(sat.name)
            f = QFormLayout(g)
            self._spin(f, f"alt{k}", "Altitude (km)", sat.altitude_km, 300, 2000, 1, " km")
            self._spin(f, f"ltan{k}", "Local time of ascending node (h)", sat.ltan_hours if sat.ltan_hours is not None else 10.5, 0, 24, 2, " h")
            self._spin(f, f"nu{k}", "Start position: true anomaly (°)", sat.true_anomaly_deg, 0, 360, 1, "°")
            (left if k == 0 else right).addWidget(g)

        g = QGroupBox("Laser link")
        f = QFormLayout(g)
        L = cfg.link
        self._spin(f, "tx", "Transmit power (W)", L.tx_power_w, 0.1, 500, 1, " W")
        self._spin(f, "div", "Beam divergence, full angle (µrad)", L.divergence_urad, 1, 5000, 1, " µrad")
        self._spin(f, "jit", "Pointing jitter, RMS per axis (µrad)", L.pointing_jitter_urad, 0, 500, 1, " µrad")
        self._spin(f, "aper", "Receive aperture (m)", L.rx_aperture_m, 0.02, 3, 2, " m")
        self._spin(f, "rate", "Symbol rate (Gbaud)", L.symbol_rate_gbaud, 1, 2000, 0, " Gbaud")
        self._spin(f, "pen", "Receiver implementation penalty (dB)", L.implementation_penalty_db, 0, 40, 1, " dB")
        right.addWidget(g)

        g = QGroupBox("Thermal / power (first-order)")
        f = QFormLayout(g)
        T = cfg.thermal
        self._spin(f, "diss", "Waste heat to radiate (kW)", T.internal_dissipation_w / 1000.0, 0.1, 500, 1, " kW")
        self._spin(f, "rad", "Radiator area (m²)", T.radiator_area_m2, 1, 2000, 1, " m²")
        self._spin(f, "arr", "Solar array area (m²)", T.solar_array_area_m2, 1, 5000, 1, " m²")
        right.addWidget(g)
        left.addStretch(1)
        right.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _spin(self, form, key, label, value, lo, hi, decimals, suffix="") -> None:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(float(value))
        sp.setSuffix(suffix)
        sp.setAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow(label, sp)
        self._w[key] = sp

    def config(self) -> SimulationConfig:
        v = {k: w.value() for k, w in self._w.items()}
        cfg = copy.deepcopy(self.base)
        d = self._start.dateTime()
        cfg.start = datetime(d.date().year(), d.date().month(), d.date().day(), d.time().hour(), d.time().minute(), d.time().second(),
                             tzinfo=timezone.utc)
        cfg.duration_h = v["duration"]
        cfg.step_s = float(self._step.currentText().split()[0])
        cfg.shadow_model = self._shadow.currentText()
        cfg.los_buffer_km = v["buffer"]
        for k in range(2):
            s = cfg.satellites[k]
            s.altitude_km, s.ltan_hours, s.raan_deg, s.true_anomaly_deg = v[f"alt{k}"], v[f"ltan{k}"], None, v[f"nu{k}"]
        cfg.link.tx_power_w, cfg.link.divergence_urad, cfg.link.pointing_jitter_urad = v["tx"], v["div"], v["jit"]
        cfg.link.rx_aperture_m, cfg.link.symbol_rate_gbaud, cfg.link.implementation_penalty_db = v["aper"], v["rate"], v["pen"]
        cfg.thermal.internal_dissipation_w, cfg.thermal.radiator_area_m2 = v["diss"] * 1000.0, v["rad"]
        cfg.thermal.solar_array_area_m2 = v["arr"]
        return cfg
