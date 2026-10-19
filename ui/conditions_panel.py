"""Conditions panel: every knob of the scenario in one place. Change a value and (with 'Live update' on) the whole simulation recomputes."""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from PySide6.QtCore import QDateTime, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

import orbital_engine as oe
from atmosphere import WEATHER_PRESETS
from events import EventsConfig
from simulation import GROUND_STATION_PRESETS, SimulationConfig, second_satellite

CN2_UNIT = 1e-15


class ConditionsPanel(QWidget):
    changed = Signal()
    place_requested = Signal()
    events_requested = Signal()

    def __init__(self, cfg: SimulationConfig, parent=None):
        super().__init__(parent)
        self._w: dict[str, QDoubleSpinBox] = {}
        self._loading = False
        self._events = EventsConfig()
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)

        # ---- ground station ------------------------------------------------------------------------------------------
        g, f = self._group("BEAM ORIGIN  (ground station)")
        self.site = QComboBox()
        self.site.addItem("Custom location")
        self.site.addItems(list(GROUND_STATION_PRESETS))
        self.site.currentTextChanged.connect(self._site_chosen)
        f.addRow("Known site", self.site)
        self._spin(f, "lat", "Latitude (°N)", 0, -90, 90, 3, "°")
        self._spin(f, "lon", "Longitude (°E)", 0, -180, 180, 3, "°")
        self._spin(f, "alt", "Altitude (m)", 0, -400, 9000, 0, " m")
        self._spin(f, "minel", "Min. elevation (°)", 20, 0, 80, 0, "°")
        place = QPushButton("Place on map ...")
        place.clicked.connect(self.place_requested)
        f.addRow(place)
        root.addWidget(g)

        # ---- weather -------------------------------------------------------------------------------------------------
        g, f = self._group("SKY & WEATHER")
        self.weather = QComboBox()
        self.weather.addItem("Custom")
        self.weather.addItems(list(WEATHER_PRESETS))
        self.weather.currentTextChanged.connect(self._weather_chosen)
        f.addRow("Preset", self.weather)
        self._spin(f, "vis", "Visibility (km)", 40, 0.1, 200, 1, " km")
        self._spin(f, "cloud", "Cloud loss, straight up (dB)", 0, 0, 80, 1, " dB")
        self._spin(f, "fog", "Fog loss, straight up (dB)", 0, 0, 80, 1, " dB")
        self._spin(f, "rain", "Rain (mm/h)", 0, 0, 150, 1, " mm/h")
        self._spin(f, "cn2", "Turbulence Cn² at ground (×1e-15)", 17, 0.1, 2000, 1, "")
        self._spin(f, "wind", "High-altitude wind (m/s)", 21, 0, 80, 0, " m/s")
        root.addWidget(g)

        # ---- satellites -----------------------------------------------------------------------------------------------
        g, f = self._group("SATELLITE")
        self._spin(f, "sat_alt", "Altitude (km)", 600, 300, 2000, 1, " km")
        self._spin(f, "sat_ltan", "Local time of node (h)", 10.5, 0, 24, 2, " h")
        self._spin(f, "sat_nu", "Start position (°)", 0, 0, 360, 1, "°")
        self.second = QCheckBox("Add a second satellite (Orbital-Node-Beta)")
        self.second.toggled.connect(self._second_toggled)
        f.addRow(self.second)
        self.target = QComboBox()
        self.target.addItems(["Orbital-Node-Alpha"])
        self.target.currentIndexChanged.connect(self._edited)
        f.addRow("Beam aims at", self.target)
        root.addWidget(g)

        # ---- laser ----------------------------------------------------------------------------------------------------
        g, f = self._group("LASER TERMINAL")
        self._spin(f, "tx", "Transmit power (W)", 10, 0.1, 500, 1, " W")
        self._spin(f, "div", "Beam divergence (µrad)", 100, 1, 5000, 1, " µrad")
        self._spin(f, "jit", "Pointing jitter RMS (µrad)", 10, 0, 500, 1, " µrad")
        self._spin(f, "aper", "Satellite aperture (m)", 0.3, 0.02, 3, 2, " m")
        self._spin(f, "rate", "Symbol rate (Gbaud)", 250, 1, 2000, 0, " Gbaud")
        self._spin(f, "pen", "Receiver penalty (dB)", 10, 0, 40, 1, " dB")
        root.addWidget(g)

        # ---- time -----------------------------------------------------------------------------------------------------
        g, f = self._group("TIME")
        self.start = QDateTimeEdit()
        self.start.setDisplayFormat("yyyy-MM-dd HH:mm 'UTC'")
        self.start.setCalendarPopup(True)
        self.start.dateTimeChanged.connect(self._edited)
        f.addRow("Start", self.start)
        self._spin(f, "hours", "Duration (h)", 24, 0.5, 168, 1, " h")
        self.step = QComboBox()
        self.step.addItems(["1 s", "10 s", "30 s"])
        self.step.currentIndexChanged.connect(self._edited)
        f.addRow("Time step", self.step)
        root.addWidget(g)

        # ---- thermal --------------------------------------------------------------------------------------------------
        g, f = self._group("THERMAL / POWER")
        self._spin(f, "diss", "Waste heat (kW)", 10, 0.1, 500, 1, " kW")
        self._spin(f, "rad", "Radiator area (m²)", 40, 1, 2000, 1, " m²")
        self._spin(f, "arr", "Solar array (m²)", 60, 1, 5000, 1, " m²")
        root.addWidget(g)

        # ---- random events --------------------------------------------------------------------------------------------
        g, f = self._group("RANDOM & RARE EVENTS")
        self.ev_on = QCheckBox("Random events on (birds, aircraft, storms, quakes ...)")
        self.ev_on.toggled.connect(self._edited)
        f.addRow(self.ev_on)
        self.ev_rate = QSlider(Qt.Orientation.Horizontal)
        self.ev_rate.setRange(0, 400)
        self.ev_rate.setValue(100)
        self.ev_rate_label = QLabel("×1")
        self.ev_rate.valueChanged.connect(self._rate_moved)
        row = QHBoxLayout()
        row.addWidget(self.ev_rate, 1)
        row.addWidget(self.ev_rate_label)
        f.addRow("Frequency", row)
        self.ev_seed = QSpinBox()
        self.ev_seed.setRange(0, 999999)
        self.ev_seed.valueChanged.connect(self._edited)
        reroll = QPushButton("Re-roll")
        reroll.setToolTip("Pick a new random seed: a different 'bad day'")
        reroll.clicked.connect(self._reroll)
        row = QHBoxLayout()
        row.addWidget(self.ev_seed)
        row.addWidget(reroll)
        f.addRow("Random seed", row)
        btn = QPushButton("Choose events / add one by hand ...")
        btn.clicked.connect(self.events_requested)
        f.addRow(btn)
        root.addWidget(g)

        # ---- apply ----------------------------------------------------------------------------------------------------
        self.live = QCheckBox("Live update (recompute as you change things)")
        self.live.setChecked(True)
        root.addWidget(self.live)
        self.apply_btn = QPushButton("Apply / recompute")
        root.addWidget(self.apply_btn)
        root.addStretch(1)
        self.load(cfg)

    # ------------------------------------------------------------------ building helpers
    def _group(self, title: str):
        g = QGroupBox(title)
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #35b6ff; border: 1px solid #263241; border-radius: 4px; margin-top: 8px; padding-top: 6px; }"
                        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")
        return g, QFormLayout(g)

    def _spin(self, form, key, label, value, lo, hi, decimals, suffix="") -> None:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSuffix(suffix)
        sp.setValue(value)
        sp.setAlignment(Qt.AlignmentFlag.AlignRight)
        sp.setKeyboardTracking(False)
        sp.valueChanged.connect(self._edited)
        form.addRow(label, sp)
        self._w[key] = sp

    # ------------------------------------------------------------------ reactions
    def _edited(self, *_) -> None:
        if not self._loading:
            self.changed.emit()

    def _rate_moved(self, v: int) -> None:
        scale = 10 ** ((v - 100) / 100.0)
        self.ev_rate_label.setText(f"×{scale:.3g}")
        self._edited()

    def _reroll(self) -> None:
        import random
        self.ev_seed.setValue(random.randint(0, 999999))

    def _second_toggled(self, on: bool) -> None:
        self._loading = True
        self.target.clear()
        self.target.addItems(["Orbital-Node-Alpha", "Orbital-Node-Beta"] if on else ["Orbital-Node-Alpha"])
        self._loading = False
        self._edited()

    def _site_chosen(self, text: str) -> None:
        gs = GROUND_STATION_PRESETS.get(text)
        if gs is None:
            return
        self._loading = True
        self._w["lat"].setValue(gs.lat_deg)
        self._w["lon"].setValue(gs.lon_deg)
        self._w["alt"].setValue(gs.alt_m)
        self._loading = False
        self._edited()

    def _weather_chosen(self, text: str) -> None:
        a = WEATHER_PRESETS.get(text)
        if a is None:
            return
        self._loading = True
        for key, val in (("vis", a.visibility_km), ("cloud", a.cloud_loss_db), ("fog", a.fog_loss_db), ("rain", a.rain_mm_h),
                         ("cn2", a.turbulence_cn2_ground / CN2_UNIT), ("wind", a.wind_rms_m_s)):
            self._w[key].setValue(val)
        self._loading = False
        self._edited()

    # ------------------------------------------------------------------ config <-> widgets
    def set_events_config(self, ec: EventsConfig) -> None:
        self._events = copy.deepcopy(ec)

    def events_config(self) -> EventsConfig:
        ec = copy.deepcopy(self._events)
        ec.enabled = self.ev_on.isChecked()
        ec.seed = self.ev_seed.value()
        ec.rate_scale = 10 ** ((self.ev_rate.value() - 100) / 100.0)
        return ec

    def load(self, cfg: SimulationConfig) -> None:
        self._loading = True
        w = self._w
        g, a, L, T, s0 = cfg.ground, cfg.atmosphere, cfg.link, cfg.thermal, cfg.satellites[0]
        vals = {"lat": g.lat_deg, "lon": g.lon_deg, "alt": g.alt_m, "minel": g.min_elevation_deg, "vis": a.visibility_km, "cloud": a.cloud_loss_db,
                "fog": a.fog_loss_db, "rain": a.rain_mm_h, "cn2": a.turbulence_cn2_ground / CN2_UNIT, "wind": a.wind_rms_m_s,
                "sat_alt": s0.altitude_km, "sat_ltan": s0.ltan_hours if s0.ltan_hours is not None else 10.5, "sat_nu": s0.true_anomaly_deg,
                "tx": L.tx_power_w, "div": L.divergence_urad, "jit": L.pointing_jitter_urad, "aper": L.rx_aperture_m, "rate": L.symbol_rate_gbaud,
                "pen": L.implementation_penalty_db, "hours": cfg.duration_h, "diss": T.internal_dissipation_w / 1000.0, "rad": T.radiator_area_m2,
                "arr": T.solar_array_area_m2}
        for k, v in vals.items():
            w[k].setValue(float(v))
        st = oe.as_utc(cfg.start)
        self.start.setDateTime(QDateTime(st.year, st.month, st.day, st.hour, st.minute, 0))
        self.step.setCurrentText(f"{cfg.step_s:g} s" if f"{cfg.step_s:g} s" in ("1 s", "10 s", "30 s") else "10 s")
        self.second.setChecked(len(cfg.satellites) > 1)
        self.target.clear()
        self.target.addItems([s.name for s in cfg.satellites])
        self.target.setCurrentIndex(min(cfg.target_index, len(cfg.satellites) - 1))
        self.weather.setCurrentText(a.label if a.label in WEATHER_PRESETS else "Custom")
        self.site.setCurrentText(next((n for n, p in GROUND_STATION_PRESETS.items() if abs(p.lat_deg - g.lat_deg) < 1e-6 and abs(p.lon_deg - g.lon_deg) < 1e-6), "Custom location"))
        self._events = copy.deepcopy(cfg.events)
        self.ev_on.setChecked(cfg.events.enabled)
        self.ev_seed.setValue(cfg.events.seed)
        import math
        self.ev_rate.setValue(int(round(100 + 100 * math.log10(max(cfg.events.rate_scale, 1e-3)))))
        self._loading = False

    def config(self, base: SimulationConfig) -> SimulationConfig:
        cfg = copy.deepcopy(base)
        v = {k: sp.value() for k, sp in self._w.items()}
        d = self.start.dateTime()
        cfg.start = datetime(d.date().year(), d.date().month(), d.date().day(), d.time().hour(), d.time().minute(), 0, tzinfo=timezone.utc)
        cfg.duration_h = v["hours"]
        cfg.step_s = float(self.step.currentText().split()[0])
        site = GROUND_STATION_PRESETS.get(self.site.currentText())
        same_place = abs(base.ground.lat_deg - v["lat"]) < 1e-6 and abs(base.ground.lon_deg - v["lon"]) < 1e-6
        name = site.name if site is not None else (base.ground.name if same_place else "Ground station")
        cfg.ground = oe.GroundStation(name, v["lat"], v["lon"], v["alt"], v["minel"])
        a = cfg.atmosphere
        a.visibility_km, a.cloud_loss_db, a.fog_loss_db, a.rain_mm_h = v["vis"], v["cloud"], v["fog"], v["rain"]
        a.turbulence_cn2_ground, a.wind_rms_m_s = v["cn2"] * CN2_UNIT, v["wind"]
        a.label = next((n for n, p in WEATHER_PRESETS.items() if (p.visibility_km, p.cloud_loss_db, p.fog_loss_db, p.rain_mm_h, round(p.turbulence_cn2_ground / CN2_UNIT, 3),
                                                                      p.wind_rms_m_s) == (a.visibility_km, a.cloud_loss_db, a.fog_loss_db, a.rain_mm_h, round(v["cn2"], 3), a.wind_rms_m_s)), "Custom")
        s0 = cfg.satellites[0]
        s0.altitude_km, s0.ltan_hours, s0.raan_deg, s0.true_anomaly_deg = v["sat_alt"], v["sat_ltan"], None, v["sat_nu"]
        if self.second.isChecked():
            if len(cfg.satellites) < 2:
                cfg.satellites.append(second_satellite())
            cfg.satellites[1].altitude_km = v["sat_alt"]
            cfg.satellites = cfg.satellites[:2]
        else:
            cfg.satellites = cfg.satellites[:1]
        cfg.target_index = max(0, min(self.target.currentIndex(), len(cfg.satellites) - 1))
        L = cfg.link
        L.tx_power_w, L.divergence_urad, L.pointing_jitter_urad, L.rx_aperture_m = v["tx"], v["div"], v["jit"], v["aper"]
        L.symbol_rate_gbaud, L.implementation_penalty_db = v["rate"], v["pen"]
        cfg.thermal.internal_dissipation_w, cfg.thermal.radiator_area_m2, cfg.thermal.solar_array_area_m2 = v["diss"] * 1000.0, v["rad"], v["arr"]
        cfg.events = self.events_config()
        return cfg
