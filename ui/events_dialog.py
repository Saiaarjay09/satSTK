"""Events dialog: choose which random events may happen, and add events by hand at a chosen time."""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QPushButton, QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout)

from events import CATEGORIES, EVENT_TYPES, EventsConfig, ManualEvent


def _effect_text(et) -> str:
    bits = []
    if et.outage:
        bits.append("beam lost")
    if et.loss_db:
        bits.append(f"+{et.loss_db[0]:g} to {et.loss_db[1]:g} dB loss")
    if et.jitter_factor[1] > 1:
        bits.append(f"pointing jitter x{et.jitter_factor[0]:g}-{et.jitter_factor[1]:g}")
    return ", ".join(bits)


def _duration_text(sec: float) -> str:
    return f"{sec:g} s" if sec < 120 else f"{sec / 60:g} min" if sec < 7200 else f"{sec / 3600:g} h"


class EventsDialog(QDialog):
    def __init__(self, cfg: EventsConfig, duration_h: float, current_events=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random & rare events")
        self.cfg = copy.deepcopy(cfg)
        self.duration_h = duration_h
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Pick which kinds of event may happen at random (turn random events on and set their frequency in the Conditions panel), or "
                              "add one by hand below. Rates are illustrative averages, not statistics for any real place; short events are applied as the "
                              "fraction of a time step they cover.", wordWrap=True))
        self.table = QTableWidget(len(EVENT_TYPES), 5)
        self.table.setHorizontalHeaderLabels(["On", "Event", "Typical rate / day", "Lasts", "Effect"])
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._rows = []
        ordered = sorted(EVENT_TYPES.values(), key=lambda t: (CATEGORIES.index(t.category), -t.rate_per_day))
        for r, et in enumerate(ordered):
            chk = QCheckBox()
            chk.setChecked(et.key not in self.cfg.disabled_types)
            self.table.setCellWidget(r, 0, chk)
            name = QTableWidgetItem(f"{et.name}   [{et.category}]")
            name.setToolTip(et.description)
            self.table.setItem(r, 1, name)
            self.table.setItem(r, 2, QTableWidgetItem(f"{et.rate_per_day:g}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{_duration_text(et.duration_s[0])} - {_duration_text(et.duration_s[1])}"))
            self.table.setItem(r, 4, QTableWidgetItem(_effect_text(et)))
            self._rows.append((et.key, chk))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(300)
        root.addWidget(self.table)

        g = QGroupBox("Add an event by hand")
        gl = QHBoxLayout(g)
        self.kind = QComboBox()
        for et in ordered:
            self.kind.addItem(et.name, et.key)
        self.start = QDoubleSpinBox()
        self.start.setRange(0, max(0.01, duration_h))
        self.start.setDecimals(2)
        self.start.setSuffix(" h from start")
        self.dur = QDoubleSpinBox()
        self.dur.setRange(0.1, 1_000_000)
        self.dur.setDecimals(1)
        self.dur.setSuffix(" s")
        self.sev = QSlider(Qt.Orientation.Horizontal)
        self.sev.setRange(0, 100)
        self.sev.setValue(60)
        self.sev.setMaximumWidth(110)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        for w in (self.kind, self.start, self.dur, QLabel("severity"), self.sev, add):
            gl.addWidget(w)
        root.addWidget(g)
        self.manual_list = QListWidget()
        self.manual_list.setMaximumHeight(90)
        root.addWidget(QLabel("Your events (select one and press Remove):"))
        root.addWidget(self.manual_list)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._remove)
        root.addWidget(rm, 0, Qt.AlignmentFlag.AlignLeft)
        if current_events:
            root.addWidget(QLabel(f"Events in the current run: {len(current_events)} - " + "; ".join(f"{e.name} at {e.start_s / 3600:.2f} h" for e in list(current_events)[:8])
                                  + (" ..." if len(current_events) > 8 else ""), wordWrap=True))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        self._kind_changed()
        self._refresh_manual()
        self.resize(900, 700)

    def _kind_changed(self) -> None:
        et = EVENT_TYPES[self.kind.currentData()]
        self.dur.setValue((et.duration_s[0] * et.duration_s[1]) ** 0.5)

    def _add(self) -> None:
        self.cfg.manual.append(ManualEvent(self.kind.currentData(), self.start.value() * 3600.0, self.dur.value(), self.sev.value() / 100.0))
        self._refresh_manual()

    def _remove(self) -> None:
        r = self.manual_list.currentRow()
        if 0 <= r < len(self.cfg.manual):
            del self.cfg.manual[r]
            self._refresh_manual()

    def _refresh_manual(self) -> None:
        self.manual_list.clear()
        for m in self.cfg.manual:
            self.manual_list.addItem(f"{EVENT_TYPES[m.key].name}: at {m.start_s / 3600:.2f} h for {_duration_text(m.duration_s)}, severity {m.severity:.0%}")

    def config(self) -> EventsConfig:
        self.cfg.disabled_types = [k for k, chk in self._rows if not chk.isChecked()]
        return self.cfg
