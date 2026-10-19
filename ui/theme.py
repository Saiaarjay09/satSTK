"""Dark mission-control colours and the global Qt style sheet."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

BG = "#0a0e13"
PANEL = "#10161d"
PANEL_2 = "#161e28"
BORDER = "#263241"
TEXT = "#d5dde7"
DIM = "#7d8a99"
ACCENT = "#35b6ff"
GREEN = "#2ecc71"
RED = "#ff4d4f"
AMBER = "#ffb020"
UMBRA = "#5b4fcf"
SAT_COLORS = ["#3fc1ff", "#ff9f43", "#c77dff", "#7bed9f", "#ff6b81", "#ffd93d"]


def sat_color(i: int) -> QColor:
    return QColor(SAT_COLORS[i % len(SAT_COLORS)])


def mono_font(point_size: int = 10) -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    f.setPointSize(point_size)
    return f


STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 12px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QDockWidget {{ titlebar-close-icon: none; font-weight: bold; }}
QDockWidget::title {{ background: {PANEL_2}; padding: 6px 8px; border-bottom: 1px solid {BORDER}; }}
QPushButton, QToolButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:checked {{ background: #12354a; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {DIM}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit, QLineEdit {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; selection-background-color: #12354a; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: #12354a; }}
QMenu {{ background: {PANEL_2}; border: 1px solid {BORDER}; }}
QStatusBar {{ background: {PANEL}; color: {DIM}; }}
QLabel[role="dim"] {{ color: {DIM}; }}
QLabel[role="section"] {{ color: {ACCENT}; font-weight: bold; padding-top: 6px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; }}
QProgressBar {{ border: 1px solid {BORDER}; border-radius: 3px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
"""


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL_2))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(PANEL_2))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#12354a"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(DIM))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
