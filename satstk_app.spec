# PyInstaller recipe for the satSTK desktop application.  Build with:   pyinstaller --noconfirm satstk_app.spec
# Produces dist/satSTK (a folder; on macOS also dist/satSTK.app).
import sys
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [("data", "data")], [], []
for pkg in ("skyfield", "pyqtgraph", "scipy"):       # skyfield ships its leap-second tables as data files
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["orbital_engine", "laser_link_engine", "atmosphere", "events", "simulation", "ui.clock", "ui.conditions_panel", "ui.events_dialog",
                  "ui.geo", "ui.globe_view", "ui.map_view", "ui.placement_dialog", "ui.telemetry_panel", "ui.theme", "ui.timeline_view"]

QT_UNUSED = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtQuick", "PySide6.QtQml",
             "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
             "PySide6.QtPdf", "PySide6.QtDesigner", "PySide6.QtSql", "PySide6.QtBluetooth"]
a = Analysis(["main_app.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["tkinter", "matplotlib", "pytest", "wx", "PyQt5", "PyQt6", "IPython"] + QT_UNUSED, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="satSTK", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="satSTK")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="satSTK.app", bundle_identifier="org.satstk.app",
                 info_plist={"CFBundleShortVersionString": "2.0.0", "NSHighResolutionCapable": True})
