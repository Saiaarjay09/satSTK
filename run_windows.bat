@echo off
rem Setu launcher for Windows: sets up a private Python environment the first time, then runs the STK automation.
rem Double-click it, or run it from PowerShell. Extra options are passed through, e.g.  run_windows.bat --offline-preview
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo.
    echo Python was not found. Install Python 3.9 to 3.12 from python.org, tick "Add python.exe to PATH", then run this again.
    pause
    exit /b 1
  )
)
call ".venv\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python -c "import agi.stk12" 2>nul
if errorlevel 1 (
  echo.
  echo The STK Python API ^(agi.stk12^) is not installed yet. It is a .whl file inside your STK install folder.
  echo Search the STK folder for "agi.stk12*.whl" ^(often under bin\AgPythonAPI^).
  set /p WHEEL=Paste the full path of that .whl file here, or press Enter to run only the offline preview: 
  if defined WHEEL (
    python -m pip install "%WHEEL%"
  ) else (
    python setu_stk.py --offline-preview
    pause
    exit /b 0
  )
)

python setu_stk.py %*
echo.
pause
