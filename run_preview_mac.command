#!/bin/sh
# satSTK launcher for macOS / Linux. STK's desktop application only runs on Windows (STK Engine also on Linux), so on a Mac this
# runs the OFFLINE PREVIEW: the analytic orbit, eclipse and line-of-sight estimate. Double-click it, or run it in Terminal.
# If macOS refuses to open it ("cannot verify"), right-click the file and choose Open, once.
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install it from python.org and run this again."; read -r _; exit 1
fi
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
python satstk.py --offline-preview "$@"
echo
printf "Press Enter to close ..."; read -r _
