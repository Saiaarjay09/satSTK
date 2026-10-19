#!/bin/sh
# One-command launcher for macOS / Linux: creates ./venv the first time, installs requirements, starts satSTK.
# Extra options are passed through, e.g.  ./run.sh --headless
cd "$(dirname "$0")" || exit 1
command -v python3 >/dev/null 2>&1 || { echo "Python 3 not found - install it from python.org"; exit 1; }
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
python -m pip install --quiet -r requirements.txt
exec python main_app.py "$@"
