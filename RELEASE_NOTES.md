## Setu-STK

Python automation for **Ansys STK**: two 600 km Sun-synchronous satellites, Node Alpha's eclipse times, and a 1550 nm laser link
between the nodes with line-of-sight access, received power, BER and latency.

| Download | For |
|---|---|
| `Setu-STK-…-windows.zip` | **Windows with STK installed.** Unzip, double-click `run_windows.bat`. Runs the full STK automation (and the offline preview) |
| `Setu-STK-…-macos.zip` | **Mac (or Linux).** STK's desktop app does not run on macOS, so this runs the **offline preview**: an analytic orbit, eclipse and line-of-sight estimate. Unzip, double-click `run_preview_mac.command` |

Both need Python 3.9 to 3.12 installed. The full STK run also needs STK 12 with the Communications capability and its Python API wheel
(`agi.stk12`, found inside the STK install folder); the launcher asks for it the first time.

### Please note
- This was written from Ansys' public documentation and has **not been run against a live STK** yet. The orbit maths and the STK plumbing are
  unit-tested; expect to adjust a name or two on the first real run. The script reports what STK rejected and saves every setting STK exposes.
- On a Mac, the first time you open the `.command` file macOS may refuse it: right-click it and choose **Open**.
