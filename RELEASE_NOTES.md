## satSTK: orbital mission & laser link simulator

A standalone desktop application: two Sun-synchronous satellites and a 1550 nm laser link, with a 3D globe, 2D map, Gantt timeline of access and eclipse
windows, live telemetry and link-budget plots. It runs entirely on your computer: no Ansys STK, no licence, no internet, nothing stored unless you export a CSV.

**Download the file for your computer** (below, under Assets):

| System | File |
|---|---|
| Mac with Apple silicon (M1 or newer) | `satSTK-…-macos-arm64.dmg`: open it and drag satSTK into Applications |
| Windows 10/11, 64-bit | `satSTK-…-windows-x64-setup.exe`: run the installer |

Intel Macs are not built (run from source: see the README). The files are about 100-200 MB because they include Python, Qt, NumPy and SciPy.

### The first time you open it

The apps are **not code-signed** (signing needs a paid developer identity), so your computer will warn you:

- **Mac:** "satSTK can't be opened because Apple cannot check it." Open **System Settings → Privacy & Security**, scroll down and press **Open Anyway**. Once only.
- **Windows:** a blue "Windows protected your PC" screen. Press **More info → Run anyway**.

### Please note

- **Not validated against Ansys STK or real hardware.** The maths is checked against textbook values by 73 automated tests; see "Honest limits" in the README.
- The macOS app was built and tested on an Apple-silicon Mac. The Windows installer is built and smoke-tested by GitHub's automated build; it has not been run on a real Windows PC by the author.
- Space = play/pause, ← / → = step, File → Scenario settings changes the orbit and laser parameters.
