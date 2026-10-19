# Setu: STK automation for the orbital laser crosslink

*Setu* is Sanskrit for "bridge": here, the laser bridge between two orbiting compute nodes. It is part of the Orbital AI Datacentre
project (the [Orbital-Datacentre](https://github.com/Saiaarjay09/Orbital-Datacentre) repository holds the optical-channel simulator).

`setu_stk.py` drives **Ansys STK** (Systems Tool Kit) from Python. It builds a 24-hour scenario with two 600 km Sun-synchronous
satellites (**Orbital-Node-Alpha**, **Orbital-Node-Beta**), reports **when Alpha is in sunlight, penumbra and umbra** (the cases the
mechanically pumped fluid loop must survive), fits Alpha with a **1550 nm laser transmitter** and Beta with a **laser receiver**
(100 microradian beam), computes the **line-of-sight access** between them, and extracts **received power, BER and latency**
during each window.

> **Status, honestly.** The script was written from Ansys' public STK documentation and has **not been run against a live STK**
> (STK was not available where it was written). The orbit mathematics and the STK plumbing are unit-tested (`python -m pytest`),
> and `--offline-preview` gives an independent analytic answer to compare with. Expect to fix a name or two on the first real run: the
> script prints what STK rejected and saves every attribute STK exposes, so any difference is quick to correct.

## Download

Go to the **Releases** page and take the zip for your computer:

| Download | For |
|---|---|
| `Setu-STK-<version>-windows.zip` | Windows with STK installed. Unzip and double-click `run_windows.bat` |
| `Setu-STK-<version>-macos.zip` | Mac or Linux. STK's desktop app does not run on macOS, so this runs the **offline preview** (double-click `run_preview_mac.command`) |

## What is in this repository, file by file

```
STK/
├── setu_stk.py              the program: orbit maths, the preview, and every STK call
├── README.md                this guide
├── requirements.txt         the Python libraries needed (numpy, pytest)
├── RELEASE_NOTES.md         the text shown on each GitHub release page
├── run_windows.bat          one-click launcher for Windows (full STK run)
├── run_preview_mac.command  one-click launcher for macOS / Linux (offline preview)
├── .gitignore               keeps results, virtual environments and caches out of git
├── tests/
│   └── test_setu_stk.py     the automated checks
└── .github/workflows/
    └── release.yml          the automatic release builder
```

### `setu_stk.py` (the program)

One file, in four layers. The settings are the upper-case numbers in the block at the top; nothing else needs editing to change the design.

1. **Settings block** (top of the file). Scenario name, start date and length (24 h), report time step, altitude (600 km), eccentricity, the
   local solar time of each satellite's ascending node (`NODE_ALPHA_LTAN_H`, `NODE_BETA_LTAN_H`), where each starts along its orbit, and the laser
   settings (wavelength 1550 nm, power, data rate, beam divergence 100 microradians, apertures, the Earth-clearance margin).
2. **Orbit maths** (no STK needed; also used by the preview).
   * `sso_inclination_deg` works out the inclination at which the Earth's bulge (J2) turns the orbit plane once a year, matching the Sun
     (about 97.79 degrees at 600 km). That drift is what "Sun-synchronous" means.
   * `sun_ecliptic_longitude_deg`, `sun_right_ascension_deg` and `raan_for_ltan_deg` place the orbit plane so the node crosses the equator
     at the chosen local solar time (accurate to about a degree).
   * `orbit_elements` bundles these into one orbit description. `parse_utcg` and `format_utcg` convert STK-style dates.
3. **Offline preview** (`offline_propagate`, `offline_sun_vectors`, `intervals`, `offline_preview`, `print_preview`). A simple analytic model:
   circular orbits, slow J2 plane drift, the Earth's shadow as a cylinder, and a straight-line test for whether the Earth blocks the view
   between the nodes. It prints Alpha's eclipse passes and the line-of-sight windows with range. It needs only NumPy and is the way to sanity-check
   STK's answers. STK's numbers are the ones to use.
4. **STK automation.** Everything that touches STK is a small function with a clear error message:
   * `start_stk` launches STK (desktop window, or Engine with no window) through the `agi.stk12` Python API, and explains how to install it if missing.
   * `connect` runs one STK "Connect" text command and returns its reply, naming the command if STK rejects it.
   * `build_scenario` creates the 24-hour scenario. `add_sso_satellite` creates a satellite on the J2 propagator with the computed orbit.
   * `alpha_eclipse_times` asks STK for Alpha's **Lighting Times** report and `parse_lighting_report` turns it into umbra / penumbra / sunlight intervals.
   * `add_laser_terminals` adds the transmitter (on Alpha) and receiver (on Beta), selects STK's laser models, and sets frequency (c / 1550 nm),
     power, data rate and beam divergence. Because the optical-antenna attribute names differ between STK releases, `list_attributes` and
     `set_first_matching` search the model's attribute list for "divergence", "beamwidth" or "diameter" rather than assuming a name.
   * `dump_attributes` saves everything STK exposes for the two laser objects, to make any mismatch easy to fix.
   * `point_at_each_other` turns on target pointing so each satellite aims at the other (a stand-in for the fast steering mirror, assuming perfect pointing).
   * `access_windows` lists the line-of-sight windows. `link_budget_samples` reads STK's *Link Information* data (range, received isotropic power,
     Eb/No, BER, link margin, propagation delay) across each window, and `summarise_link` reduces it to min / mean / max, ignoring STK's "-9999.99
     means no link" flag.
   * `write_csv` saves results; `run_stk` runs the whole sequence; `main` reads the command-line options.

Command-line options: `--offline-preview` (no STK), `--engine` (STK Engine, no window), `--keep-open` (leave STK open afterwards).

### `README.md`
This file.

### `requirements.txt`
`numpy` (orbit maths and the preview) and `pytest` (the tests). The STK Python API is **not** on this list: it is not on PyPI for STK 12 and
comes as a file inside your STK installation (see below).

### `run_windows.bat`
Double-click launcher for Windows. The first time it creates a private Python environment (`.venv`), installs `requirements.txt`, checks for
the STK Python API (asking for the path of the `.whl` file if it is missing, or offering the offline preview instead), then runs `setu_stk.py`.
Any options you type after it are passed on, e.g. `run_windows.bat --engine`.

### `run_preview_mac.command`
Double-click launcher for macOS / Linux. It makes the same private environment and runs `setu_stk.py --offline-preview`. (STK's desktop app is
Windows-only, so a Mac can run the preview but not STK.) If macOS refuses to open it the first time, right-click it and choose **Open**.

### `tests/test_setu_stk.py`
Twelve automated checks, run with `python -m pytest`. They cover: the Sun-synchronous inclination against textbook values (97.79 degrees at
600 km); the Sun's position and the local-time-to-RAAN conversion; date formatting; that the preview orbit is circular with the right period;
that eclipse lasts about 35 minutes per orbit and the line of sight is genuinely intermittent; that the Earth blocks a link through its centre; the
parsing of STK's Lighting Times report; error handling around Connect commands; the attribute search (using a stand-in for STK, so no STK is needed);
the link summary and its handling of the "no link" flag; the 1550 nm frequency; and the message shown when the STK Python API is missing.

### `RELEASE_NOTES.md`
The text GitHub shows on each release page (what each download is for, and the honest status).

### `.gitignore`
Keeps `stk_output/` (your results), the virtual environments and Python caches out of git.

### `.github/workflows/release.yml`
GitHub's automatic builder. When a version tag such as `v0.1.0` is pushed it runs the tests and the preview, builds the two zips (Windows with
`run_windows.bat`; macOS with `run_preview_mac.command`), and publishes them as a GitHub Release using `RELEASE_NOTES.md`. To make a release:
`git tag v0.1.0 && git push origin v0.1.0`.

## What you need to run the full STK simulation

| Need | Notes |
|---|---|
| **Windows 10/11** with **Ansys STK 12.x**, licensed | STK's desktop application is Windows-only (Linux has STK Engine only; macOS is not supported) |
| The **Communications** capability in your STK licence | needed for transmitters, receivers and link budgets |
| **Python 3.9 to 3.12** (match your STK release notes) | from python.org |
| The STK Python API, `agi.stk12` | a `.whl` file that ships **inside the STK install folder** (search for `agi.stk12*.whl`, commonly under `...\STK 12\bin\AgPythonAPI\`) |
| `numpy` | for the orbit maths and the preview (installed by the launcher) |

Manual install, if you prefer (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "C:\Program Files\AGI\STK 12\bin\AgPythonAPI\agi.stk12-<version>-py3-none-any.whl"   # use the real file name on your PC
python setu_stk.py
```

Results are written to `stk_output\`:

| File | Contents |
|---|---|
| `alpha_lighting.csv`, `alpha_lighting_report_raw.txt` | sunlight / penumbra / umbra intervals for Node Alpha (STK "Lighting Times" report) |
| `access_windows.csv` | start, stop and duration of each Alpha to Beta line-of-sight window |
| `link_budget.csv` | per time step: range, received isotropic power, carrier power, Eb/No, BER, link margin, propagation delay |
| `transmitter_attributes.txt`, `receiver_attributes.txt` | every setting STK exposes for the two laser objects |
| `Orbital_AI_Datacentre.sc` | the scenario, which you can open in STK |

Run the offline preview first, even on a Mac: `python setu_stk.py --offline-preview` prints the Sun-synchronous inclination, Alpha's eclipse
passes (about 35 minutes of shadow every 96.7-minute orbit at this altitude) and the line-of-sight windows (about 18 minutes at a time, 4,300 to
5,200 km range with the default offsets). Agreement with STK within a few minutes is the sanity check.

## Changing the design

Everything is in the marked block at the top of the script: altitude, the **local time of each node** (`NODE_*_LTAN_H`; different times mean
different orbit planes, which is what makes the line of sight come and go), where Beta starts (`NODE_BETA_TRUE_ANOMALY_DEG`), laser power,
data rate, wavelength, beam divergence and aperture. To make the link continuous, give both nodes the same local time and a small separation;
to make it rarer, increase the separation.

## How it works, and where it may need adjusting

* **Orbits.** The Sun-synchronous inclination is solved from the J2 nodal drift and the node's RAAN from the local time and the Sun's position
  (accurate to about a degree). STK propagates with its **J2 Perturbation** propagator so the plane really drifts one turn per year.
  Satellite creation follows the pattern in AGI's own Python example.
* **Eclipse.** STK's `Lighting Times` report for Alpha, parsed into umbra / penumbra / sunlight intervals. Umbra is the full-shadow time
  the thermal design cares about.
* **Laser terminals.** STK's own `Laser_Transmitter_Model` and `Laser_Receiver_Model`, set through Connect commands. Frequency is
  c / 1550 nm, power and data rate are set directly. Because the exact attribute names for the optical antenna differ between releases, the
  beam divergence (and apertures) are found by **searching the model's attribute list** for "divergence" or "beamwidth" (or "diameter"); if none
  is found the script says so and the attribute dump shows what to set by hand.
* **Pointing.** A 100 microradian beam only works if it is aimed, so both satellites are told to point at each other (target pointing). This
  stands in for the fast steering mirror: it assumes perfect pointing. The boresight that STK aligns to the target must be the one your
  transmitter points along. If every sample comes back as **-9999.99 dB**, STK is saying there is no valid link: check the pointing (and that
  the transmitter and receiver frequencies overlap).
* **Link budget.** Read from the access's `Link Information` data provider: received isotropic power (`Rcvd. Iso. Power`), `BER`,
  `Eb/No`, `Link Margin` and `Propagation Delay`. **Latency** here is one-way propagation delay only (range / c, about 14 to 17 ms for
  4,300 to 5,200 km). It does not include processing or queuing.
* **STK 13 / Ansys STK 2025 and later** ship a newer Python package, `ansys-stk-core`, whose class and method names are different (for example
  `start_application`, `new_scenario`). This script uses the `agi.stk12` names. If you have STK 13 or later, either keep an STK 12
  installation for this script or port the object-model calls (only `start_stk`, `build_scenario`, `add_sso_satellite`, `access_windows`,
  `link_budget_samples` and `point_at_each_other` touch the object model; the rest is Connect commands, which are the same across versions).

## Limits

The 1 Tbit/s figure is the configured data rate. Whether a real terminal can close it depends on the receiver's sensitivity and noise
settings (STK's default laser receiver values are generic): treat the received-power and BER results as a **starting point for the link
budget**, to be refined with your actual terminal's numbers (the Orbital-Datacentre simulator in this repository models the optical channel in
more detail).
