# satSTK — Orbital Mission & Laser Link Simulator

*satSTK* is the mission-analysis companion to [satCAD](https://github.com/Saiaarjay09/satCAD): satCAD designs the spacecraft, satSTK simulates its orbit, eclipses and laser crosslink.

It is a free, standalone simulator for a small **orbital AI datacentre**: two satellites (**Orbital-Node-Alpha** and **Orbital-Node-Beta**) in 600 km
Sun-synchronous orbits talking over a **1550 nm laser link**. It runs natively on **macOS and Linux**, needs **no Ansys STK, no licence, no Windows COM,
no internet and no account**, and stores nothing unless you export a CSV.

![satSTK window: 3D globe, 2D map, Gantt timeline, telemetry](docs/screenshot.png)

*(Screenshot rendered by the app itself with `--screenshot`.)*

## Run it (macOS)

You need Python 3.10 or newer (`python3 --version` to check; get it from python.org if missing). Then, in Terminal, from this folder:

```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

and launch the app with the single command:

```bash
python main_app.py
```

Next time you only need `source venv/bin/activate` and `python main_app.py` (or just `./run.sh`, which does all of it for you).
Linux is the same. Windows should work with `venv\Scripts\activate` but has **not been tried**.

## What you see

| Pane | What it shows |
|---|---|
| **3D view** (top left) | Shaded globe with day/night terminator, orbit paths (bright trail behind, faint path ahead), live satellite markers, and the laser link: **green line = locked, dashed red = blocked by the Earth**. Drag to rotate, wheel to zoom, double-click to reset; the drop-down can follow either satellite. The camera is inertial, so you see the Earth turn under the orbits. |
| **2D map** (top middle) | Equirectangular ("Mercator-style") world map with the terminator, sub-satellite ground tracks, coverage footprints (0° elevation), the sub-solar point and the link between the two sub-satellite points. |
| **Timeline** (bottom) | Gantt chart of access (line-of-sight) windows and eclipse intervals (penumbra / umbra) for the whole scenario, a red playhead you can click or drag, and **Play / Pause / Step / speed** controls (1× to 3600×). Keyboard: Space, ←, →. |
| **Telemetry dock** (right) | Live numbers for the link (range, range rate, latency, Doppler, received power in dBm, SNR, Eb/N0, BER, throughput, path/divergence/pointing losses, grazing clearance) and for each satellite (lat/lon/alt, speed, eclipse state, beta angle, radiator temperature, solar-array power, laser-turret azimuth/elevation and slew rates). Below: live plots of **range, received power, BER, throughput and radiator temperature**. A drop-down picks BPSK / QPSK / 8PSK / 16QAM. |

All four views share one clock, so they always show the same instant. **File → Scenario settings** changes the start time, duration (up to 7 days), 1 s / 10 s / 30 s time step, altitude, local time of the ascending node, start phase, laser parameters and thermal numbers, and recomputes. **File → Export link data as CSV** saves the time series (only when you ask).

No window? `python main_app.py --headless` prints a text summary (`--csv out.csv` writes the data; `--hours`, `--step`, `--start` change the scenario).

## What is inside (file by file)

| File | Job |
|---|---|
| `orbital_engine.py` | Orbits and geometry: J2 propagator, Sun-synchronous inclination, local-time → RAAN, Sun position, ECI/ECEF/lat-lon-alt, **line of sight vs. the Earth ellipsoid + 100 km atmosphere buffer**, **Earth shadow (conical with penumbra, or cylindrical)**, beta angle, radiator temperature and solar-array power, interval finder. |
| `laser_link_engine.py` | 1550 nm link budget at every time step: range, free-space path loss, Gaussian beam-divergence loss (100 µrad), pointing-jitter loss, received power (dBm), SNR, latency, BER for BPSK/QPSK/8PSK/16QAM, throughput, Doppler, and turret pointing angles + slew rates (deg/s). |
| `simulation.py` | Scenario description (`SimulationConfig`) and `run_simulation()` that wires the two engines together for every satellite pair (two or more satellites are supported; the GUI shows the first pair). No GUI code. |
| `main_app.py` | The desktop app and the `--headless` / `--screenshot` command line. |
| `ui/` | The Qt widgets: globe, map, timeline, telemetry dock, scenario dialog, playback clock, dark theme. |
| `data/land_110m.json` | Land outlines from Natural Earth (public domain), 76 KB. |
| `tests/` | 73 automated tests (see below). |

Requirements are only `numpy`, `scipy`, `skyfield`, `PySide6`, `pyqtgraph` (`requirements-dev.txt` adds `pytest`). Skyfield is used for its built-in time scale (leap seconds, Earth rotation angle), which works offline. The Sun position uses a short analytic formula, so **no ephemeris file is ever downloaded**. (Plotly is not used.)

## The physics in plain words

* **Orbit.** Gravity plus the Earth's equatorial bulge (the "J2" term) is integrated numerically (SciPy DOP853). J2 makes the orbit plane turn; at 97.79° inclination (600 km) it turns exactly once per year, keeping the orbit at a fixed angle to the Sun — that is what *Sun-synchronous* means. Default planes cross the equator at local solar time 10:30 (Alpha) and 13:30 (Beta).
* **Line of sight.** The straight line between the satellites is tested exactly against the WGS-84 ellipsoid enlarged by 100 km (a laser grazing lower than that is treated as lost in the atmosphere). Window edges are sharpened by interpolation, so they are accurate to a second or two even with 10 s steps.
* **Eclipse.** The Sun and Earth are discs as seen from the satellite; their overlap gives the fraction of sunlight left (1 = sunlit, 0 = umbra, in between = penumbra). Solar-array power follows that fraction; the radiator's equilibrium temperature follows `ε σ A T⁴ = internal heat + absorbed sunlight`, so it dips in eclipse.
* **Laser link.** The Gaussian beam is `divergence × range` wide at the receiver; the receiver catches the part that lands on its aperture. Received power = transmit power × optics efficiencies × captured fraction × pointing-jitter loss. Free-space path loss `(4πR/λ)²` is reported too (and is checked to agree with the beam calculation in the far field). Noise is the quantum (shot-noise) limit of a coherent receiver plus a configurable "implementation penalty" (10 dB default). BER uses the standard Gray-coded formulas. Throughput is the raw rate minus FEC overhead, and zero when the raw BER is above the FEC limit (3.8×10⁻³).
* **Turret slew.** Each satellite points nadir; the required azimuth/elevation of its laser turret and their rates are computed analytically in that satellite's own orbit frame.

## Default scenario: what it produces

Start 2026-12-21 00:00 UTC, 24 h, 10 s steps, both satellites at 600 km. From `python main_app.py --headless`: 15 eclipses per satellite of about 34.7 min (Alpha's beta angle is −17°), 30 line-of-sight windows (about two per orbit, 49.9 % of the day), range 3100–5260 km, received power −24.4 to −19.9 dBm, latency 10.4–17.5 ms, Doppler about ±2.1 GHz, peak turret slew about 0.10 °/s. With the default 250 Gbaud, QPSK always closes the link (467 Gbit/s after FEC); 8PSK and 16QAM only work at shorter ranges, which the BER and throughput plots show.

## Tests

```bash
pip install -r requirements-dev.txt
```
```bash
python -m pytest -q
```

73 tests pass on macOS (Apple Silicon, Python 3.14); 1 is skipped. They check the code against known values rather than against itself: 600 km orbital period (96.69 min), two-body orbit closing after one period and conserving energy, Sun-synchronous inclination (97.79°) and precession (0.9856°/day, both analytically and from the numerical J2 integration), ISS-like westward regression, Sun declination at the solstice/equinox, GMST at J2000, grazing-limit geometry for the 100 km buffer (43.6° / 5185 km), conical vs. cylindrical shadow edges, eclipse duration (35.5 min at β = 0) and its beta-angle dependence, the 66.1° no-eclipse limit, Stefan–Boltzmann radiator temperature, free-space path loss (258.18 dB at 1000 km, 1550 nm), Gaussian capture fraction, agreement between the antenna-gain and beam-spread views, pointing loss against a Monte-Carlo simulation, Q-function and BPSK BER values, the 9-photons-per-bit quantum limit, Doppler of a 7 km/s closing speed (4.516 GHz), turret rates against finite differences, and that nothing in the code imports `comtypes` / `win32com` / any STK package. A GUI smoke test builds the real window offscreen and checks every pane paints. One optional test compares our Sun with Skyfield's DE421 ephemeris and is skipped unless `data/de421.bsp` is present.

GitHub Actions (`.github/workflows/ci.yml`) runs the same tests on Linux and macOS for every push.

## Honest limits — please read

* **Not validated against Ansys STK or any real hardware.** The checks above show the maths is internally correct and matches textbook values; they do not prove agreement with STK's full models.
* Orbit model: J2 only. No atmospheric drag, solar-radiation pressure, Moon/Sun gravity or precession/nutation (the frame is "mean equator of date"). Start elements are treated as osculating, so the real mean altitude differs from 600 km by a few km and the precession rate is within about 2 % of ideal Sun-synchronous.
* Sun position accuracy ≈ 0.01° (seconds of eclipse timing). Earth shadow ignores atmospheric refraction. Eclipse window edges are placed at the nearest time step (± half a step); line-of-sight edges are interpolated.
* The 100 km buffer is modelled as an ellipsoid with both radii grown by 100 km (not an exact offset surface; the difference is well under a kilometre).
* The link budget is a textbook calculation, not a waveform simulation: no laser phase noise, no acquisition/tracking dynamics, no FEC latency; pointing jitter is applied as an average loss. Absolute numbers depend entirely on the parameters you enter (they are placeholders, not a real terminal's specification). The 100 µrad divergence is much wider than a 30 cm telescope's diffraction limit (about 7 µrad), as specified.
* The thermal model is steady-state (no thermal mass): temperatures jump at eclipse entry/exit. The radiator is a zenith-facing plate with no Earth infrared or albedo.
* The globe is drawn in software (no OpenGL) with a spherical Earth; the map uses a 110 m-resolution coastline. About 14 frames per second at 600× on the author's Mac.
* Tested only on macOS (Apple Silicon, Python 3.14). The Linux/macOS CI is configured; treat Linux as untested until you see it pass. Windows is untried.

## Privacy

Nothing leaves your computer and nothing is saved automatically. The only files the app writes are a CSV (or a PNG with `--screenshot`) that *you* ask for.

## What changed from 0.1

Version 0.1 was a script that drove Ansys STK through its Windows API and was never run against a real STK. It has been removed entirely (no `comtypes`, no `win32com`, no `agi.stk12`), along with its Windows launcher. Everything it estimated is now computed natively here.

## Credits

Land outlines: [Natural Earth](https://www.naturalearthdata.com/) (public domain). Shadow model after Montenbruck & Gill, *Satellite Orbits*. Sun formula from the Astronomical Almanac.
