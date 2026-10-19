# satSTK — Orbital Mission & Laser Link Simulator

*satSTK* is the mission-analysis companion to [satCAD](https://github.com/Saiaarjay09/satCAD): satCAD designs the spacecraft, satSTK simulates its orbit, eclipses and laser link.

A laser **ground station at one fixed place** (you choose it) aims a **1550 nm beam at a satellite** (**Orbital-Node-Alpha**, a 600 km Sun-synchronous orbital AI datacentre node).
The beam crosses the atmosphere, can be interrupted by **random and rare events** (birds, aircraft, storms, earthquakes ...), and you can **tinker with the conditions** and watch
the link budget respond. It runs natively on **macOS and Linux** (and as an app on Windows), needs **no Ansys STK, no licence, no Windows COM, no internet and no account**, and stores nothing unless you export a CSV.

> **What was Orbital-Node-Beta?** In earlier versions it was a second satellite the first one talked to (a satellite-to-satellite link). The beam now starts on the ground, so Beta is
> optional: tick *Add a second satellite* in the Conditions panel to get a second target (the beam can aim at either), plus a computed (not drawn) crosslink between them.

![satSTK window: conditions, 3D globe, 2D map, Gantt timeline, telemetry](docs/screenshot.png)

*(Screenshot rendered by the app itself with `--screenshot`.)*

## Download the app (no Python needed)

The **Releases** page has a `.dmg` for Apple-silicon Macs (drag satSTK into Applications) and a `-setup.exe` installer for 64-bit Windows 10/11. The apps are
unsigned, so the first launch shows a warning: on a Mac use **System Settings → Privacy & Security → Open Anyway**; on Windows press **More info → Run anyway**.
The macOS app was tested on a real Mac; the Windows installer is built and smoke-tested by GitHub Actions but has not been run on a real PC by the author.

## Run it from source (macOS)

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

## How to use it

1. **Place the beam origin.** On launch a map opens: click where the laser ground station stands (or pick a known optical ground station such as Teide, Table Mountain or Hanle, or type coordinates). The green circle is where a 600 km satellite is above the minimum elevation. Press **Start simulation**. (Later: *File → Place ground station*, or **Ctrl/⌘+click** anywhere on the 2D map.)
2. **Watch it.** Press Play (Space). **Next pass** (N) jumps to just before the next time the satellite rises above the mask.
3. **Tinker.** The **Conditions** panel on the left changes the sky, the laser, the satellite and the time; with *Live update* ticked everything recomputes about half a second after you stop typing.
4. **Break things.** Tick *Random events on*, slide the frequency (×0.1 to ×1000), press *Re-roll* for a different "bad day", or open *Choose events / add one by hand* to switch kinds on and off or to drop, say, a volcanic ash cloud at 2.5 h into the run.

## What you see

| Pane | What it shows |
|---|---|
| **3D view** | Shaded globe with day/night terminator, orbit paths, the satellite, the **ground station (green triangle)** and the beam: **green = data flowing, amber = on the satellite but too lossy for usable data, red dashed = blocked** (below the elevation mask, or an event cut it). Drag to rotate, wheel to zoom, the drop-down follows a satellite. The camera is inertial, so you see the Earth turn. |
| **2D map** | World map with the terminator, ground tracks, footprints, the station, its **elevation-mask coverage circle** and the beam line to the sub-satellite point. |
| **Timeline** | Gantt chart: when the satellite is above the mask, when the beam is on it, eclipses (penumbra / umbra), and **rows for events** by category (hover for names and effects). Click or drag to jump; Play / Pause / Step / speed 1×–3600×. |
| **Telemetry dock** | Beam status and reason, elevation, slant range, range rate, latency, Doppler, received power (dBm), SNR, Eb/N0, BER, throughput, best modulation, and the loss breakdown (path loss, divergence, pointing, **clear-air/haze, cloud/fog/rain, turbulence fade, events**), Fried r0, whether it is night at the station, the **events happening now**, ground-telescope and satellite-terminal pointing and slew rates, per-satellite state (lat/lon/alt, eclipse, radiator temperature, solar power), and live plots of range, elevation, received power, channel loss, BER, throughput and radiator temperature. A drop-down picks BPSK / QPSK / 8PSK / 16QAM. |
| **Conditions dock** | Every knob: station (lat/lon/altitude/min elevation), weather preset (clear, hazy, thin cirrus, light/dense fog, overcast, rain, thunderstorm, hot afternoon) and its numbers (visibility, cloud, fog, rain, turbulence strength, wind), satellite altitude / local node time / phase, second satellite and which one the beam aims at, laser power / divergence / jitter / aperture / symbol rate / receiver penalty, start time / duration / time step, radiator and solar-array sizes, and random events. |

**File → Export beam data as CSV** saves the time series (only when you ask). No window? `python main_app.py --headless` prints a text summary; options include `--lat --lon --alt-m --min-elevation --weather Overcast --events --event-seed 3 --event-rate 5 --two-satellites --hours --step --csv out.csv`.

## What is inside (file by file)

| File | Job |
|---|---|
| `orbital_engine.py` | Orbits and geometry: J2 propagator, Sun-synchronous setup, Sun position, frames, **ground station and topocentric elevation/azimuth/rates**, satellite-to-satellite line of sight (Earth ellipsoid + 100 km buffer), Earth shadow (conical / cylindrical), beta angle, radiator temperature and solar-array power. |
| `atmosphere.py` | What the air does to the beam: clear-air/haze (Kim model, air mass), cloud/fog/rain, Hufnagel-Valley turbulence (Fried parameter, Rytov variance, scintillation fade). |
| `events.py` | The event catalogue (16 kinds), seeded Poisson generation, manual events, and how events turn into loss / outage / extra jitter per time step. |
| `laser_link_engine.py` | 1550 nm link budget: range, path loss, Gaussian beam-divergence loss, pointing loss, received power, SNR, latency, BER for BPSK/QPSK/8PSK/16QAM, throughput, Doppler, telescope pointing and slew rates; the ground-to-satellite link adds the atmosphere and events. |
| `simulation.py` | `SimulationConfig` and `run_simulation()` wiring it all together (no GUI code). |
| `main_app.py`, `ui/` | The desktop app, the `--headless` / `--screenshot` / `--selftest` command line, and the Qt widgets. |
| `data/land_110m.json` | Land outlines from Natural Earth (public domain). |
| `tests/` | 122 automated tests (see below). |
| `satstk_app.spec`, `satstk_app.iss` | Packaging (PyInstaller; Inno Setup for the Windows installer). `.github/workflows/app-release.yml` builds both apps and publishes a release when a `v*` tag is pushed. |

Requirements are only `numpy`, `scipy`, `skyfield`, `PySide6-Essentials` (the lighter Qt package; full `PySide6` also works) and `pyqtgraph`; `requirements-dev.txt` adds `pytest`. Skyfield is used for its built-in time scale, which works offline. The Sun uses a short analytic formula, so **no ephemeris file is ever downloaded**.

## The physics in plain words

* **Orbit.** Gravity plus the Earth's equatorial bulge (J2) is integrated numerically. At 97.79° inclination (600 km) J2 turns the orbit plane exactly once a year, keeping a fixed angle to the Sun: that is *Sun-synchronous*. Default local node time is 10:30.
* **Geometry from the ground.** The satellite's direction from the station gives azimuth and elevation; the beam is usable only above your minimum elevation. Doppler uses the satellite's velocity relative to the station, which the spinning Earth carries along at up to 0.46 km/s.
* **Eclipse and thermal.** Sun and Earth are discs seen from the satellite; their overlap gives sunlit / penumbra / umbra. Solar-array power follows it; the radiator's equilibrium temperature follows `ε σ A T⁴ = internal heat + absorbed sunlight`.
* **The beam in space.** The Gaussian beam is `divergence × range` wide at the satellite, which catches only the part landing on its aperture. Received power = transmit power × optics × captured fraction × pointing loss. Free-space path loss `(4πR/λ)²` is reported too and agrees with the beam calculation in the far field. Noise is the quantum limit of a coherent receiver plus an "implementation penalty" (10 dB default). BER uses standard Gray-coded formulas. Throughput is the raw rate minus FEC overhead, and zero when the raw BER is above the FEC limit.
* **The beam in the air.** Extra loss from clear-air/haze (more air near the horizon, less from a mountain top), cloud/fog/rain (you give the straight-up loss; it grows as 1/sin(elevation)), and **turbulence**: a fade margin that is exceeded only 0.1 % of the time, plus slight beam spreading and extra pointing jitter.
* **Events.** Each kind has a typical rate per day, a duration range and an effect (extra dB, complete outage, or shaking). They arrive at random (Poisson), are fixed by the seed, and events shorter than a time step (a 2 s bird in a 10 s step) are applied as the fraction of the step they cover, so throughput drops by that fraction instead of the event vanishing. Aircraft cause outages because real laser stations switch the beam off for eye safety.

## Default scenario: what it produces

Start 2026-12-21 00:00 UTC, 24 h, 10 s steps, ground station at Teide (28.3°N, 16.5°W, 2390 m), 20° minimum elevation, clear sky, no events. From `python main_app.py --headless`: three passes with the beam on the satellite, about 10.5 minutes in total (a low-Earth satellite is only overhead for minutes), slant range 626–1385 km, peak elevation 73°, received power −20 to −9 dBm, atmosphere and turbulence loss 3–7 dB, Doppler about ±4 GHz, peak ground-telescope slew 0.7 °/s. With the default 250 Gbaud, the best modulation delivers up to about 73 TB in those minutes (hypothetical, for the placeholder terminal). The `Overcast` preset (25 dB of cloud) leaves the geometry intact but no usable data; the beam turns amber.

## Tests

```bash
pip install -r requirements-dev.txt
```
```bash
python -m pytest -q
```

122 tests pass on macOS (Apple Silicon, Python 3.14); 1 is skipped. They check the code against known values rather than against itself: 600 km orbital period (96.69 min), two-body orbit closing after one period and conserving energy, Sun-synchronous inclination (97.79°) and precession (0.9856°/day, both analytically and from the numerical J2 integration), ISS-like westward regression, Sun declination at the solstice/equinox, GMST at J2000, grazing-limit geometry for the 100 km buffer (43.6° / 5185 km), conical vs. cylindrical shadow edges, eclipse duration (35.5 min at β = 0) and its beta-angle dependence, the 66.1° no-eclipse limit, Stefan–Boltzmann radiator temperature, free-space path loss (258.18 dB at 1000 km, 1550 nm), Gaussian capture fraction, agreement between the antenna-gain and beam-spread views, pointing loss against a Monte-Carlo simulation, Q-function and BPSK BER values, the 9-photons-per-bit quantum limit, Doppler of a 7 km/s closing speed (4.516 GHz), ground-station geometry (overhead = 90° and range = altitude, azimuth of N/E/S/W points, horizon distance, analytic az/el rates against finite differences), air mass and Kim visibility values, Hufnagel-Valley Fried parameter and scintillation behaviour, a log-normal fade against Monte Carlo, the ground link budget worked by hand at the peak of a pass, event counts against the Poisson rates over hundreds of seeds, fractional-step event application, and events blocking or splitting a pass in the full simulation, and that nothing in the code imports `comtypes` / `win32com` / any STK package. A GUI smoke test builds the real window offscreen and checks every pane paints. One optional test compares our Sun with Skyfield's DE421 ephemeris and is skipped unless `data/de421.bsp` is present.

GitHub Actions (`.github/workflows/ci.yml`) runs the same tests on Linux and macOS for every push.

## Honest limits — please read

* **Not validated against Ansys STK or any real hardware.** The checks above show the maths is internally correct and matches textbook values; they do not prove agreement with STK's full models.
* Orbit model: J2 only. No atmospheric drag, solar-radiation pressure, Moon/Sun gravity or precession/nutation (the frame is "mean equator of date"). Start elements are treated as osculating, so the real mean altitude differs from 600 km by a few km and the precession rate is within about 2 % of ideal Sun-synchronous.
* Sun position accuracy ≈ 0.01° (seconds of eclipse timing). Earth shadow ignores atmospheric refraction. Eclipse window edges are placed at the nearest time step (± half a step); line-of-sight edges are interpolated.
* The 100 km buffer is modelled as an ellipsoid with both radii grown by 100 km (not an exact offset surface; the difference is well under a kilometre).
* The link budget is a textbook calculation, not a waveform simulation: no laser phase noise, no acquisition/tracking dynamics, no FEC latency; pointing jitter is applied as an average loss. Absolute numbers depend entirely on the parameters you enter (they are placeholders, not a real terminal's specification). The 100 µrad divergence is much wider than a 30 cm telescope's diffraction limit (about 7 µrad), as specified.
* **Atmosphere, weather and turbulence are rules of thumb**, not a validated channel model or a forecast: cloud/fog losses are numbers you set (the presets are round, illustrative values), rain uses a common optical rule (1.076 R^0.67 dB/km over a 2 km layer), turbulence is a Hufnagel-Valley profile with a plane-wave Rytov approximation (a conservative stand-in for an uplink), and the turbulence fade is a 0.1 %-outage margin rather than a time series of flickering. Real sites differ hugely.
* **Events are illustrative.** Their rates, durations and effects are plausible guesses chosen so interesting things happen in a 24 h run, not statistics for any place; the Poisson assumption (independent, constant rate) is a simplification (storms cluster, bird crossings depend on the season and site). Overlapping losses add in dB; an outage is a total loss. Events apply to the beam to the target satellite only (satellite-side events such as safe mode are treated as ground-visible outages).
* The ground station is a point on the WGS-84 ellipsoid with a flat elevation mask: no terrain, buildings or trees, no sun-exclusion angle, no daytime sky-background noise, no uplink/downlink distinction beyond the beam spreading. Only the first pair of satellites' crosslink is computed, and it is reported in the summary but not drawn.
* The thermal model is steady-state (no thermal mass): temperatures jump at eclipse entry/exit. The radiator is a zenith-facing plate with no Earth infrared or albedo.
* The globe is drawn in software (no OpenGL) with a spherical Earth; the map uses a 110 m-resolution coastline. About 15-20 frames per second on the author's Mac.
* Tested by the author only on macOS (Apple Silicon, Python 3.14). GitHub's automated tests passed on Linux and macOS for the previous version; the Windows installer is built and smoke-tested by GitHub Actions but has not been run on a real PC.

## Privacy

Nothing leaves your computer and nothing is saved automatically. The only files the app writes are a CSV (or a PNG with `--screenshot`) that *you* ask for.

## What changed from 0.1

Version 0.1 was a script that drove Ansys STK through its Windows API and was never run against a real STK. It has been removed entirely (no `comtypes`, no `win32com`, no `agi.stk12`), along with its Windows launcher. Everything it estimated is now computed natively here.

## Credits

Land outlines: [Natural Earth](https://www.naturalearthdata.com/) (public domain). Shadow model after Montenbruck & Gill, *Satellite Orbits*. Sun formula from the Astronomical Almanac.
