# v2.0.0: Continuous 24/7 GEO Optical Relay Architecture

satSTK 2.0 adds a **geostationary optical relay network** so the laser link from the ground station to the 600 km LEO datacentre node (**Orbital-Node-Alpha**) no longer drops when the satellite sinks below the horizon.
From the default **South India ground station (13.0° N, 77.5° E)** the direct beam alone carries usable data for only **1.25 % of the day** (three passes, about 18 minutes). With the relay network the link is up
**100.000 % of the day in clear sky**.

### What is new

- **Three GEO relay satellites** (0° inclination, 35,786 km, one orbit per sidereal day, 23 h 56 min 04 s): `GEO-Relay-IND` at 77.5° E (74.7° elevation from the station, always visible), `GEO-Relay-ATL` at 45° W, `GEO-Relay-PAC` at 165° W. Together they see every point of the 600 km shell.
- **Dynamic routing every time step, with make-before-break handovers:** (1) direct LEO beam while the satellite is above the 10° mask (≈ 2–7 ms), (2) ground → GEO-IND → satellite (≈ 240–270 ms), (3) ground → GEO-IND → GEO-ATL/PAC → satellite (≈ 480–510 ms). 38 handovers per day, all hitless in clear sky.
- **Relay link budget:** 300 mm telescopes, 10 W EDFAs and 10 µrad beams on the relay hops (35,000–44,000 km, about 289 dB free-space loss), cascaded bent-pipe SNR, atmosphere on the ground hop.
- **GUI:** the 35,786 km GEO ring on the 3D globe with a **true / compressed scale** switch; the active path in **neon green (direct)** or **bright cyan (via GEO)**; a live **Active Route** indicator (`GS-INDIA -> GEO-IND -> NODE-ALPHA`), end-to-end latency, **contact uptime %**, **cumulative data (TB)** and handover counts; a **GEO relay ON / OFF** switch to compare against the direct-only link.
- The default ground station is now GS-INDIA with a 10° minimum elevation. Everything from 0.3 is still there: placing the beam origin on a map, the Conditions panel, weather and turbulence, and 16 kinds of random and rare events.

### Download the file for your computer (below, under Assets)

| System | File |
|---|---|
| Mac with Apple silicon (M1 or newer) | `satSTK-…-macos-arm64.dmg`: open it and drag satSTK into Applications |
| Windows 10/11, 64-bit | `satSTK-…-windows-x64-setup.exe`: run the installer |

Intel Macs are not built (run from source: see the README).

### The first time you open it

The apps are **not code-signed**, so your computer will warn you:

- **Mac:** "satSTK can't be opened because Apple cannot check it." Open **System Settings → Privacy & Security**, scroll down and press **Open Anyway**. Once only.
- **Windows:** a blue "Windows protected your PC" screen. Press **More info → Run anyway**.

### Please note (honest limits)

- **100 % means 100 % of the time in clear sky with no events.** Every path starts in the air above the ground station, so heavy cloud or fog there still takes the link down; events and satellite faults hit every path too.
- At the default 250 Gbaud the relay paths run at a raw BER of about 1–2×10⁻³ and rely on the 7 % FEC (modelled as a threshold, not a decoder). At 50 Gbaud, BPSK has a raw BER below 10⁻⁹ on every path.
- The GEO model is idealised (perfectly station-kept relays, simple point-to-point budgets, `Σ 1/SNR` noise accumulation, no acquisition delay or packet re-ordering). Relay terminal beam width and jitter are my assumptions. **Not validated against Ansys STK or real hardware.**
- The macOS app is built and tested on an Apple-silicon Mac; the Windows installer is built and smoke-tested by GitHub's automated build but has not been run on a real PC by the author.
