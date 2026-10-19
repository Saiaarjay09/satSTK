## satSTK 0.3.0: laser ground station → satellite

satSTK now simulates what the name promises: a **laser beam from one fixed place on the ground to a satellite**. **You place the beam origin on a map, then simulate.**
The beam crosses the real atmosphere (haze, cloud, fog, rain, turbulence), and a **Conditions panel** lets you change almost everything and watch the result
update live. Switch on **random and rare events** (birds, aircraft, storms, earthquakes, volcanic ash, solar storms ...) or place one by hand.

**Download the file for your computer** (below, under Assets):

| System | File |
|---|---|
| Mac with Apple silicon (M1 or newer) | `satSTK-…-macos-arm64.dmg`: open it and drag satSTK into Applications |
| Windows 10/11, 64-bit | `satSTK-…-windows-x64-setup.exe`: run the installer |

Intel Macs are not built (run from source: see the README).

### What changed from 0.2

- One satellite (Orbital-Node-Alpha) is the default. Orbital-Node-Beta, the old second node, is now an optional second satellite you can add and aim the beam at.
- The beam starts at a ground station: pick it on the map, from a list of known optical ground stations, or type coordinates.
- New: elevation mask, atmosphere/weather/turbulence losses, 16 event types, a Conditions panel, a "Next pass" button, ground-telescope slew rates.

### The first time you open it

The apps are **not code-signed**, so your computer will warn you:

- **Mac:** "satSTK can't be opened because Apple cannot check it." Open **System Settings → Privacy & Security**, scroll down and press **Open Anyway**. Once only.
- **Windows:** a blue "Windows protected your PC" screen. Press **More info → Run anyway**.

### Please note

- **Not validated against Ansys STK or real hardware.** The weather, turbulence and event models are engineering rules of thumb and illustrative rates, not forecasts or statistics. See "Honest limits" in the README.
- The macOS app is built and tested on an Apple-silicon Mac; the Windows installer is built and smoke-tested by GitHub's automated build but has not been run on a real PC by the author.
