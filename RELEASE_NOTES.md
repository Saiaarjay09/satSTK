# v2.4.0: Eclipse BMS, 48V Busbar Physics & Autonomous DVFS Load Shedding

satSTK 2.4 adds the **electrical power subsystem** of the Alpha-Sigma orbital AI datacentre: the solar array, an 8-module Li-Ion battery with its BMS, the four 48 V copper busbars and an autonomous load-shedding state machine, solved every time step in a closed loop with the cooling loop and the AI scheduler.

### What is new

- **`eps_power_engine.py`**: a **182.4 m², 32 % GaAs array** (derated 0.25 %/°C above 28 °C, cos θ from the beta angle of the Sun-tracking wings, illumination and Sun distance from the orbit engine); a **320 kWh, 8 × 40 kWh pack** with a Shepherd open-circuit curve and a Thevenin R0 + R1‖C1 branch behind the BCDU, Coulomb-counted SoC, C-rate, Joule heat fed into the cooling loop; and **four 60 × 15 mm copper busbars** solved exactly for the voltage at each of the 32 blades with ρ(T), checked against the 45.5–50.0 V server VRM band.
- **Four-mode power state machine**: *sunlit peak* (full compute, battery topped up), *sunlit recharge* (35 % of the array reserved for the charger), *eclipse cruise* (DVFS eco-clock: 60 % power, 80 % speed) and *eclipse survival* below 35 % SoC (checkpoint flush to the OBC, Bank B 48 V breaker open, OISL lasers low-power, pumps at 40 % RPM). Below 25 % SoC Bank A is shed too (my addition).
- **New tab "EPS Power, Battery & 48V Busbar"**: a live power-flow schematic, the SoC / solar / load sawtooth over the whole scenario with every eclipse shaded, the delivered volts at Blade 1 … Blade 32, and stress tests: **Simulate 1-Panel Solar Wing Failure**, **Force Extended 45-Min Eclipse**, **Battery Cell Degradation** slider.
- **Closed loops**: `thermal_loop.simulate_loop` accepts a coupler (the EPS supplies the real waste heat and the pump speed); the AI scheduler hibernates Bank B when the breaker opens, runs at the DVFS speed and flushes its checkpoint when survival starts.
- **Digital twin**: the UDP packet to satCAD now carries `eps_state` (`soc_pct`, `busbar_current_a`, `busbar_temp_c`, `bank_b_breaker_closed`, plus mode, solar / load kW and the lowest blade voltage). Older viewers ignore it. The satCAD-side visualisation is a separate repository and is not part of this release.
- `python eps_power_engine.py --hours 24` prints a per-eclipse report; `python main_app.py --screenshot shot.png --show-eps` renders the tab.
- Fixes a Windows-only bug in the satCAD status listener (it stopped after an oversized datagram). Includes the v2.3.0 radiation-tolerant scheduler.

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

- **In the default scenario the power system is relaxed**: with a 1.2 kW-per-blade load (a **placeholder**, chosen so that the 182.4 m² array can sustain and recharge it; 768 real accelerators would draw far more), a 35-minute eclipse uses 5.3 % of the 320 kWh. The 35 % survival and 20 % floor logic is reached only with the stress tests. At 91 % capacity lost and 45-minute eclipses the floor holds (lowest SoC 27 %); at 95 % lost and 60-minute eclipses it does not (12.8 %), and the tab says so.
- **The 48 V busbar drop is about 0.1 V** (worst blade 47.89 V at 801 A), not a volt: that is what 8 m of 900 mm² copper gives. The delivered voltage is always inside the 45.5–50 V band here.
- Only the array area, efficiency, solar constant, battery size, bus voltage and busbar geometry are the design's numbers. The converter efficiencies, battery resistances, array and busbar temperature models, DVFS factors, hysteresis bands and the 160 m² radiator used for this loop are placeholders. Nothing is a flight-qualified power budget.
- Not modelled: the OISL low-power state in the link budget, an abrupt end for a stretched eclipse (no penumbra), albedo, cell balancing, battery temperature dynamics. `thermal_engine.py` does not exist; the cooling loop is `thermal_loop.py`.
- All earlier limits still apply (the radiation numbers are simulation assumptions, J2-only orbit, no FEC in the DSP). The macOS app is built and tested on an Apple-silicon Mac; the Windows installer is built and smoke-tested by GitHub's automated build but has not been run on a real PC by the author.
