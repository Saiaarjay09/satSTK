# v2.3.0: Radiation-Tolerant 32-Blade PyTorch Scheduler & SAA Bit-Flip Recovery

satSTK 2.3 trains a real neural network across the **32 server blades of the Alpha-Sigma orbital datacentre** (a 1:1 satCAD design, 2 banks × 16 trays) while cosmic-ray bit flips, driven by the satellite's live latitude / longitude / altitude, corrupt the blades' weights and gradients, and shows how the system heals itself without pausing the other blades.

### What is new

- **Radiation model (`seu_injector.py`)**: 1e-5 upsets per blade per second at nominal LEO, **15× over the polar horns** (|lat| > 60°) and **100× over the South Atlantic Anomaly** (lat −50…0°, lon −90…+40°), plus an altitude factor and a 1000× solar-flare switch. `inject_ieee754_bitflip` flips one random bit of a float32 / bfloat16 / float16 / float64 NumPy array or torch tensor with XOR and reports whether it hit the **mantissa** (noise), **sign** (inverted weight) or **exponent** (up to ~1e38, Inf or NaN). A rare **Single-Event Latchup** takes a blade offline for 15 s until the 48 V breaker power-cycles it.
- **Fault-tolerant 32-blade orchestrator (`radiation_pytorch_scheduler.py`)** with four layers: a gradient **sanitizer** (CRC-32, L2 norm, NaN/Inf, quarantine above 5× the median); **TMR shielding** (10 triads, 2-of-3 gradient voting, 2 hot spares) switched on when the orbit predicts the SAA; **ECC parity + a SHA-256 verified OBC checkpoint** that restores a damaged blade over a modelled 100 Gbps backplane while the other 31 keep training; and **thermal coupling**: at 75 °C Bank B hibernates and Bank A takes the whole batch.
- **New tab "Radiation-Tolerant AI Scheduler"**: a live 2×16 blade grid (green healthy, yellow = mantissa flip scrubbed by ECC, red = exponent NaN strike caught, purple = TMR shielding, grey = hibernated or power-cycling), a protected-vs-unprotected loss chart (the unprotected baseline goes NaN when upsets hit; the protected one converges), the **SAA and auroral glow on the 2D map and the 3D globe**, and chaos controls: **Force Solar Flare / CME (1000× flux)**, **Toggle ECC & TMR Protection**, **Export Recovered .pt Weights**, which can fire the file straight through the existing **16QAM optical downlink**.
- **PyTorch or NumPy**: `torch` is now in `requirements.txt`. Without it a built-in NumPy engine does the same job (same loss and gradients to 1e-5); the packaged desktop apps use the NumPy engine.
- Command line: `python radiation_pytorch_scheduler.py --hours 3` (add `--flare`, `--export weights.pt`), and `python main_app.py --screenshot shot.png --show-radiation`.
- Includes the digital-twin telemetry stream to satCAD that was added after v2.1.0.

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

- **The radiation numbers are simulation assumptions, not measurements**: the upset rates, the 15× / 100× multipliers, the zone boxes (the real SAA is not a rectangle), the 15 s latchup recovery, the 100 Gbps backplane and the 5× norm threshold are placeholders, not from flight data or beam tests. **Nothing here is flight-qualified** and no hardware fault rate is claimed. The 32 "blades" are replicas in one Python process, and the workload is a small MLP, not a language model.
- A random bit is rarely fatal (about 1 hit in 32 flips bit 30), so one SAA pass can leave the unprotected twin alive; the flare button makes its failure certain. The protected system survived every test, which is evidence, not a guarantee.
- Reed-Solomon coding was not implemented (SHA-256 plus a mirrored copy protect the checkpoint). The workload is not fed back into the thermal model.
- All v2.1 limits still apply (J2-only orbit, no FEC in the DSP, rules-of-thumb atmosphere). Not validated against Ansys STK or real hardware.
- The macOS app is built and tested on an Apple-silicon Mac; the Windows installer is built and smoke-tested by GitHub's automated build but has not been run on a real PC by the author.
