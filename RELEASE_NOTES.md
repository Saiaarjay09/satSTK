# v2.1.0: Integrated End-to-End Optical DSP & Payload Transmission

satSTK 2.1 lets you **fire a real file or message through the live orbital simulation, symbol by symbol**. The three physical-layer modules of the Orbital-Datacentre optical link simulator (IQ Mach-Zehnder transmitter, vacuum laser channel, PAT + demodulating receiver) are now part of satSTK (`dsp/`), and a new bridge feeds them the simulation's real orbital state.

### What is new

- **Optical Payload Transmitter dock** (a tab beside the Telemetry dock): type text, load a file, or press **Generate 63 KB Test Payload** (1,008 self-checking 64-byte lines); choose BPSK / QPSK / 8PSK / 16QAM or **Auto-Adaptive**, laser power, beacon power, symbol rate, **FSM closed-loop ON/OFF**; press **Fire Laser Payload**.
- **Pick any moment and any route:** direct LEO, 2-hop via GEO-IND, or 3-hop via GEO-IND and GEO-ATL/PAC. Each hop uses the simulation's real range, aperture, power, beam width, pointing jitter and (on the ground hop) the atmosphere and events at that time; the GEO relays receive, add noise, and re-amplify with their EDFAs.
- **End-to-end diagnostics:** IQ constellation, Fast Steering Mirror quadrant-detector trajectory (spiral search then PI tracking onto the centre), side-by-side payload diff (wrong bytes in red), CRC-32 pass/fail badge, exact BER, bytes wrong, 64-byte blocks bad, frame loss %, end-to-end latency, PAT time to lock.
- **Auto-Adaptive modulation:** predicts the route's end-to-end Es/N0 and picks the densest scheme that should still pass the CRC; steps down from 16QAM to 8PSK/QPSK/BPSK as the route gets longer or the symbols faster.
- **Continuous file stream:** slices a file into frames and sends them while the satellite moves, over the route in force at each moment. Example: the 63 KB payload streamed across a GEO → direct → GEO handover delivers 63 of 63 frames and a matching whole-file CRC, with every handover make-before-break.
- Command line: `--send-text`, `--send-file`, `--send-test-payload`, `--stream`, `--route`, `--mod`, `--dsp-rate`, `--no-fsm`. Everything is NumPy/SciPy only.

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

- The DSP default is **10 Gbaud**, not the simulation's 250 Gbaud: the DSP has **no forward error correction**, and its real receiver performs about 3 dB worse than textbook theory, so at 250 Gbaud a 63 KB frame cannot pass the CRC over the GEO paths (the tests check this).
- Doppler is assumed removed by the receiver; relays are idealised amplifiers and assumed locked (only the satellite terminal runs PAT); scintillation is one random fade per frame. Not validated against Ansys STK or real hardware.
- The macOS app is built and tested on an Apple-silicon Mac; the Windows installer is built and smoke-tested by GitHub's automated build but has not been run on a real PC by the author. A 63 KB BPSK payload peaks at about 1 GB of memory.
