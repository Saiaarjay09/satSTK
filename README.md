# Setu: STK automation for the orbital laser crosslink

*Setu* is Sanskrit for "bridge": the laser link between two orbiting compute nodes. Part of the Orbital AI Datacentre project (see the Orbital-Datacentre repository for the optical-channel simulator).

`setu_stk.py` drives **Ansys STK** from Python. It builds a 24-hour scenario with two 600 km Sun-synchronous
satellites (**Orbital-Node-Alpha**, **Orbital-Node-Beta**), reports **when Alpha is in sunlight, penumbra and umbra** (the cases the
mechanically pumped fluid loop must survive), fits Alpha with a **1550 nm laser transmitter** and Beta with a **laser receiver**
(100 microradian beam), computes the **line-of-sight access** between them and extracts **received power, BER and latency**
during each window.

> **Status, honestly.** The script was written from Ansys' public STK documentation and has **not been run against a live STK**
> (STK was not available where it was written). The orbit mathematics and the plumbing are unit-tested
> (`python -m pytest tests/test_setu_stk.py`), and `--offline-preview` gives an independent analytic answer to compare with. Expect to
> fix a name or two on the first real run; the script prints what STK rejected and saves every attribute STK exposes in
> `stk_output/transmitter_attributes.txt` and `receiver_attributes.txt`, so any difference is quick to correct.

## What you need

| Need | Notes |
|---|---|
| **Windows 10/11** with **Ansys STK 12.x** installed and licensed | STK's desktop application is Windows-only (Linux has STK Engine only; macOS is not supported) |
| The **Communications** capability in your STK licence | needed for transmitters, receivers and link budgets |
| **Python 3.9 to 3.12** (match your STK release notes) | from python.org |
| The STK Python API, `agi.stk12` | a `.whl` file that ships **inside the STK install folder** (search for `agi.stk12*.whl`, commonly under `...\STK 12\bin\AgPythonAPI\`) |
| `numpy` | for the offline preview and the orbit maths |

## Install and run (Windows, PowerShell)

```powershell
python -m venv stkenv
stkenv\Scripts\activate
pip install numpy comtypes
pip install "C:\Program Files\AGI\STK 12\bin\AgPythonAPI\agi.stk12-<version>-py3-none-any.whl"   # use the real file name on your PC
python setu_stk.py
```

STK opens, builds the scenario while you watch, and prints the results. Files go to `stk_output\`:

| File | Contents |
|---|---|
| `alpha_lighting.csv`, `alpha_lighting_report_raw.txt` | sunlight / penumbra / umbra intervals for Node Alpha (STK "Lighting Times" report) |
| `access_windows.csv` | start, stop and duration of each Alpha to Beta line-of-sight window |
| `link_budget.csv` | per time step: range, received isotropic power, carrier power, Eb/No, BER, link margin, propagation delay |
| `transmitter_attributes.txt`, `receiver_attributes.txt` | every setting STK exposes for the two laser objects |
| `Orbital_AI_Datacentre.sc` | the scenario, which you can open in STK |

Other ways to run it:

```powershell
python setu_stk.py --offline-preview   # no STK: analytic eclipse + line-of-sight estimate (works on any computer with numpy)
python setu_stk.py --engine            # STK Engine, no window (also works on Linux)
python setu_stk.py --keep-open         # leave STK open to explore the scenario
```

`--offline-preview` is worth running first, even on a Mac: it prints the Sun-synchronous inclination (97.79 degrees at 600 km), Alpha's
eclipse passes (about 35 minutes of shadow every 96.7-minute orbit at this altitude) and the line-of-sight windows (about 18 minutes at a
time, 4,300 to 5,200 km range, with the default offsets). It is a simple circular-orbit and cylindrical-shadow model: **STK's numbers are the
ones to use**, and agreement within a few minutes is the sanity check.

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
