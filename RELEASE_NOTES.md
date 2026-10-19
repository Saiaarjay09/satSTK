## satSTK 0.2.0 - standalone simulator

satSTK no longer needs Ansys STK, a licence, Windows or COM. It is now a native Python orbital mission and 1550 nm laser-link simulator
with a mission-control desktop app (3D globe, 2D map, Gantt timeline, telemetry and live plots).

- Physics: J2 orbit propagation, exact line of sight over the Earth ellipsoid + 100 km atmosphere, conical/cylindrical eclipse,
  radiator and solar-array model, link budget, BER for BPSK/QPSK/8PSK/16QAM, Doppler and turret slew rates.
- Runs on macOS and Linux (tested on macOS); no network needed.
- Download: unzip, then follow the three setup commands in README.md.
- Not validated against Ansys STK or real hardware. See "Honest limits" in the README.
