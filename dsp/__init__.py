"""
dsp - symbol-level optical physical layer, migrated from the Orbital-Datacentre three-module simulator:

    transmitter.py   payload bytes -> framing (length + CRC-32) -> Gray-mapped BPSK/QPSK/8PSK/16QAM -> RRC pulse shaping -> IQ Mach-Zehnder modulator
    channel.py       one vacuum laser hop: Gaussian beam optics, pointing jitter, laser phase noise, receiver noise, beacon power
    receiver.py      PAT (spiral search, quadrant detector, PI-controlled Fast Steering Mirror) + matched filter, preamble sync, carrier PLL, CRC-32

satSTK's optical_dsp_bridge.py chains these through the live orbital state (direct LEO, 2-hop or 3-hop GEO relay). Pure NumPy / SciPy.
"""
