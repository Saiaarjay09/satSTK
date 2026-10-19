"""The migrated DSP chain on its own: framing, constellations, pulse shaping, modulator, channel optics, PAT / FSM, demodulation, test payload."""
import math
import zlib

import numpy as np
import pytest

from dsp import channel as dch
from dsp import payload as pl
from dsp import receiver as drx
from dsp import transmitter as dtx

LAM = 1550e-9


def _loopback(payload, mod, link=None, seed=1, closed_loop=True, rate=1e9, power=1.0, pre_locked=False):
    link = link or dch.LinkConfig(range_km=1500.0)
    sig = dtx.transmit(payload, mod, rate, 8, 0.35, 16, LAM, power)
    beam, metrics, bud, geo = dch.propagate(sig, link, np.random.default_rng(seed), seed)
    rep = drx.receive(beam, drx.PATConfig(), np.random.default_rng(seed + 1), closed_loop, pre_locked)
    return sig, beam, metrics, bud, rep


# ---- framing and constellations -----------------------------------------------------------------------------------------
def test_frame_round_trip_and_crc_detects_a_single_flipped_bit():
    data = bytes(range(200))
    bits = dtx.build_frame_bits(data)
    assert len(bits) == 32 + 8 * 200 + 32
    got, ok, status = dtx.parse_frame_bits(bits)
    assert ok and got == data and "OK" in status
    bad = bits.copy()
    bad[100] ^= 1
    got, ok, status = dtx.parse_frame_bits(bad)
    assert not ok and "MISMATCH" in status
    hdr = bits.copy()
    hdr[0] ^= 1                                               # corrupt the length field: must not allocate gigabytes or crash
    assert dtx.parse_frame_bits(hdr)[0] is None


def test_bits_bytes_round_trip():
    data = bytes(np.random.default_rng(1).integers(0, 256, 100, dtype=np.uint8))
    assert dtx.bits_to_bytes(dtx.bytes_to_bits(data)) == data


@pytest.mark.parametrize("mod,n", [("BPSK", 2), ("QPSK", 4), ("8PSK", 8), ("16QAM", 16)])
def test_constellations_have_unit_energy_and_slicer_inverts_the_mapping(mod, n):
    pts = dtx.constellation(mod)
    assert len(pts) == n and np.mean(np.abs(pts) ** 2) == pytest.approx(1.0)
    bps = dtx.BITS_PER_SYMBOL[mod]
    bits = np.random.default_rng(2).integers(0, 2, bps * 500, dtype=np.uint8)
    syms, pad = dtx.bits_to_symbols(bits, mod)
    assert pad == 0 and np.array_equal(dtx.symbols_to_bits(syms, mod), bits)


@pytest.mark.parametrize("mod", ["QPSK", "8PSK", "16QAM"])
def test_nearest_neighbours_differ_by_exactly_one_bit_gray_coding(mod):
    pts = dtx.constellation(mod)
    d = np.abs(pts[:, None] - pts[None, :]) + np.eye(len(pts)) * 99
    dmin = d.min()
    for a in range(len(pts)):
        for b in range(len(pts)):
            if a != b and d[a, b] < dmin * 1.01:
                assert bin(a ^ b).count("1") == 1


def test_rrc_filter_is_unit_energy_and_two_of_them_make_a_nyquist_pulse():
    sps = 8
    h = dtx.rrc_taps(0.35, sps, 16)
    assert np.sum(h ** 2) == pytest.approx(1.0)
    rc = np.convolve(h, h)
    centre = len(rc) // 2
    assert rc[centre] == pytest.approx(1.0, abs=1e-3)
    assert np.max(np.abs(rc[centre + sps::sps][:8])) < 5e-3                    # zero crossings at every symbol period: no inter-symbol interference


def test_mach_zehnder_predistortion_gives_an_exact_copy_and_linear_drive_compresses():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000) + 1j * rng.normal(size=5000)
    peak = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)))
    ideal = dtx.iq_mach_zehnder(x, drive=0.7, predistort=True)
    assert np.max(np.abs(ideal - 0.7 * x / peak)) < 1e-9                                  # arcsin pre-distortion undoes the sin() transfer exactly
    linear = dtx.iq_mach_zehnder(x, drive=0.95, predistort=False)
    ref = 0.95 * x / peak
    k = np.vdot(ref, linear) / np.vdot(ref, ref)                                         # best complex scale, as the transmitter measures fidelity
    err = np.mean(np.abs(linear - k * ref) ** 2) / np.mean(np.abs(linear) ** 2)
    assert err > 1e-4                                                                    # driven linearly, the sine transfer visibly distorts the waveform
    clean = dtx.transmit(b"m" * 400, "16QAM", 1e9, 8, 0.35, 16, LAM, 1.0, drive=0.95, predistort=True)
    dirty = dtx.transmit(b"m" * 400, "16QAM", 1e9, 8, 0.35, 16, LAM, 1.0, drive=0.95, predistort=False)
    assert clean.modulator_error_db < -100 and dirty.modulator_error_db > -60


def test_transmit_sets_launch_power_and_preamble():
    sig = dtx.transmit(b"abc" * 50, "16QAM", 2e9, 8, 0.35, 16, LAM, 2.5, n_preamble=64)
    assert np.mean(np.abs(sig.field) ** 2) == pytest.approx(2.5)
    assert sig.n_symbols == 64 + sig.n_data_symbols and sig.bit_rate == pytest.approx(8e9) and sig.n_preamble == 64
    with pytest.raises(ValueError):
        dtx.transmit(b"", "QPSK")


# ---- channel optics ------------------------------------------------------------------------------------------------------
def test_gaussian_beam_budget_matches_a_hand_calculation():
    cfg = dch.LinkConfig(range_km=2000.0, tx_aperture_m=0.10, tx_fill=0.89, rx_aperture_m=0.30, tx_jitter_urad=0.0, point_ahead_residual_urad=0.0)
    geo = dch.resolve_geometry(cfg)
    bud = dch.beam_budget(cfg, geo, LAM)
    w0 = 0.89 * 0.05
    zr = math.pi * w0 ** 2 / LAM
    w = w0 * math.sqrt(1 + (2.0e6 / zr) ** 2)
    assert bud.waist_m == pytest.approx(w0) and bud.rayleigh_range_m == pytest.approx(zr)
    assert bud.spot_radius_m == pytest.approx(w) and bud.divergence_half_urad == pytest.approx(w / 2.0e6 * 1e6)
    assert bud.eta_geometric == pytest.approx(1 - math.exp(-2 * 0.15 ** 2 / w ** 2))
    assert bud.eta_tx_trunc == pytest.approx(1 - math.exp(-2 / 0.89 ** 2))
    assert bud.expected_pointing_eff == pytest.approx(1.0)                      # no jitter, no bias


def test_pointing_efficiency_expectation_matches_monte_carlo():
    rng = np.random.default_rng(6)
    sigma, bias, div = 12e-6, 8e-6, 40e-6
    e = rng.normal(0, sigma, (400_000, 2)) + np.array([bias, -bias / 2])
    mc = np.mean(np.exp(-2 * np.sum(e ** 2, axis=1) / div ** 2))
    assert dch.pointing_efficiency_expectation(bias, -bias / 2, sigma, div) == pytest.approx(mc, rel=0.01)


def test_ar1_jitter_has_the_requested_rms():
    y = dch.ar1_process(400_000, 1.5e-6, 500.0, 1e6, np.random.default_rng(1))
    assert np.std(y) == pytest.approx(1.5e-6, rel=0.05)


def test_channel_latency_is_range_over_c_and_geometry_checks_the_limb():
    cfg = dch.LinkConfig(range_km=3000.0, altitude_km=600.0)
    geo = dch.resolve_geometry(cfg)
    assert geo.range_m == 3.0e6 and not np.isnan(geo.separation_deg)
    sig = dtx.transmit(b"x" * 32, "QPSK", 1e9)
    _, m, _, _ = dch.propagate(sig, cfg, np.random.default_rng(1), 1)
    assert m.propagation_ms == pytest.approx(3.0e6 / dtx.C_LIGHT * 1e3) and m.latency_ms == pytest.approx(m.propagation_ms + m.serialization_ms)
    assert not dch.resolve_geometry(dch.LinkConfig(range_km=10000.0, altitude_km=600.0)).los_ok        # a 10,000 km chord at 600 km altitude dips into the Earth


def test_theoretical_ber_textbook_values():
    assert float(dch.theoretical_ber("BPSK", 1.0)) == pytest.approx(0.5 * math.erfc(1.0))               # Q(sqrt(2 Es/N0)) with Es = Eb
    assert float(dch.theoretical_ber("QPSK", 2.0)) == pytest.approx(float(dch.theoretical_ber("BPSK", 1.0)))   # same BER at the same Eb/N0 (Es = 2 Eb)
    assert float(dch.theoretical_ber("16QAM", 100.0)) > float(dch.theoretical_ber("QPSK", 100.0))


def test_oracle_receiver_ber_matches_theory_over_a_noisy_vacuum_hop():
    """Starve the link of power until the BER is a few 1e-3 and compare the Monte-Carlo BER with the Gray-coded AWGN formula (same channel realisation)."""
    data = bytes(np.random.default_rng(8).integers(0, 256, 6000, dtype=np.uint8))
    link = dch.LinkConfig(range_km=1500.0, tx_jitter_urad=0.0, laser_linewidth_hz=0.0)
    meas, theo = [], []
    for seed in range(1, 6):
        sig = dtx.transmit(data, "QPSK", 1e9, 8, 0.35, 16, LAM, 2.0e-4)               # Es/N0 about 9 dB at this range
        _, m, _, _ = dch.propagate(sig, link, np.random.default_rng(seed), seed)
        meas.append(m.ber)
        theo.append(m.ber_theory)
    assert 1e-3 < np.mean(theo) < 5e-2
    assert np.mean(meas) == pytest.approx(np.mean(theo), rel=0.2)


# ---- receiver: PAT and demodulation --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16QAM"])
def test_clean_link_delivers_every_modulation_with_a_passing_crc(mod):
    data = bytes(np.random.default_rng(5).integers(0, 256, 700, dtype=np.uint8))
    sig, beam, metrics, bud, rep = _loopback(data, mod)
    assert rep.sync_ok and rep.crc_ok and rep.payload == data and rep.ber == 0.0
    assert metrics.ber == 0.0 and rep.pat.locked and rep.pat.t_lock_s is not None and rep.pat.t_lock_s < 0.01


def test_fsm_closed_loop_locks_open_loop_cannot_sync():
    data = b"pointing, acquisition and tracking " * 8
    _, _, _, _, closed = _loopback(data, "QPSK", closed_loop=True)
    _, _, _, _, opened = _loopback(data, "QPSK", closed_loop=False)
    assert closed.crc_ok and closed.pat.rms_residual_urad < 8.0                       # the loop pulls a >300 urad offset down to a few urad
    assert not opened.sync_ok and not opened.crc_ok and opened.pat.rms_residual_urad > 100.0
    assert opened.mean_coupling < 1e-6                                                # a beam 300 urad off a 12 urad fibre lets almost nothing in


def test_pat_search_then_track_events_and_trace():
    _, _, _, _, rep = _loopback(b"z" * 100, "QPSK")
    kinds = " | ".join(t for _, t in rep.pat.events)
    assert "SEARCH" in kinds and "BEACON DETECTED" in kinds and "LOCK" in kinds and "RECEIVE" in kinds
    tr = rep.pat.trace
    assert np.any(tr["state"] == 0) and np.any(tr["state"] == 1)                       # a search phase, then a tracking phase
    assert np.linalg.norm(tr["err_urad"][tr["state"] == 1][-5:], axis=1).max() < 30.0
    assert rep.pat.t_detect_s < rep.pat.t_lock_s


def test_a_weak_beacon_is_never_detected():
    link = dch.LinkConfig(range_km=40000.0, altitude_km=40000.0, beacon_power_w=1e-6, beacon_divergence_urad=200.0)
    sig = dtx.transmit(b"q" * 64, "BPSK", 1e9, 8, 0.35, 16, LAM, 1.0)
    beam, _, _, _ = dch.propagate(sig, link, np.random.default_rng(1), 1)
    rep = drx.receive(beam, drx.PATConfig(), np.random.default_rng(2), True)
    assert rep.pat.t_detect_s is None and any("TIMEOUT" in t for _, t in rep.pat.events)


def test_pre_locked_receiver_skips_the_search():
    data = b"already tracking " * 6
    _, _, _, _, rep = _loopback(data, "QPSK", pre_locked=True)
    assert rep.crc_ok and rep.pat.t_lock_s == 0.0
    assert not any("SEARCH" in t for _, t in rep.pat.events) and any("PRE-LOCKED" in t for _, t in rep.pat.events)
    assert rep.pat.rms_residual_urad < 8.0


def test_a_corrupted_frame_fails_the_crc_but_still_returns_the_bytes_and_the_ber():
    data = bytes(np.random.default_rng(10).integers(0, 256, 3000, dtype=np.uint8))
    sig, beam, metrics, bud, rep = _loopback(data, "16QAM", link=dch.LinkConfig(range_km=1500.0), power=1.0e-3)      # Es/N0 about 16 dB: 16-QAM struggles
    assert rep.sync_ok and not rep.crc_ok and rep.payload is not None and rep.ber > 1e-3 and rep.n_bit_errors > 20
    assert "MISMATCH" in rep.status


# ---- the 63 KB test payload ----------------------------------------------------------------------------------------------
def test_test_payload_is_63kb_of_1008_aligned_64_byte_lines_each_with_its_own_crc():
    d = pl.generate_test_payload()
    assert len(d) == 64512 == 63 * 1024 and len(d) % 64 == 0
    lines = d.split(b"\n")[:-1]
    assert len(lines) == 1008 and all(len(l) == 63 for l in lines)
    head = lines[7][:55]
    assert lines[7].startswith(b"B0007 ") and int(lines[7][55:63], 16) == zlib.crc32(head)
    assert all(32 <= c < 127 or c == 10 for c in d)                                    # printable ASCII: it is a valid .txt file
    assert pl.generate_test_payload() == d and pl.generate_test_payload(seed=1) != d  # deterministic per seed


def test_block_verifier_pinpoints_damaged_blocks_and_missing_ones():
    d = pl.generate_test_payload()
    rep = pl.verify_blocks(d)
    assert (rep.n_blocks, rep.n_ok, rep.n_bad, rep.aligned, rep.is_test_payload) == (1008, 1008, 0, True, True)
    bad = bytearray(d)
    bad[64 * 3 + 20] ^= 0x01
    bad[64 * 900 + 60] ^= 0x80
    rep = pl.verify_blocks(bytes(bad))
    assert rep.bad_blocks == [3, 900]
    short = pl.verify_blocks(d[:64 * 1000], 1008)
    assert short.n_bad == 8 and short.bad_blocks == list(range(1000, 1008))             # blocks that never arrived count as bad
    assert not pl.verify_blocks(b"hello world").is_test_payload and pl.verify_blocks(None, 4).n_bad == 4


def test_byte_diff_marks_wrong_and_missing_bytes():
    sent = b"abcdef"
    assert list(pl.byte_diff(sent, b"abXdef")) == [False, False, True, False, False, False]
    assert list(pl.byte_diff(sent, b"abc")) == [False, False, False, True, True, True]
    assert pl.byte_diff(sent, None).all()
