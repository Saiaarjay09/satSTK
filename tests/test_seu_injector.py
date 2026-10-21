"""The radiation model: zones and multipliers, IEEE-754 bit flips (positions, field classification, effects), the Poisson upset generator."""
import math

import numpy as np
import pytest

import seu_injector as si


# ---- fields and bit positions ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,sign,exp_lo,exp_hi,man_hi", [("float32", 31, 23, 30, 22), ("bfloat16", 15, 7, 14, 6), ("float16", 15, 10, 14, 9), ("float64", 63, 52, 62, 51)])
def test_field_boundaries(dtype, sign, exp_lo, exp_hi, man_hi):
    assert si.field_of_bit(sign, dtype) == "sign"
    assert si.field_of_bit(exp_hi, dtype) == si.field_of_bit(exp_lo, dtype) == "exponent"
    assert si.field_of_bit(exp_lo - 1, dtype) == si.field_of_bit(man_hi, dtype) == si.field_of_bit(0, dtype) == "mantissa"
    with pytest.raises(ValueError):
        si.field_of_bit(si.LAYOUTS[dtype][0], dtype)


def test_every_float32_bit_flips_exactly_that_bit_and_is_classified():
    for bit in range(32):
        a = np.array([0.1, 0.2], dtype=np.float32)
        before = a.view(np.uint32).copy()
        f = si.inject_ieee754_bitflip(a, index=1, bit=bit)
        after = a.view(np.uint32)
        assert before[0] == after[0]                                         # the neighbour is untouched
        assert int(before[1]) ^ int(after[1]) == 1 << bit                    # exactly one bit changed: the XOR of old and new is that bit
        assert f.bit == bit and f.index == 1 and f.field == si.field_of_bit(bit)
        assert f.old == pytest.approx(0.2, rel=1e-6) and (f.new == a[1] or (math.isnan(f.new) and math.isnan(a[1])))


def test_flipping_twice_restores_the_value():
    a = np.array([3.14159], dtype=np.float32)
    si.inject_ieee754_bitflip(a, 0, 17)
    si.inject_ieee754_bitflip(a, 0, 17)
    assert a[0] == np.float32(3.14159)


def test_mantissa_sign_and_exponent_effects():
    w = np.float32(0.1)
    m = si.flip_bit_value(w, 3)
    assert abs(m - w) / w < 1e-5                                               # low mantissa bit: parts-per-million noise
    assert si.flip_bit_value(w, 22) == pytest.approx(0.06875, rel=1e-3)        # top mantissa bit: still the same order of magnitude
    assert si.flip_bit_value(w, 31) == -w                                       # sign: the weight is inverted
    assert si.flip_bit_value(w, 30) > 1e37                                      # exponent MSB: explosion to ~1e38
    assert si.flip_bit_value(w, 23) == pytest.approx(w / 2, rel=1e-6)          # lowest exponent bit: a factor of two
    assert math.isinf(si.flip_bit_value(1.0, 30))                               # exponent all ones with a zero mantissa: Inf
    assert math.isnan(si.flip_bit_value(1.5, 30))                               # ... with a non-zero mantissa: NaN


def test_report_effect_text_and_blowup_flag():
    a = np.array([0.1], dtype=np.float32)
    f = si.inject_ieee754_bitflip(a, 0, 30)
    assert f.field == "exponent" and f.blowup and "explosion" in f.effect and f.new > 1e37
    a = np.array([0.1], dtype=np.float32)
    f = si.inject_ieee754_bitflip(a, 0, 5)
    assert f.field == "mantissa" and not f.blowup and f.effect == "minor noise"
    a = np.array([0.1], dtype=np.float32)
    assert si.inject_ieee754_bitflip(a, 0, 31).effect == "inverted weight"
    a = np.array([0.1], dtype=np.float32)
    f = si.inject_ieee754_bitflip(a, 0, 23)
    assert f.field == "exponent" and not f.blowup and "power of two" in f.effect


def test_random_choices_cover_all_fields_with_the_right_proportions():
    rng = np.random.default_rng(3)
    counts = {"mantissa": 0, "exponent": 0, "sign": 0}
    for _ in range(4000):
        a = np.array([0.1], dtype=np.float32)
        counts[si.inject_ieee754_bitflip(a, rng=rng).field] += 1
    assert counts["mantissa"] / 4000 == pytest.approx(23 / 32, abs=0.04)
    assert counts["exponent"] / 4000 == pytest.approx(8 / 32, abs=0.04)
    assert counts["sign"] / 4000 == pytest.approx(1 / 32, abs=0.02)


def test_float16_and_float64_and_bad_inputs():
    h = np.array([1.0], dtype=np.float16)
    f = si.inject_ieee754_bitflip(h, 0, 14)
    assert f.field == "exponent" and f.dtype == "float16" and f.blowup                  # 1.0 -> exponent 11111 -> Inf
    d = np.array([0.1], dtype=np.float64)
    assert si.inject_ieee754_bitflip(d, 0, 62).blowup
    with pytest.raises(TypeError):
        si.inject_ieee754_bitflip(np.array([1], dtype=np.int32), 0, 0)
    with pytest.raises(ValueError):
        si.inject_ieee754_bitflip(np.zeros((4, 4), dtype=np.float32)[:, ::2], 0, 0)
    with pytest.raises(IndexError):
        si.inject_ieee754_bitflip(np.zeros(3, dtype=np.float32), 5, 0)


def test_torch_tensors_match_numpy():
    torch = pytest.importorskip("torch")
    for bit in (0, 10, 22, 23, 29, 30, 31):
        t = torch.tensor([0.1, 0.2], dtype=torch.float32)
        f = si.inject_ieee754_bitflip(t, 1, bit)
        n = np.array([0.1, 0.2], dtype=np.float32)
        g = si.inject_ieee754_bitflip(n, 1, bit)
        assert f.field == g.field and f.blowup == g.blowup
        assert float(t[1]) == pytest.approx(float(n[1])) and float(t[0]) == pytest.approx(0.1, rel=1e-6)
    b = torch.tensor([0.1, 0.2], dtype=torch.bfloat16)                                    # bfloat16: 16 bits, 8 exponent bits
    f = si.inject_ieee754_bitflip(b, 0, 14)
    assert f.dtype == "bfloat16" and f.field == "exponent" and f.blowup and float(b[0]) > 1e37
    b = torch.tensor([0.1], dtype=torch.bfloat16)
    assert si.inject_ieee754_bitflip(b, 0, 15).field == "sign" and float(b[0]) < 0
    with pytest.raises(ValueError):
        si.inject_ieee754_bitflip(torch.zeros(4, 4)[:, ::2], 0, 0)


# ---- zones and multipliers ------------------------------------------------------------------------------------------------------------
def test_zones():
    assert si.classify_zone(0.0, -150.0) == si.NOMINAL
    assert si.classify_zone(-25.0, -45.0) == si.SAA                                       # off the Brazilian coast
    assert si.classify_zone(-50.0, -90.0) == si.SAA and si.classify_zone(0.0, 40.0) == si.SAA       # the box corners are inside
    assert si.classify_zone(-50.1, -45.0) == si.NOMINAL and si.classify_zone(0.1, -45.0) == si.NOMINAL
    assert si.classify_zone(-25.0, -90.1) == si.NOMINAL and si.classify_zone(-25.0, 40.1) == si.NOMINAL
    assert si.classify_zone(-25.0, 315.0) == si.SAA                                       # 315 E is 45 W
    assert si.classify_zone(75.0, 20.0) == si.POLAR_N and si.classify_zone(-75.0, 100.0) == si.POLAR_S
    assert si.classify_zone(60.0, 0.0) == si.NOMINAL and si.classify_zone(60.1, 0.0) == si.POLAR_N


def test_multipliers_and_rates():
    assert si.flux_multiplier(0, -150) == 1.0
    assert si.flux_multiplier(75, 20) == 15.0 and si.flux_multiplier(-70, 0) == 15.0
    assert si.flux_multiplier(-25, -45) == 100.0
    assert si.flux_multiplier(-25, -45, flare=1000.0) == 100_000.0 and si.flux_multiplier(0, -150, flare=si.FLARE_MULT) == 1000.0
    assert si.seu_rate_per_blade_s(0, -150) == pytest.approx(1e-5)
    assert si.seu_rate_per_blade_s(75, 20) == pytest.approx(1.5e-4)
    assert si.seu_rate_per_blade_s(-25, -45) == pytest.approx(1e-3)
    assert si.altitude_factor(600.0) == 1.0 and si.altitude_factor(900.0) > 1.0 > si.altitude_factor(400.0)
    assert si.altitude_factor(5000.0) == 4.0 and si.altitude_factor(0.0) == 0.25
    assert si.flux_multiplier(-25, -45, 900.0) == pytest.approx(100.0 * math.e)


def test_poisson_draw_matches_the_expected_rate_and_the_latchup_fraction():
    inj = si.SEUInjector(seed=5, n_blades=32)
    n_seu = n_sel = 0
    draws = 600
    for _ in range(draws):
        ups = inj.draw(1000.0, 0.0, -150.0)                                               # nominal: 1e-5 * 1000 s = 0.01 per blade
        n_seu += sum(u.kind == "SEU" for u in ups)
        n_sel += sum(u.kind == "SEL" for u in ups)
    expected = draws * 32 * 0.01
    assert n_seu == pytest.approx(expected, rel=0.12)
    inj2 = si.SEUInjector(seed=6, n_blades=32)
    saa = sum(len(inj2.draw(1000.0, -25.0, -45.0)) for _ in range(60))                    # SAA: 100x more
    assert saa / 60 == pytest.approx(32 * 0.01 * 100, rel=0.12)
    inj3 = si.SEUInjector(seed=7, n_blades=32, sel_fraction=0.5)
    ups = [u for _ in range(100) for u in inj3.draw(1000.0, -25.0, -45.0)]
    frac = sum(u.kind == "SEL" for u in ups) / len(ups)
    assert 0.3 < frac < 0.4                                                               # 0.5 per upset -> 0.5/1.5 of all events


def test_upsets_are_replayable_and_target_split():
    inj = si.SEUInjector(seed=1)
    ups = [u for _ in range(30) for u in inj.draw(1000.0, -25.0, -45.0) if u.kind == "SEU"]
    assert 0.6 < np.mean([u.target == "weights" for u in ups]) < 0.8                       # P_WEIGHT_HIT = 0.7
    a, b = np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32)
    u = ups[0]
    f1 = si.inject_ieee754_bitflip(a, index=int(u.u_index * 100), bit=int(u.u_bit * 32))
    f2 = si.inject_ieee754_bitflip(b, index=int(u.u_index * 100), bit=int(u.u_bit * 32))
    assert (f1.index, f1.bit) == (f2.index, f2.bit) and np.array_equal(a, b, equal_nan=True)


def test_seeded_injector_repeats():
    a = si.SEUInjector(seed=11).draw(500.0, -25.0, -45.0)
    b = si.SEUInjector(seed=11).draw(500.0, -25.0, -45.0)
    assert [(u.blade, u.kind, u.when) for u in a] == [(u.blade, u.kind, u.when) for u in b]


def test_saa_entry_prediction():
    def pos(t):
        return -20.0, -130.0 + 0.05 * t                                                    # reaches lon -90 at t = 800 s
    assert si.predict_saa_entry_s(pos, 0.0, 300.0) is None
    eta = si.predict_saa_entry_s(pos, 600.0, 300.0)
    assert eta is not None and 150.0 <= eta <= 240.0
    assert si.predict_saa_entry_s(pos, 900.0, 300.0) == 0.0                                 # already inside
