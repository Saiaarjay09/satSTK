"""The electrical power subsystem: solar array, Shepherd/Thevenin battery, 48 V busbar I^2R (checked against an independent hand calculation), the
four-mode load-shedding state machine, the closed loop with the cooling loop and the radiation scheduler, and the stress tests."""
import math

import numpy as np
import pytest

import eps_power_engine as ep
import orbital_engine as oe
import simulation as sim
import thermal_loop as tl
from helpers import direct_cfg

CFG = ep.EPSConfig()
PEAK_W = 182.4 * 0.32 * 1361.0                                      # 79,441 W at normal incidence, 28 C cells, full sun


@pytest.fixture(scope="module")
def world():
    return sim.run_simulation(direct_cfg(duration_h=24, step_s=10))


@pytest.fixture(scope="module")
def default_run(world):
    return ep.simulate_eps(world)


def make_ctrl(cfg=None, ill=1.0, soc=1.0, n=1, cos=1.0, flux=1361.0, step=10.0):
    c = ep.EPSController(cfg or ep.EPSConfig(), np.full(n, ill, float) if np.isscalar(ill) else np.asarray(ill, float), np.full(n, cos), np.full(n, flux), step)
    c.soc = soc
    c.t_array_k = 28.0 + 273.15
    return c


def tick(c, i=0, temp=40.0, mode=tl.NOMINAL, frac=1.0):
    heat, rpm = c(i, temp, mode, frac, 0.0)
    return heat, rpm, c.h


# ---- solar array -----------------------------------------------------------------------------------------------------------------------
def test_solar_peak_matches_the_design_numbers():
    assert ep.solar_power_w(1361.0, 1.0, 1.0, 28.0, CFG) == pytest.approx(PEAK_W) and 79_000 < PEAK_W < 80_000
    assert CFG.array_area_m2 == 182.4 and CFG.panels_total == 12 and CFG.cell_efficiency == 0.32


def test_solar_scales_with_illumination_incidence_failed_panels_and_cell_temperature():
    f = lambda **k: ep.solar_power_w(**{**dict(flux_w_m2=1361.0, cos_incidence=1.0, illumination=1.0, t_cell_c=28.0, cfg=CFG), **k})  # noqa: E731
    assert f(illumination=0.0) == 0.0 and f(illumination=0.5) == pytest.approx(PEAK_W / 2)           # umbra / half-way through the penumbra
    assert f(cos_incidence=math.cos(math.radians(60))) == pytest.approx(PEAK_W / 2)
    assert f(cos_incidence=-0.3) == 0.0                                                                # the Sun behind the panels
    assert f(t_cell_c=48.0) == pytest.approx(PEAK_W * (1 - 0.0025 * 20))                              # -0.25 % per C above 28 C
    assert f(t_cell_c=-60.0) == pytest.approx(PEAK_W)                                                  # cold cells: no bonus in this model
    assert f(cfg=ep.EPSConfig(panels_failed=1)) == pytest.approx(PEAK_W * 11 / 12)
    assert f(cfg=ep.EPSConfig(panels_failed=12)) == 0.0


def test_solar_constant_follows_the_earth_sun_distance():
    near, far = np.array([[0.983 * oe.AU_KM, 0, 0]]), np.array([[1.017 * oe.AU_KM, 0, 0]])
    assert ep.solar_constant_at(near)[0] == pytest.approx(1361.0 / 0.983 ** 2) and ep.solar_constant_at(far)[0] < 1361.0 < ep.solar_constant_at(near)[0]


def test_array_temperature_is_hot_in_sun_and_cold_in_eclipse():
    sun = ep.array_equilibrium_k(1361.0, 1.0, 1.0, 0.32, CFG) - 273.15
    shadow = ep.array_equilibrium_k(1361.0, 1.0, 0.0, 0.32, CFG) - 273.15
    assert 20.0 < sun < 60.0 and -200.0 < shadow < -100.0


# ---- battery ---------------------------------------------------------------------------------------------------------------------------
def test_capacity_and_pack_numbers():
    assert CFG.capacity_kwh == 320.0 and CFG.n_modules * CFG.module_kwh == 320.0 and CFG.nominal_v == 100.0
    assert CFG.capacity_ah == pytest.approx(3200.0) and ep.EPSConfig(soh=0.8).capacity_kwh == pytest.approx(256.0)


def test_shepherd_curve_is_monotonic_and_near_100_volts():
    v = [ep.shepherd_voc(s, CFG) for s in np.linspace(0.05, 1.0, 40)]
    assert all(b > a for a, b in zip(v, v[1:]))
    assert ep.shepherd_voc(0.5, CFG) == pytest.approx(100.0, abs=0.5) and 104.0 < ep.shepherd_voc(1.0, CFG) < 106.0 and ep.shepherd_voc(0.05, CFG) < 90.0


def test_internal_resistance_rises_when_empty_cold_and_old():
    base = ep.internal_resistance_ohm(0.5, 25.0, CFG)
    assert base == pytest.approx(0.004 * (1 + 1.5 * 0.125), rel=1e-9)
    assert ep.internal_resistance_ohm(0.1, 25.0, CFG) > base and ep.internal_resistance_ohm(0.5, -10.0, CFG) > base
    assert ep.internal_resistance_ohm(0.5, 25.0, ep.EPSConfig(soh=0.6)) == pytest.approx(base * 1.8)


def test_battery_current_solves_the_power_equation():
    e, r = 100.0, 0.004
    for p in (-30_000.0, 0.0, 10_000.0, 50_000.0, 150_000.0):
        i, ok = ep.battery_current(p, e, r)
        assert ok and e * i - r * i * i == pytest.approx(p, abs=1e-6)
    assert ep.battery_current(-30_000.0, e, r)[0] < 0                                      # charging: negative current
    i, ok = ep.battery_current(e * e / (4 * r) * 1.01, e, r)                               # more than the maximum power transfer
    assert not ok and i == pytest.approx(e / (2 * r))


def test_soc_follows_coulomb_counting_and_matches_the_energy_formula():
    """10 kW out for 10 minutes, then 10 kW in: dSoC = P dt / E within the voltage-sag and efficiency terms (the brief's formula)."""
    n = 60
    c = ep.EPSController(ep.EPSConfig(blade_tdp_w=0.0, pump_w=0.0, oisl_w=0.0, housekeeping_w=10_000.0 * 0.97, blade_idle_frac=0.0), np.zeros(n), np.ones(n), np.full(n, 1361.0), 10.0)
    for i in range(n):
        tick(c, i)
    drop = 1.0 - c.soc
    expected = 10_000.0 * 600.0 / 3.6e6 / 320.0                                            # P dt / E
    assert drop == pytest.approx(expected, rel=0.08) and c.soc < 1.0
    # charging at the BCDU's 40 kW limit, less the converter's 97 % and the 99 % Coulomb efficiency
    cc = ep.EPSController(ep.EPSConfig(blade_tdp_w=0.0, pump_w=0.0, oisl_w=0.0, housekeeping_w=0.0, blade_idle_frac=0.0), np.ones(n), np.ones(n), np.full(n, 1361.0), 10.0)
    cc.soc = 0.5
    cc.t_array_k = 301.0
    for i in range(n):
        tick(cc, i)
    assert cc.soc - 0.5 == pytest.approx(40_000.0 * 0.97 * 0.99 * 600.0 / 3.6e6 / 320.0, rel=0.05)


def test_power_balance_closes_on_every_step(default_run):
    h = default_run.history
    resid = 0.97 * h.solar_w - (h.load_w - h.bat_power_w + h.shunt_w)
    assert np.abs(resid).max() < 1e-6 * h.solar_w.max() + 1e-3


# ---- busbars -----------------------------------------------------------------------------------------------------------------------------
def _brute_force(p_bank, bank, t_c, cfg):
    """An independent scalar implementation of the same physics, for cross-checking the vectorised solver."""
    r = 1.68e-8 * (1 + 0.00393 * (t_c - 20)) / (0.060 * 0.015)
    z = [cfg.z_first_blade_m + k * (cfg.z_last_blade_m - cfg.z_first_blade_m) / 31 for k in range(32)]
    z = z[:16] if bank == 0 else z[16:]
    feeder = 8.0 - (3.2 + 3.7)
    lens = [z[0] - (-3.7) + feeder] + [z[k] - z[k - 1] for k in range(1, 16)]
    v = [48.0] * 16
    for _ in range(30):
        i = [p_bank / 16 / vk for vk in v]
        v = [48.0 - 2 * r * sum(lens[j] * sum(i[j:]) for j in range(k + 1)) for k in range(16)]
    i = [p_bank / 16 / vk for vk in v]
    loss = 2 * r * sum(lens[j] * sum(i[j:]) ** 2 for j in range(16))
    return v, i, loss


def test_busbar_matches_an_independent_hand_calculation():
    prof = ep.busbar_profile(19_200.0, 19_200.0, 20.0, CFG)
    va, ia, la = _brute_force(19_200.0, 0, 20.0, CFG)
    vb, ib, lb = _brute_force(19_200.0, 1, 20.0, CFG)
    assert np.allclose(prof["v_blade"][:16], va, atol=1e-9) and np.allclose(prof["v_blade"][16:], vb, atol=1e-9)
    assert prof["loss_w"] == pytest.approx(la + lb, rel=1e-6)
    assert prof["i_bank_a"] == pytest.approx(sum(ia), rel=1e-6)


def test_busbar_geometry_and_resistivity():
    assert CFG.bar_area_m2 == pytest.approx(900e-6) and CFG.bus_length_m == 8.0
    assert ep.copper_resistivity(20.0) == pytest.approx(1.68e-8) and ep.copper_resistivity(70.0) == pytest.approx(1.68e-8 * (1 + 0.00393 * 50))
    z = ep.blade_z_m(CFG)
    assert z[0] == pytest.approx(-3.2) and z[-1] == pytest.approx(3.2) and np.allclose(np.diff(z), 6.4 / 31)


def test_busbar_energy_is_conserved_and_the_drop_grows_with_height_and_temperature():
    cold, hot = ep.busbar_profile(19_200.0, 19_200.0, 20.0, CFG), ep.busbar_profile(19_200.0, 19_200.0, 120.0, CFG)
    for prof, t in ((cold, 20.0), (hot, 120.0)):
        delivered = float(np.sum(prof["v_blade"] * prof["i_blade"]))
        assert 48.0 * prof["i_total"] == pytest.approx(delivered + prof["loss_w"], rel=1e-9)         # power at the base = power at the blades + I^2 R
    assert np.all(np.diff(cold["v_blade"][:16]) < 0) and np.all(np.diff(cold["v_blade"][16:]) < 0)    # higher blade, lower voltage
    assert cold["v_blade"][31] < cold["v_blade"][15]                                                    # Bank B's bars are longer
    assert hot["drop_max_v"] / cold["drop_max_v"] == pytest.approx(1 + 0.00393 * 100, rel=0.01)         # copper gets worse when hot


def test_busbar_voltages_at_the_design_load_and_the_vrm_band():
    prof = ep.busbar_profile(19_200.0, 19_200.0, 30.0, CFG)
    assert prof["i_bank_a"] == pytest.approx(19_200 / 48.0, rel=0.01) and 790 < prof["i_total"] < 810                 # 400 A per bank at 48 V
    assert 47.85 < prof["v_min"] < 47.95 and prof["v_max"] < 48.0 and 0.05 < prof["drop_max_v"] < 0.2                  # about 0.1 V from 8 m of 900 mm^2 copper
    assert prof["in_band"] and 40.0 < prof["loss_w"] < 120.0
    assert ep.busbar_profile(0.0, 0.0, 30.0, CFG)["v_min"] == pytest.approx(48.0) and ep.busbar_profile(0.0, 0.0, 30.0, CFG)["loss_w"] == 0.0
    overload = ep.busbar_profile(1_500_000.0, 1_500_000.0, 30.0, CFG)                                                     # absurd load: the band check fails
    assert not overload["in_band"] and overload["v_min"] < 45.5


# ---- illumination helpers ----------------------------------------------------------------------------------------------------------------
def test_eclipse_runs_and_forced_extension():
    ill = np.ones(100)
    ill[10:20] = 0.0
    ill[50:53] = 0.0
    ill[53:55] = [0.2, 0.7]
    runs = ep.eclipse_runs(ill)
    assert runs == [(10, 20), (50, 54)]
    out = ep.extend_eclipses(ill, 60.0, 15.0)                                                    # 15 minutes at 60 s steps = 15 samples
    assert [b - a for a, b in ep.eclipse_runs(out)] == [15, 15] and np.all(out[10:25] == 0.0) and ill[20] == 1.0     # the input is not modified
    assert np.array_equal(ep.extend_eclipses(ill, 60.0, 4.0), ill)                               # already longer: untouched


# ---- the state machine --------------------------------------------------------------------------------------------------------------------
def test_sunlit_peak_runs_everything_and_tops_up_the_battery():
    c = make_ctrl(ill=1.0, soc=0.95)
    heat, rpm, h = tick(c)
    assert c.mode == ep.SUNLIT_PEAK and h["cfrac"][0] == 1.0 and not h["dvfs"][0] and h["bclosed"][0] and rpm == 1.0
    assert h["pa"][0] == pytest.approx(16 * 1200.0) and h["pb"][0] == pytest.approx(16 * 1200.0)
    assert h["batp"][0] < 0 and h["ibat"][0] < 0 and c.soc > 0.95                           # surplus solar charges the pack
    assert heat == pytest.approx(h["pa"][0] + h["pb"][0] + h["pfix"][0] + h["bloss"][0] + h["bheat"][0] + (-h["batp"][0]) * 0.03, rel=1e-9)


def test_sunlit_recharge_reserves_35_percent_of_the_array_for_the_charger():
    c = make_ctrl(ep.EPSConfig(panels_failed=9), ill=1.0, soc=0.80)                          # 3 panels left: 19.9 kW
    _, _, h = tick(c)
    p_bus = 0.97 * h["solar"][0]
    assert c.mode == ep.SUNLIT_RECHARGE and 0.0 < h["cfrac"][0] < 0.5                        # compute scaled to the headroom
    assert -h["batp"][0] >= 0.35 * p_bus - 1.0                                               # at least 35 % went to the battery
    assert h["load"][0] <= 0.65 * p_bus + 1.0


def test_recharge_mode_with_a_healthy_array_charges_hard_and_keeps_full_compute():
    c = make_ctrl(ill=1.0, soc=0.7)
    _, _, h = tick(c)
    assert c.mode == ep.SUNLIT_RECHARGE and h["cfrac"][0] == 1.0 and -h["batp"][0] > 0.35 * 0.97 * h["solar"][0]


def test_modes_follow_soc_with_hysteresis_in_the_sun():
    c = make_ctrl(n=4, ill=1.0, soc=0.895)
    tick(c, 0)
    assert c.mode == ep.SUNLIT_RECHARGE
    c.soc = 0.905
    tick(c, 1)
    assert c.mode == ep.SUNLIT_RECHARGE                                                       # below the 91 % exit
    c.soc = 0.915
    tick(c, 2)
    assert c.mode == ep.SUNLIT_PEAK
    c2 = make_ctrl(soc=0.905)
    tick(c2)
    assert c2.mode == ep.SUNLIT_PEAK                                                          # entering from above: the 90 % threshold itself


def test_eclipse_cruise_engages_dvfs_on_both_banks_and_draws_on_the_battery():
    c = make_ctrl(ill=0.0, soc=0.9)
    _, rpm, h = tick(c)
    assert c.mode == ep.ECLIPSE_CRUISE and h["dvfs"][0] and h["solar"][0] == 0.0
    assert h["pa"][0] == pytest.approx(16 * 1200.0 * 0.60) and h["pb"][0] == pytest.approx(16 * 1200.0 * 0.60)     # -40 % power
    assert h["bclosed"][0] and not h["oisl_low"][0] and rpm == 1.0
    assert h["batp"][0] == pytest.approx(h["load"][0]) and h["ibat"][0] > 250 and c.soc < 0.9      # all of it from the battery
    assert ep.EPSConfig().dvfs_speed == 0.80


def test_eclipse_survival_flushes_opens_bank_b_lowers_lasers_and_pumps():
    c = make_ctrl(n=3, ill=0.0, soc=0.34)
    heat, rpm, h = tick(c, 0)
    assert c.mode == ep.ECLIPSE_SURVIVAL and c.flush_steps == [0]                              # checkpoint flush at the moment of entry
    assert not h["bclosed"][0] and h["pb"][0] == 0.0 and h["oisl_low"][0] and rpm == 0.40
    assert h["pfix"][0] == pytest.approx(2500 * 0.4 ** 3 + 700 + 1500)                        # pump power falls with the cube of the speed; lasers low-power
    tick(c, 1)
    assert c.flush_steps == [0]                                                               # flushed once, not every step
    c.soc = 0.37
    tick(c, 2)
    assert c.mode == ep.ECLIPSE_SURVIVAL                                                      # hysteresis: stays until 40 %
    c.soc = 0.41
    c.ill[:] = 0.0
    c(2, 40.0, tl.NOMINAL, 1.0, 0.0)
    assert c.mode == ep.ECLIPSE_CRUISE


def test_bank_b_breaker_recloses_only_after_the_pack_recovers():
    c = make_ctrl(n=4, ill=[0.0, 1.0, 1.0, 1.0], soc=0.30)
    tick(c, 0)
    assert not c.h["bclosed"][0]
    tick(c, 1)                                                                                 # sunrise, but only 30 %: the breaker stays open
    assert c.mode == ep.SUNLIT_RECHARGE and not c.h["bclosed"][1] and c.h["pb"][1] == 0.0
    c.soc = 0.45
    tick(c, 2)
    assert c.h["bclosed"][2] and c.h["pb"][2] > 0


def test_critical_soc_sheds_bank_a_compute_and_the_lasers_too():
    c = make_ctrl(ill=0.0, soc=0.22)
    _, _, h = tick(c)
    assert c.mode == ep.ECLIPSE_SURVIVAL and h["crit"][0] and h["cfrac"][0] == 0.0
    assert h["pa"][0] == pytest.approx(16 * 1200.0 * 0.08) and h["pb"][0] == 0.0 and h["pfix"][0] == pytest.approx(2500 * 0.064 + 1500)


def test_thermal_soft_throttle_hibernates_bank_b_and_the_85c_throttle_scales_compute():
    c = make_ctrl(n=3, ill=1.0, soc=1.0)
    _, _, h = tick(c, 0, temp=76.0)
    assert h["bhib"][0] and h["pb"][0] == pytest.approx(16 * 1200.0 * 0.08) and h["pa"][0] == pytest.approx(16 * 1200.0)
    tick(c, 1, temp=72.0)
    assert c.h["bhib"][1]                                                                      # hysteresis: released only at 70 C
    tick(c, 2, temp=60.0)
    assert not c.h["bhib"][2]
    c2 = make_ctrl(ill=1.0)
    _, _, h2 = tick(c2, temp=90.0, mode=tl.THROTTLE, frac=0.6)
    assert h2["cfrac"][0] == pytest.approx(0.6)
    c3 = make_ctrl(ill=1.0)
    _, _, h3 = tick(c3, temp=99.0, mode=tl.SHUTDOWN, frac=0.0)
    assert h3["cfrac"][0] == 0.0 and h3["pa"][0] == pytest.approx(16 * 96.0)


def test_scheduler_and_eps_agree_on_the_soft_throttle_thresholds():
    import radiation_pytorch_scheduler as rps
    assert ep.EPSConfig().soft_throttle_c == rps.SOFT_THROTTLE_C and ep.EPSConfig().soft_release_c == rps.SOFT_RELEASE_C


def test_a_dead_battery_browns_out_instead_of_crashing():
    c = make_ctrl(ill=0.0, soc=0.0)
    _, _, h = tick(c)
    assert h["brown"][0] and c.soc == 0.0


def test_config_validation():
    for bad in (dict(panels_failed=13), dict(soh=0.01), dict(soc_initial=1.5)):
        with pytest.raises(ValueError):
            ep.EPSConfig(**bad).validate()


# ---- the full closed loop over a day ------------------------------------------------------------------------------------------------------
def test_default_day_is_the_classic_sawtooth(default_run):
    h, s = default_run.history, default_run.history.summary()
    assert len(s["eclipses"]) in (14, 15) and all(34.0 <= e["duration_min"] <= 36.0 for e in s["eclipses"])             # ~35-minute eclipses every ~96 minutes
    starts = [e["start_min"] for e in s["eclipses"]]
    assert np.mean(np.diff(starts)) == pytest.approx(96.7, abs=1.0)
    for e in s["eclipses"][1:]:
        assert e["soc_entry_pct"] > 99.0 and 90.0 < e["soc_min_pct"] < 97.0 and 12.0 < e["energy_kwh"] < 20.0       # tops up to 100 % before every eclipse
    assert h.solar_w.max() == pytest.approx(PEAK_W, rel=0.04) and h.solar_w[h.illumination == 0].max() == 0.0
    assert s["floor_violation_steps"] == 0 and s["brownout_steps"] == 0 and s["flushes"] == 0 and s["bank_b_open_min"] == 0
    assert s["vrm_ok"] and 47.85 < s["v_min"] < 47.95 and 780 < s["max_bus_current_a"] < 830
    assert s["mode_minutes"][ep.ECLIPSE_CRUISE] == pytest.approx(sum(e["duration_min"] for e in s["eclipses"]), abs=15.0)
    assert 0.05 < s["max_c_rate"] < 0.2 and s["peak_load_kw"] < s["peak_solar_kw"]
    assert np.all((h.soc >= 0) & (h.soc <= 1.0 + 1e-12))


def test_eclipse_uses_dvfs_and_runs_at_80_percent_speed(default_run):
    h = default_run.history
    dark = h.illumination < 0.5
    assert np.all(h.dvfs[dark]) and not np.any(h.dvfs[~dark])
    tp = h.throughput()
    assert tp[dark].mean() == pytest.approx(0.80, abs=0.02) and tp[~dark].mean() > 0.97
    assert h.load_w[dark].mean() < 0.7 * h.load_w[~dark].mean()


def test_penumbra_ramps_the_array_and_the_cell_temperature_follows_with_a_lag(default_run, world):
    h = default_run.history
    mid = (h.illumination > 0.05) & (h.illumination < 0.95)
    assert mid.any() and np.all(h.solar_w[mid] < PEAK_W) and np.all(h.solar_w[mid] > 0)
    i = int(np.flatnonzero(np.diff((h.illumination < 0.5).astype(int)) == -1)[2]) + 1              # a sunrise
    assert h.t_array_c[i] < -50.0 and h.t_array_c[i + 360] > 20.0                                    # the array is frozen at sunrise and warm an hour later
    assert h.solar_w[i + 30] > h.solar_w[i + 300] > 0.95 * h.solar_w[i + 30]                          # cold cells give a little more power than warm ones


def test_heat_goes_to_the_cooling_loop_and_the_loop_stays_sane(default_run):
    h, th = default_run.history, default_run.thermal
    assert np.all(h.heat_w > h.load_w - 1e-6)                                                       # waste heat = load + battery Joule heat + converter losses
    assert np.all(h.heat_w[1:] >= h.load_w[1:] + h.bat_heat_w[1:] - 1e-6)
    assert set(th.throttle_mode) == {tl.NOMINAL} and 25.0 < th.core_temp_c.min() and th.core_temp_c.max() < 75.0
    assert h.bat_heat_w.max() > 300.0 and h.bus_loss_w.max() > 40.0
    dark = h.illumination < 0.5
    assert th.core_temp_c[dark].min() < th.core_temp_c[~dark].max()


def test_coupling_changes_the_thermal_history_versus_the_standalone_model(default_run, world):
    legacy = tl.simulate_loop(world.tracks[world.target].radiator_solar_load_w, 10.0, world.config.thermal)
    assert not np.allclose(legacy.core_temp_c, default_run.thermal.core_temp_c, atol=1.0)


# ---- stress tests ------------------------------------------------------------------------------------------------------------------------
def test_one_panel_failure_costs_one_twelfth_of_the_array(world, default_run):
    run = ep.simulate_eps(world, ep.EPSConfig(panels_failed=1))
    assert run.history.solar_w.max() == pytest.approx(default_run.history.solar_w.max() * 11 / 12, rel=0.01)
    assert run.history.summary()["floor_violation_steps"] == 0 and run.history.soc.min() > 0.9


def test_extended_45_minute_eclipse(world, default_run):
    run = ep.simulate_eps(world, ep.EPSConfig(force_eclipse_min=45.0))
    s0, s1 = default_run.history.summary(), run.history.summary()
    assert all(e["duration_min"] >= 44.9 for e in s1["eclipses"])
    assert s1["eclipses"][2]["dod_pct"] > 1.2 * s0["eclipses"][2]["dod_pct"] and s1["floor_violation_steps"] == 0
    assert ep.eclipse_runs(world.tracks[0].sun_visibility)[0][1] - ep.eclipse_runs(world.tracks[0].sun_visibility)[0][0] < 220      # the real shadow is untouched


def test_degraded_battery_enters_survival_and_defends_the_floor(world):
    run = ep.simulate_eps(world, ep.EPSConfig(soh=0.10, force_eclipse_min=45.0))
    h, s = run.history, run.history.summary()
    assert s["mode_minutes"][ep.ECLIPSE_SURVIVAL] > 10 and s["flushes"] >= 5 and s["bank_b_open_min"] > 30
    assert s["floor_violation_steps"] == 0 and s["soc_min_pct"] > 25.0                              # survival mode held the 20 % floor
    surv = np.array([m == ep.ECLIPSE_SURVIVAL for m in h.mode])
    assert not h.bank_b_closed[surv].any() and h.oisl_low_power[surv].all() and np.allclose(h.pump_rpm[surv], 0.4)
    assert np.all(h.p_bank_b_w[surv] == 0.0)
    assert h.throughput().mean() < ep.simulate_eps(world).history.throughput().mean()


def test_pumps_at_40_percent_cut_the_coolant_flow(world):
    run = ep.simulate_eps(world, ep.EPSConfig(soh=0.10, force_eclipse_min=45.0))
    surv = np.array([m == ep.ECLIPSE_SURVIVAL for m in run.history.mode])
    base = ep.simulate_eps(world)
    assert run.thermal.coolant_flow_lpm[surv].mean() < 0.6 * base.thermal.coolant_flow_lpm[base.history.illumination < 0.5].mean()


def test_the_floor_cannot_be_defended_against_an_absurd_stress_and_that_is_reported(world):
    run = ep.simulate_eps(world, ep.EPSConfig(soh=0.05, force_eclipse_min=60.0))
    s = run.history.summary()
    assert s["floor_violation_steps"] > 0 and s["soc_min_pct"] < 20.0                                # 16 kWh and a 60-minute shadow: physics wins


def test_degradation_raises_battery_heat_and_resistance(world, default_run):
    run = ep.simulate_eps(world, ep.EPSConfig(soh=0.5))
    assert run.history.r_int_mohm.mean() > default_run.history.r_int_mohm.mean() * 1.5 and run.history.bat_heat_w.max() > default_run.history.bat_heat_w.max()


# ---- the radiation scheduler follows the power system ---------------------------------------------------------------------------------------
def test_scheduler_loses_bank_b_in_survival_flushes_a_checkpoint_and_runs_at_dvfs_speed(world):
    import radiation_pytorch_scheduler as rps
    run = ep.simulate_eps(world, ep.EPSConfig(soh=0.10, force_eclipse_min=45.0))
    h = run.history
    k = h.flush_steps[0]
    orbit = rps.TrackOrbit.from_result(world)
    o = rps.RadiationOrchestrator(orbit, rps.CoreTemp(world, run.thermal), rps.OrchestratorConfig(engine="numpy"), eps=h)
    o.resync(float(h.t_s[k]) - 100.0)
    reports = []
    for s in range(1, 200):
        reports += [r[0] for r in o.advance_to(float(h.t_s[k]) - 100.0 + 20.0 * s)]
        if len(reports) >= 14:
            break
    log = " ".join(o.log)
    assert "flushed to a fresh OBC checkpoint" in log and "Bank B 48 V breaker OPEN" in log
    assert o.protected.obc.checkpoints_taken >= 2
    late = reports[-1]
    assert all(o.protected.display_state(b) == rps.HIBERNATED for b in rps.BANK_B) and late.accepted_units == 16
    assert late.rel_throughput == pytest.approx(0.5 * 0.8)                                            # half the blades, 80 % clocks


def test_scheduler_without_eps_is_unchanged():
    import radiation_pytorch_scheduler as rps
    env = rps.StepEnv(0.0, 20.0, 0.0, -150.0, 600.0)
    assert env.bank_b_powered and env.dvfs_speed == 1.0 and env.eps_mode is None


# ---- command line --------------------------------------------------------------------------------------------------------------------------
def test_command_line_report(capsys):
    assert ep.main(["--hours", "4", "--degradation", "90", "--eclipse-min", "45"]) == 0
    out = capsys.readouterr().out
    assert "eclipse 1:" in out and "Busbars:" in out and "Eclipse Survival" in out and "VRM band 45.5-50.0 V OK" in out
