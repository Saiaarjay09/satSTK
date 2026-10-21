"""
eps_power_engine.py - the electrical power subsystem (EPS) of the Alpha-Sigma orbital AI datacentre: solar array, 8-module Li-Ion battery with its
BMS and BCDU, the 48 V copper busbars, and the autonomous load-shedding state machine that carries the 32 server blades through the eclipse.

It is a time-stepped, closed-loop solver that runs on top of satSTK's other engines:

  orbital_engine.py            illumination (1 sunlit .. 0 umbra, a ramp in the penumbra), the beta angle and the Sun distance of every time step
  thermal_loop.py              the cooling loop. Every step it hands this module the core temperature and the throttle state, and receives back the waste
                               heat (compute + pumps + lasers + battery Joule heat + busbar I^2 R + converter losses) and the pump speed: the loop is closed.
  radiation_pytorch_scheduler  reads the Bank B breaker, the DVFS speed and the survival checkpoint flush from the EPS history (see that module)

  SOLAR      P = A_eff * eta(T_cell) * G * cos(theta) * illumination;   eta = 0.32 * (1 - 0.0025 * max(0, T_cell - 28 C));   G = 1361 W/m^2 * (1 AU / d)^2
             The wings track the Sun about the orbit normal (the SADA), so the remaining incidence angle is the beta angle: cos(theta) = cos(beta).
             The array temperature follows a first-order lag toward its radiative equilibrium (a cold array leaving eclipse is more efficient).
  BATTERY    8 modules x 40 kWh = 320 kWh at a nominal 100 V (modules in parallel). A Shepherd open-circuit voltage curve plus a Thevenin R0 + (R1 || C1)
             branch gives the terminal voltage; the current solves  P_terminal = (V_oc - V_1) * I - R0 * I^2.  SoC is Coulomb-counted:
             dSoC = -eta_c * I * dt / Q   (equivalent to SoC += eta_c * (P_solar - P_load) * dt / E for small steps).  Joule heat Q = I^2 (R0 + R1).
  BUSBARS    four 60 x 15 mm copper bars (a supply and a return for each bank), rho(T) = 1.68e-8 (1 + 0.00393 (T - 20)), exact resistive drop from the base
             (Z = -3700 mm) to every blade tap (Z = -3200 .. +3200 mm) with the blades drawing constant power, so I = P / V at each tap.
  POLICY     SUNLIT PEAK (sunlit, SoC >= 90 %): full compute.   SUNLIT RECHARGE (sunlit, SoC < 90 %): 35 % of the array is reserved for the BCDU charger and
             compute scales to the rest.   ECLIPSE CRUISE (eclipse, SoC >= 35 %): DVFS eco-clocking (-40 % power, 80 % speed).   ECLIPSE SURVIVAL (SoC < 35 %):
             checkpoint flush, Bank B breaker open, OISL lasers to low power, pumps to 40 % RPM; below 25 % SoC (this module's own addition) Bank A compute is
             shed too, to defend the 20 % floor.

COMPUTED vs ASSUMED: the array area, efficiency, solar constant, battery size, bus voltage and busbar geometry are the brief's numbers. Everything else (blade
TDP, pump / laser / housekeeping power, converter efficiencies, battery resistances, the OCV curve, array thermal properties, the radiator area, the bus
temperature model, the DVFS factors and every threshold's hysteresis) is a PLACEHOLDER chosen to give plausible numbers; nothing is measured. The blade TDP
in particular (1.2 kW) is chosen so that the orbit-average budget closes with a 182.4 m^2 array; it is far below what 768 real accelerators draw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

import orbital_engine as oe
import thermal_loop as tl

SUNLIT_PEAK, SUNLIT_RECHARGE, ECLIPSE_CRUISE, ECLIPSE_SURVIVAL = "SUNLIT PEAK", "SUNLIT RECHARGE", "ECLIPSE CRUISE", "ECLIPSE SURVIVAL"
MODES = (SUNLIT_PEAK, SUNLIT_RECHARGE, ECLIPSE_CRUISE, ECLIPSE_SURVIVAL)
N_BLADES, BANK_SIZE = 32, 16
VRM_MIN_V, VRM_MAX_V = 45.5, 50.0
SOC_FLOOR = 0.20


@dataclass
class EPSConfig:
    # ---- solar array (the brief's numbers) ----
    array_area_m2: float = 182.4
    panels_total: int = 12                       # two 6-panel accordion wings
    panels_failed: int = 0
    cell_efficiency: float = 0.32
    derate_per_c: float = 0.0025                 # -0.25 % per degree C above 28 C
    derate_ref_c: float = 28.0
    sada_error_deg: float = 0.0                  # extra pointing error of the Sun-tracking drive
    solar_reg_eff: float = 0.97                  # array regulator / MPPT, placeholder
    array_absorptivity: float = 0.92             # placeholders for the array's own temperature
    array_emissivity_front: float = 0.85
    array_emissivity_back: float = 0.80
    array_earth_ir_w_m2: float = 60.0
    array_tau_s: float = 300.0
    # ---- battery ----
    n_modules: int = 8
    module_kwh: float = 40.0
    nominal_v: float = 100.0
    soh: float = 1.0                             # state of health: 1.0 = full capacity (the degradation slider lowers it)
    soc_initial: float = 1.0
    coulomb_eff: float = 0.99                    # charging
    r0_ohm: float = 0.004                        # whole pack, 50 % SoC, 25 C, new
    r1_ohm: float = 0.002
    tau1_s: float = 60.0
    voc_e0: float = 101.0                        # Shepherd curve: V = E0 - K (1/SoC - 1) + A exp(-B (1 - SoC))
    voc_k: float = 1.0
    voc_a: float = 4.0
    voc_b: float = 8.0
    bcdu_eff: float = 0.97
    bcdu_max_charge_w: float = 40_000.0
    bcdu_max_discharge_w: float = 150_000.0
    # ---- 48 V busbars (the brief's numbers) ----
    bus_v: float = 48.0                          # BCDU regulation set-point at the base
    bar_width_m: float = 0.060
    bar_thickness_m: float = 0.015
    bus_length_m: float = 8.0
    z_base_m: float = -3.7
    z_first_blade_m: float = -3.2
    z_last_blade_m: float = 3.2
    bus_tau_s: float = 600.0
    bus_conductance_w_per_k: float = 6.0         # per bar to its surroundings (placeholder)
    # ---- loads (placeholders) ----
    blade_tdp_w: float = 1200.0
    blade_idle_frac: float = 0.08
    pump_w: float = 2500.0
    oisl_w: float = 2000.0
    oisl_low_w: float = 700.0
    housekeeping_w: float = 1500.0
    dvfs_power: float = 0.60                     # DVFS eco-clock: 20 % lower clocks / voltage cut the power to ~60 % ...
    dvfs_speed: float = 0.80                     # ... while keeping 80 % of the training speed
    pump_rpm_survival: float = 0.40
    # ---- policy ----
    recharge_reserve: float = 0.35
    soc_peak: float = 0.90
    soc_peak_exit: float = 0.91                  # hysteresis (not in the brief): leave RECHARGE at 91 %
    soc_survival: float = 0.35
    soc_survival_exit: float = 0.40              # hysteresis (not in the brief)
    soc_critical: float = 0.25                   # this module's addition: shed Bank A compute too
    soc_floor: float = SOC_FLOOR
    sunlit_threshold: float = 0.5                # illumination below this counts as eclipse
    soft_throttle_c: float = 75.0                # must equal radiation_pytorch_scheduler.SOFT_THROTTLE_C (a test checks)
    soft_release_c: float = 70.0
    # ---- stress controls ----
    force_eclipse_min: float = 0.0               # lengthen every eclipse to at least this many minutes (0 = off)
    # ---- coupling with the thermal loop ----
    radiator_area_m2: float = 160.0              # the Alpha-Sigma radiator wings (a placeholder; satSTK's default 40 m^2 is sized for 10 kW)

    @property
    def usable_area_m2(self) -> float:
        return self.array_area_m2 * (self.panels_total - self.panels_failed) / self.panels_total

    @property
    def capacity_kwh(self) -> float:
        return self.n_modules * self.module_kwh * self.soh

    @property
    def capacity_ah(self) -> float:
        return self.capacity_kwh * 1000.0 / self.nominal_v

    @property
    def bar_area_m2(self) -> float:
        return self.bar_width_m * self.bar_thickness_m

    def validate(self) -> None:
        if not 0 <= self.panels_failed <= self.panels_total:
            raise ValueError("panels_failed must be between 0 and the number of panels")
        if not 0.05 <= self.soh <= 1.0:
            raise ValueError("soh (state of health) must be between 0.05 and 1")
        if not 0.0 <= self.soc_initial <= 1.0:
            raise ValueError("soc_initial must be between 0 and 1")


# ====================================================================================================================================
# Solar array
# ====================================================================================================================================
def solar_constant_at(sun_eci: np.ndarray) -> np.ndarray:
    """1361 W/m^2 scaled by the inverse square of the Earth-Sun distance (+3.4 % at the January perihelion)."""
    d = np.linalg.norm(np.atleast_2d(sun_eci), axis=1)
    return oe.SOLAR_CONSTANT * (oe.AU_KM / d) ** 2


def cell_efficiency(t_cell_c: float, cfg: EPSConfig) -> float:
    """Triple-junction GaAs efficiency, derated 0.25 %/C above 28 C (cooler than that: no bonus)."""
    return cfg.cell_efficiency * max(0.0, 1.0 - cfg.derate_per_c * max(0.0, t_cell_c - cfg.derate_ref_c))


def solar_power_w(flux_w_m2: float, cos_incidence: float, illumination: float, t_cell_c: float, cfg: EPSConfig) -> float:
    return cfg.usable_area_m2 * cell_efficiency(t_cell_c, cfg) * flux_w_m2 * max(0.0, cos_incidence) * max(0.0, min(1.0, illumination))


def array_equilibrium_k(flux_w_m2: float, cos_incidence: float, illumination: float, eta: float, cfg: EPSConfig) -> float:
    """Radiative equilibrium of the array: absorbed sunlight not turned into electricity + Earth infrared = emitted from both faces."""
    absorbed = max(0.0, cfg.array_absorptivity - eta) * flux_w_m2 * max(0.0, cos_incidence) * illumination + cfg.array_earth_ir_w_m2 * cfg.array_emissivity_back
    t4 = absorbed / ((cfg.array_emissivity_front + cfg.array_emissivity_back) * oe.STEFAN_BOLTZMANN) + 3.0 ** 4
    return t4 ** 0.25


# ====================================================================================================================================
# Battery (Shepherd open-circuit voltage + Thevenin R0 + R1||C1)
# ====================================================================================================================================
def shepherd_voc(soc: float, cfg: EPSConfig) -> float:
    s = min(max(soc, 0.02), 1.0)
    return cfg.voc_e0 - cfg.voc_k * (1.0 / s - 1.0) + cfg.voc_a * math.exp(-cfg.voc_b * (1.0 - s))


def internal_resistance_ohm(soc: float, t_bat_c: float, cfg: EPSConfig) -> float:
    """Pack R0: rises at low SoC and when cold, and with ageing (the degradation slider)."""
    s = min(max(soc, 0.0), 1.0)
    cold = min(max(math.exp(0.02 * (25.0 - t_bat_c)), 0.6), 3.0)
    return cfg.r0_ohm * (1.0 + 1.5 * (1.0 - s) ** 3) * cold * (1.0 + 2.0 * (1.0 - cfg.soh))


def battery_current(p_terminal_w: float, e_eff_v: float, r0: float):
    """
    Solve  P = E I - R0 I^2  for the pack current (positive = discharge). Returns (I, deliverable) where `deliverable` is False when the pack cannot supply that
    power at all (the discriminant is negative) and the current is then the maximum-power one.
    """
    disc = e_eff_v * e_eff_v - 4.0 * r0 * p_terminal_w
    if disc < 0.0:
        return e_eff_v / (2.0 * r0), False
    return (e_eff_v - math.sqrt(disc)) / (2.0 * r0), True


# ====================================================================================================================================
# 48 V busbars
# ====================================================================================================================================
def copper_resistivity(t_c: float) -> float:
    return 1.68e-8 * (1.0 + 0.00393 * (t_c - 20.0))


def blade_z_m(cfg: EPSConfig) -> np.ndarray:
    """Blade 1 .. 32 tap heights: evenly spaced from -3200 mm to +3200 mm."""
    return np.linspace(cfg.z_first_blade_m, cfg.z_last_blade_m, N_BLADES)


def _segment_lengths(cfg: EPSConfig) -> list:
    """Per bank: the length of the conductor run feeding each of its 16 blades (the first run includes the cable from the BCDU to the base)."""
    z = blade_z_m(cfg)
    feeder = max(cfg.bus_length_m - (cfg.z_last_blade_m - cfg.z_base_m), 0.0)
    out = []
    for lo in (0, BANK_SIZE):
        zz = z[lo:lo + BANK_SIZE]
        seg = np.diff(np.concatenate([[cfg.z_base_m], zz]))
        seg[0] += feeder
        out.append(seg)
    return out


def busbar_profile(p_bank_a_w: float, p_bank_b_w: float, t_bus_c: float, cfg: EPSConfig, iterations: int = 4) -> dict:
    """
    Exact resistive distribution of the 48 V bus. Each bank has a supply bar and a return bar, so the drop is twice a single bar's. The blades of a bank
    tap their bar in turn and draw constant power (I = P / V), each blade a sixteenth of its bank's power. Returns the delivered voltage at every blade, the
    current in each bank's bars, the total and per-bank ohmic loss and the worst drop.
    """
    rho = copper_resistivity(t_bus_c)
    r_per_m = rho / cfg.bar_area_m2
    segs = _segment_lengths(cfg)
    v = np.full(N_BLADES, cfg.bus_v)
    p = np.concatenate([np.full(BANK_SIZE, p_bank_a_w / BANK_SIZE), np.full(BANK_SIZE, p_bank_b_w / BANK_SIZE)])
    i = np.zeros(N_BLADES)
    loss = [0.0, 0.0]
    cur = [0.0, 0.0]
    for _ in range(iterations):
        for b, lo in enumerate((0, BANK_SIZE)):
            vb = v[lo:lo + BANK_SIZE]
            ib = p[lo:lo + BANK_SIZE] / vb
            suffix = np.cumsum(ib[::-1])[::-1]                                   # current in the run below each blade
            drop = 2.0 * r_per_m * np.cumsum(segs[b] * suffix)
            v[lo:lo + BANK_SIZE] = cfg.bus_v - drop
            i[lo:lo + BANK_SIZE] = ib
            loss[b] = float(2.0 * r_per_m * np.sum(segs[b] * suffix ** 2))
            cur[b] = float(suffix[0])
    # one last pass so that the currents match the final voltages
    for b, lo in enumerate((0, BANK_SIZE)):
        ib = p[lo:lo + BANK_SIZE] / v[lo:lo + BANK_SIZE]
        suffix = np.cumsum(ib[::-1])[::-1]
        i[lo:lo + BANK_SIZE] = ib
        loss[b] = float(2.0 * r_per_m * np.sum(segs[b] * suffix ** 2))
        cur[b] = float(suffix[0])
    return dict(v_blade=v, i_blade=i, i_bank_a=cur[0], i_bank_b=cur[1], i_total=cur[0] + cur[1], loss_w=loss[0] + loss[1], loss_a_w=loss[0], loss_b_w=loss[1],
                v_min=float(v.min()), v_max=float(v.max()), drop_max_v=float(cfg.bus_v - v.min()), r_per_m=r_per_m,
                in_band=bool(v.min() >= VRM_MIN_V and v.max() <= VRM_MAX_V))


# ====================================================================================================================================
# Illumination helpers
# ====================================================================================================================================
def eclipse_runs(illumination: np.ndarray, threshold: float = 0.5) -> list:
    """[(first_index, last_index_exclusive)] of every contiguous eclipse."""
    dark = np.asarray(illumination) < threshold
    edges = np.diff(np.concatenate([[0], dark.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def extend_eclipses(illumination: np.ndarray, step_s: float, minutes: float, threshold: float = 0.5) -> np.ndarray:
    """A copy of the illumination in which every eclipse lasts at least `minutes` (the umbra is held, the exit is delayed). Longer eclipses are left alone."""
    out = np.array(illumination, dtype=float)
    want = int(round(minutes * 60.0 / step_s))
    for a, b in eclipse_runs(out, threshold):
        if b - a < want:
            out[b:min(a + want, len(out))] = 0.0
    return out


# ====================================================================================================================================
# The closed-loop controller
# ====================================================================================================================================
@dataclass
class EPSHistory:
    t_s: np.ndarray
    illumination: np.ndarray
    cos_incidence: np.ndarray
    solar_w: np.ndarray
    load_w: np.ndarray                  # electrical demand on the bus (blades + pumps + lasers + housekeeping + busbar loss)
    bat_power_w: np.ndarray             # + discharging, - charging, at the BCDU bus side
    soc: np.ndarray
    dod_pct: np.ndarray
    bat_current_a: np.ndarray
    c_rate: np.ndarray
    pack_v: np.ndarray
    r_int_mohm: np.ndarray
    bat_heat_w: np.ndarray
    bus_current_a: np.ndarray           # both banks' bars together
    bus_loss_w: np.ndarray
    p_bank_a_w: np.ndarray
    p_bank_b_w: np.ndarray
    p_fixed_w: np.ndarray               # pumps + lasers + housekeeping
    v_min: np.ndarray                   # delivered voltage at the worst blade
    v_blade_a1: np.ndarray
    v_blade_32: np.ndarray
    t_bus_c: np.ndarray
    t_array_c: np.ndarray
    t_bat_c: np.ndarray
    heat_w: np.ndarray                  # waste heat handed to the cooling loop
    shunt_w: np.ndarray                 # solar power nobody could use (battery full)
    compute_frac: np.ndarray
    dvfs: np.ndarray                    # bool
    bank_b_closed: np.ndarray           # bool: the Bank B 48 V solid-state breaker
    bank_b_hibernated: np.ndarray       # bool: Bank B asleep because of the thermal soft throttle
    oisl_low_power: np.ndarray          # bool
    pump_rpm: np.ndarray
    critical: np.ndarray                # bool: SoC below the critical level, Bank A compute shed
    brownout: np.ndarray                # bool: the battery could not deliver the demand
    mode: list
    flush_steps: list                   # step indices where the survival checkpoint flush happened
    cfg: EPSConfig = field(default_factory=EPSConfig)

    @property
    def n(self) -> int:
        return len(self.t_s)

    def index_at(self, t: float) -> int:
        step = float(self.t_s[1] - self.t_s[0]) if self.n > 1 else 1.0
        return int(np.clip(round(t / step), 0, self.n - 1))

    def throughput(self) -> np.ndarray:
        """Training throughput relative to 32 blades at full clocks."""
        a = np.where(self.bank_b_closed & ~self.bank_b_hibernated, 1.0, 0.0)
        return self.compute_frac * np.where(self.dvfs, self.cfg.dvfs_speed, 1.0) * (1.0 + a) / 2.0

    def blades(self, i: int) -> dict:
        """The full 32-blade voltage profile at step i (recomputed from the stored bank powers and bus temperature)."""
        return busbar_profile(float(self.p_bank_a_w[i]), float(self.p_bank_b_w[i]), float(self.t_bus_c[i]), self.cfg)

    def eps_state(self, t: float) -> dict:
        """The values streamed to satCAD's digital twin."""
        i = self.index_at(t)
        return dict(soc_pct=float(self.soc[i] * 100.0), busbar_current_a=float(self.bus_current_a[i]), busbar_temp_c=float(self.t_bus_c[i]),
                    bank_b_breaker_closed=bool(self.bank_b_closed[i]), mode=self.mode[i], solar_kw=float(self.solar_w[i] / 1000.0), load_kw=float(self.load_w[i] / 1000.0),
                    min_blade_v=float(self.v_min[i]))

    def summary(self) -> dict:
        step = float(self.t_s[1] - self.t_s[0]) if self.n > 1 else 1.0
        runs = eclipse_runs(self.illumination, self.cfg.sunlit_threshold)
        eclipses = []
        for a, b in runs:
            eclipses.append(dict(start_min=float(self.t_s[a] / 60.0), duration_min=(b - a) * step / 60.0, soc_entry_pct=float(self.soc[a] * 100.0),
                                 soc_min_pct=float(self.soc[a:b + 1].min() * 100.0), dod_pct=float((self.soc[a] - self.soc[a:b + 1].min()) * 100.0),
                                 energy_kwh=float(np.sum(self.load_w[a:b]) * step / 3.6e6)))
        modes = {m: float(sum(1 for x in self.mode if x == m) * step / 60.0) for m in MODES}
        return dict(eclipses=eclipses, soc_min_pct=float(self.soc.min() * 100.0), floor_violation_steps=int(np.sum(self.soc < self.cfg.soc_floor - 1e-9)),
                    brownout_steps=int(self.brownout.sum()), v_min=float(self.v_min.min()), v_max=float(np.max(self.v_blade_a1)),
                    drop_max_v=float(self.cfg.bus_v - self.v_min.min()), vrm_ok=bool(self.v_min.min() >= VRM_MIN_V), peak_solar_kw=float(self.solar_w.max() / 1000.0),
                    peak_load_kw=float(self.load_w.max() / 1000.0), peak_bat_current_a=float(np.abs(self.bat_current_a).max()),
                    max_c_rate=float(self.c_rate.max()), mode_minutes=modes, flushes=len(self.flush_steps), max_bus_current_a=float(self.bus_current_a.max()),
                    max_bus_loss_w=float(self.bus_loss_w.max()), max_bat_heat_w=float(self.bat_heat_w.max()), bank_b_open_min=float((~self.bank_b_closed).sum() * step / 60.0),
                    throughput_mean=float(self.throughput().mean()))


class EPSController:
    """
    The stateful per-step solver. Instances are callable with the signature `thermal_loop.simulate_loop` expects of a coupler, so the thermal loop advances the
    electrical state once per time step and receives the heat it produced.
    """

    def __init__(self, cfg: EPSConfig, illumination: np.ndarray, cos_incidence: np.ndarray, flux: np.ndarray, step_s: float, t_s: np.ndarray | None = None):
        cfg.validate()
        self.cfg, self.step_s = cfg, float(step_s)
        n = len(illumination)
        self.ill, self.cosi, self.flux = np.asarray(illumination, float), np.asarray(cos_incidence, float), np.asarray(flux, float)
        self.soc, self.v1 = cfg.soc_initial, 0.0
        self.mode, self.prev_mode = SUNLIT_PEAK, SUNLIT_PEAK
        self.bank_b_open_latched = False
        self.hibernated = False
        self.critical = False
        self.t_array_k = array_equilibrium_k(self.flux[0], self.cosi[0], self.ill[0], cfg.cell_efficiency, cfg)
        self.t_bus_c = 25.0
        self.nominal_heat_w = self._full_load_w() * 1.03
        z = np.zeros
        self.h = dict(t=(np.arange(n) * step_s) if t_s is None else np.asarray(t_s, float), solar=z(n), load=z(n), batp=z(n), soc=z(n), dod=z(n), ibat=z(n), crate=z(n), vpack=z(n),
                      rint=z(n), bheat=z(n), ibus=z(n), bloss=z(n), pa=z(n), pb=z(n), pfix=z(n), vmin=z(n), va1=z(n), v32=z(n), tbus=z(n), tarr=z(n), tbat=z(n),
                      heat=z(n), shunt=z(n), cfrac=z(n), dvfs=z(n, bool), bclosed=np.ones(n, bool), bhib=z(n, bool), oisl_low=z(n, bool), rpm=np.ones(n),
                      crit=z(n, bool), brown=z(n, bool))
        self.modes: list = [SUNLIT_PEAK] * n
        self.flush_steps: list = []

    def _full_load_w(self) -> float:
        c = self.cfg
        return N_BLADES * c.blade_tdp_w + c.pump_w + c.oisl_w + c.housekeeping_w

    # ---- the policy ------------------------------------------------------------------------------------------------------------
    def _decide_mode(self, sunlit: bool) -> str:
        c, s = self.cfg, self.soc
        if sunlit:
            if self.prev_mode == SUNLIT_RECHARGE:
                return SUNLIT_PEAK if s >= c.soc_peak_exit else SUNLIT_RECHARGE
            return SUNLIT_PEAK if s >= c.soc_peak else SUNLIT_RECHARGE
        if self.prev_mode == ECLIPSE_SURVIVAL:
            return ECLIPSE_SURVIVAL if s < c.soc_survival_exit else ECLIPSE_CRUISE
        return ECLIPSE_CRUISE if s >= c.soc_survival else ECLIPSE_SURVIVAL

    def __call__(self, i: int, t_core_c: float, thermal_mode: str, thermal_frac: float, t_rad_c: float):
        c, dt = self.cfg, self.step_s
        h = self.h
        ill, cosi, g = float(self.ill[i]), float(self.cosi[i]), float(self.flux[i])
        sunlit = ill >= c.sunlit_threshold
        # ---- solar
        t_arr_c = self.t_array_k - 273.15
        p_solar = solar_power_w(g, cosi, ill, t_arr_c, c)
        p_sol_bus = c.solar_reg_eff * p_solar
        # ---- thermal soft throttle of Bank B (same rule as radiation_pytorch_scheduler)
        if not self.hibernated and t_core_c >= c.soft_throttle_c:
            self.hibernated = True
        elif self.hibernated and t_core_c <= c.soft_release_c:
            self.hibernated = False
        # ---- mode
        mode = self._decide_mode(sunlit)
        survival = mode == ECLIPSE_SURVIVAL
        if survival and self.prev_mode != ECLIPSE_SURVIVAL:
            self.flush_steps.append(i)                                           # flush PyTorch checkpoints to the OBC
        if survival:
            self.bank_b_open_latched = True
            self.critical = self.soc < (c.soc_critical + (0.03 if self.critical else 0.0))
        else:
            if self.bank_b_open_latched and (self.soc >= c.soc_survival_exit):   # the breaker recloses only once the pack has recovered
                self.bank_b_open_latched = False
            self.critical = False
        bank_b_closed = not self.bank_b_open_latched
        dvfs = mode in (ECLIPSE_CRUISE, ECLIPSE_SURVIVAL)
        rpm = c.pump_rpm_survival if survival else 1.0
        # ---- loads
        p_pumps = c.pump_w * rpm ** 3
        oisl_low = survival
        p_oisl = 0.0 if self.critical else (c.oisl_low_w if oisl_low else c.oisl_w)
        p_fixed = p_pumps + p_oisl + c.housekeeping_w
        blade_full = c.blade_tdp_w * (c.dvfs_power if dvfs else 1.0)
        idle = c.blade_tdp_w * c.blade_idle_frac
        frac_th = {tl.NOMINAL: 1.0, tl.THROTTLE: 0.6, tl.SHUTDOWN: 0.0}.get(thermal_mode, thermal_frac)
        frac = min(1.0, frac_th)
        if self.critical:
            frac = 0.0
        if mode == SUNLIT_RECHARGE:
            b_active = bank_b_closed and not self.hibernated
            n_on = BANK_SIZE * (2 if b_active else 1)
            idle_other = 0.0 if (b_active or not bank_b_closed) else BANK_SIZE * idle              # a hibernated Bank B still idles
            budget = (1.0 - c.recharge_reserve) * p_sol_bus - p_fixed - 150.0 - idle_other         # what is left for the working blades
            frac = min(frac, min(1.0, max(0.0, (budget / n_on - idle) / max(blade_full - idle, 1.0))))
        a_power = BANK_SIZE * (idle + (blade_full - idle) * frac)
        if not bank_b_closed:
            b_power = 0.0
        elif self.hibernated:
            b_power = BANK_SIZE * idle
        else:
            b_power = BANK_SIZE * (idle + (blade_full - idle) * frac)
        bus = busbar_profile(a_power, b_power, self.t_bus_c, c)
        p_demand = a_power + b_power + p_fixed + bus["loss_w"]
        # ---- power balance and the battery
        net = p_sol_bus - p_demand
        shunt = 0.0
        e_oc = shepherd_voc(self.soc, c)
        t_bat = min(max(t_core_c - 15.0, 0.0), 45.0)
        r0 = internal_resistance_ohm(self.soc, t_bat, c)
        if net >= 0.0:
            want = min(net, c.bcdu_max_charge_w)
            taper = min(max((1.0 - self.soc) / 0.02, 0.15), 1.0) if self.soc < 1.0 else 0.0     # constant-voltage taper over the last 2 %, with a floor so it really reaches 100 %
            p_chg = want * taper
            shunt = net - p_chg
            p_term = -p_chg * c.bcdu_eff
            bcdu_loss = p_chg * (1.0 - c.bcdu_eff)
            bat_bus_w = -p_chg
        else:
            p_dis = min(-net, c.bcdu_max_discharge_w)
            p_term = p_dis / c.bcdu_eff
            bcdu_loss = p_term - p_dis
            bat_bus_w = p_dis
        i_bat, ok = battery_current(p_term, e_oc - self.v1, r0)
        brown = (not ok) or (net < -c.bcdu_max_discharge_w)
        if self.soc <= 0.0 and i_bat > 0.0:
            i_bat, brown = 0.0, True
        v_term = (e_oc - self.v1) - i_bat * r0
        q_ah = c.capacity_ah
        eta = c.coulomb_eff if i_bat < 0.0 else 1.0
        new_soc = min(max(self.soc - eta * i_bat * dt / 3600.0 / q_ah, 0.0), 1.0)
        decay = math.exp(-dt / c.tau1_s)
        self.v1 = i_bat * c.r1_ohm + (self.v1 - i_bat * c.r1_ohm) * decay
        bat_heat = i_bat * i_bat * (r0 + c.r1_ohm)
        # ---- temperatures (array, bus) for the next step
        eta_cell = cell_efficiency(t_arr_c, c)
        t_eq = array_equilibrium_k(g, cosi, ill, eta_cell, c)
        self.t_array_k += (t_eq - self.t_array_k) * (1.0 - math.exp(-dt / c.array_tau_s))
        t_bus_eq = t_core_c - 10.0 + (bus["loss_w"] / 4.0) / c.bus_conductance_w_per_k
        self.t_bus_c += (t_bus_eq - self.t_bus_c) * (1.0 - math.exp(-dt / c.bus_tau_s))
        heat = a_power + b_power + p_fixed + bus["loss_w"] + bat_heat + bcdu_loss
        # ---- record
        h["solar"][i], h["load"][i], h["batp"][i], h["soc"][i], h["dod"][i] = p_solar, p_demand, bat_bus_w, self.soc, (1.0 - self.soc) * 100.0
        h["ibat"][i], h["crate"][i], h["vpack"][i], h["rint"][i], h["bheat"][i] = i_bat, abs(i_bat) / q_ah, v_term, r0 * 1000.0, bat_heat
        h["ibus"][i], h["bloss"][i], h["pa"][i], h["pb"][i], h["pfix"][i] = bus["i_total"], bus["loss_w"], a_power, b_power, p_fixed
        h["vmin"][i], h["va1"][i], h["v32"][i] = bus["v_min"], bus["v_blade"][0], bus["v_blade"][-1]
        h["tbus"][i], h["tarr"][i], h["tbat"][i], h["heat"][i], h["shunt"][i] = self.t_bus_c, t_arr_c, t_bat, heat, shunt
        h["cfrac"][i], h["dvfs"][i], h["bclosed"][i], h["bhib"][i], h["oisl_low"][i] = frac, dvfs, bank_b_closed, self.hibernated, oisl_low
        h["rpm"][i], h["crit"][i], h["brown"][i] = rpm, self.critical, brown
        self.modes[i] = mode
        self.soc, self.prev_mode, self.mode = new_soc, mode, mode
        return heat, rpm

    def history(self, cfg: EPSConfig | None = None) -> EPSHistory:
        h = self.h
        return EPSHistory(t_s=h["t"], illumination=self.ill, cos_incidence=self.cosi, solar_w=h["solar"], load_w=h["load"], bat_power_w=h["batp"], soc=h["soc"], dod_pct=h["dod"],
                          bat_current_a=h["ibat"], c_rate=h["crate"], pack_v=h["vpack"], r_int_mohm=h["rint"], bat_heat_w=h["bheat"], bus_current_a=h["ibus"], bus_loss_w=h["bloss"],
                          p_bank_a_w=h["pa"], p_bank_b_w=h["pb"], p_fixed_w=h["pfix"], v_min=h["vmin"], v_blade_a1=h["va1"], v_blade_32=h["v32"], t_bus_c=h["tbus"],
                          t_array_c=h["tarr"], t_bat_c=h["tbat"], heat_w=h["heat"], shunt_w=h["shunt"], compute_frac=h["cfrac"], dvfs=h["dvfs"], bank_b_closed=h["bclosed"],
                          bank_b_hibernated=h["bhib"], oisl_low_power=h["oisl_low"], pump_rpm=h["rpm"], critical=h["crit"], brownout=h["brown"], mode=list(self.modes),
                          flush_steps=list(self.flush_steps), cfg=cfg or self.cfg)


@dataclass
class EPSRun:
    history: EPSHistory
    thermal: tl.ThermalHistory
    cfg: EPSConfig


def scaled_loop_config(heat_w: float, reference_heat_w: float, loop_cfg: tl.ThermalLoopConfig | None = None) -> tl.ThermalLoopConfig:
    """
    satSTK's cooling-loop defaults are sized for a 10 kW core (2 K per kW across the cold plate, a pump good for 60 L/min). For the several-times larger heat of the
    32-blade datacentre the cold-plate resistance shrinks, and the pump range, flow floor and thermal mass grow, in proportion (placeholder scaling).
    """
    base = loop_cfg or tl.ThermalLoopConfig()
    k = max(heat_w / max(reference_heat_w, 1.0), 1.0)
    return replace(base, plate_resistance_k_per_w=base.plate_resistance_k_per_w / k, flow_min_lpm=base.flow_min_lpm * k, flow_max_lpm=base.flow_max_lpm * k,
                   core_heat_capacity_j_per_k=base.core_heat_capacity_j_per_k * k)


def simulate_eps(result, cfg: EPSConfig | None = None, loop_cfg: tl.ThermalLoopConfig | None = None) -> EPSRun:
    """
    Run the closed electrical-thermal loop over a whole `SimulationResult`: the orbit gives the illumination, beta angle and Sun distance of the target satellite,
    the controller decides the load every step, and the cooling loop (given the radiator area `cfg.radiator_area_m2`) turns the heat it produced into the core
    temperature that the controller sees at the next step.
    """
    cfg = cfg or EPSConfig()
    track = result.tracks[result.target]
    step = float(result.config.step_s)
    ill = np.array(track.sun_visibility, dtype=float)
    if cfg.force_eclipse_min > 0:
        ill = extend_eclipses(ill, step, cfg.force_eclipse_min, cfg.sunlit_threshold)
    cosi = np.cos(np.radians(np.asarray(track.beta_deg, float))) * math.cos(math.radians(cfg.sada_error_deg))
    flux = solar_constant_at(result.sun_eci)
    ctrl = EPSController(cfg, ill, cosi, flux, step, result.t_s)
    base = result.config.thermal
    scale = cfg.radiator_area_m2 / max(base.radiator_area_m2, 1e-9)
    thermal_cfg = replace(base, radiator_area_m2=cfg.radiator_area_m2)
    absorbed = np.asarray(track.radiator_solar_load_w, float) * scale
    th = tl.simulate_loop(absorbed, step, thermal_cfg, scaled_loop_config(ctrl.nominal_heat_w, base.internal_dissipation_w, loop_cfg), coupler=ctrl)
    return EPSRun(ctrl.history(cfg), th, cfg)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run the EPS / battery / 48 V busbar simulation over satSTK's orbit and print a per-eclipse report (no window).")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--wing-failure", type=int, default=0, metavar="PANELS", help="number of failed solar panels (of 12)")
    ap.add_argument("--eclipse-min", type=float, default=0.0, help="lengthen every eclipse to at least this many minutes")
    ap.add_argument("--degradation", type=float, default=0.0, metavar="PCT", help="battery capacity lost, percent")
    args = ap.parse_args(argv)
    from simulation import SimulationConfig, run_simulation
    result = run_simulation(SimulationConfig(duration_h=args.hours, step_s=10.0))
    cfg = EPSConfig(panels_failed=args.wing_failure, force_eclipse_min=args.eclipse_min, soh=1.0 - args.degradation / 100.0)
    run = simulate_eps(result, cfg)
    s = run.history.summary()
    print(f"Array {cfg.usable_area_m2:.1f} m^2 -> peak {s['peak_solar_kw']:.1f} kW; battery {cfg.capacity_kwh:.0f} kWh; peak load {s['peak_load_kw']:.1f} kW")
    for k, e in enumerate(s["eclipses"]):
        print(f"  eclipse {k + 1}: T+{e['start_min']:6.1f} min, {e['duration_min']:5.1f} min long, SoC {e['soc_entry_pct']:5.1f} -> {e['soc_min_pct']:5.1f} % (DoD {e['dod_pct']:4.1f} %), {e['energy_kwh']:.1f} kWh used")
    print(f"Lowest SoC {s['soc_min_pct']:.1f} % ({'floor VIOLATED' if s['floor_violation_steps'] else 'above the 20 % floor'}); brownout steps {s['brownout_steps']}; survival flushes {s['flushes']}")
    print(f"Busbars: up to {s['max_bus_current_a']:.0f} A, worst blade {s['v_min']:.3f} V (drop {s['drop_max_v'] * 1000:.0f} mV, loss up to {s['max_bus_loss_w']:.0f} W); VRM band "
          f"{VRM_MIN_V}-{VRM_MAX_V} V {'OK' if s['vrm_ok'] else 'VIOLATED'}; battery heat up to {s['max_bat_heat_w']:.0f} W, up to {s['max_c_rate']:.2f} C")
    print("Time in each mode (min): " + ", ".join(f"{m.title()} {v:.0f}" for m, v in s["mode_minutes"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
