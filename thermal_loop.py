"""
thermal_loop.py - a small, honest model of the compute core's liquid cooling loop, for the digital twin.

The orbital engine already gives the radiator's equilibrium temperature at every time step (from the waste heat, the sunlight it absorbs and
the eclipse). This module adds what that engine does not have, the loop between the compute core and the radiator:

    compute core --(cold plate)--> coolant --(pump, bypass valve)--> radiator --> deep space

  * radiator panel     first-order lag toward the engine's equilibrium temperature (the panel has thermal mass; the engine does not)
  * coolant supply     the radiator outlet plus a small approach temperature; when the radiator is colder than the minimum supply temperature the
                       BYPASS VALVE diverts part of the flow around the radiator so the coolant does not over-cool (closed form below)
  * pump               flow follows the heat load so the coolant warms by a target number of kelvin across the cold plate (clamped to the pump's range)
  * compute core       lumped thermal mass heated by the compute load, cooled through the cold plate resistance
  * protection         the core THROTTLES the compute load at 85 degC (back to NOMINAL below 80) and SHUTS DOWN at 95 degC (restart below 70)

COMPUTED vs ASSUMED: the radiator equilibrium and the sunlight on it come from satSTK's orbit, eclipse and radiator engines. Everything in this file
(plate resistance, pump limits, thermal masses, coolant properties, thresholds) is a PLACEHOLDER chosen to give plausible numbers, not data about a
real vehicle. It is a steady-ish lumped model with explicit Euler steps, meant to make the 3D twin move believably, not to size a coolant loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import orbital_engine as oe

NOMINAL, THROTTLE, SHUTDOWN = "NOMINAL", "THROTTLE", "SHUTDOWN"


@dataclass
class ThermalLoopConfig:
    plate_resistance_k_per_w: float = 2.0e-3      # core to coolant (2 K per kW of compute heat)
    coolant_cp_j_per_kg_k: float = 3400.0         # water-glycol
    coolant_kg_per_l: float = 1.05
    coolant_rise_target_k: float = 5.0            # the pump aims for this warm-up across the cold plate
    flow_min_lpm: float = 5.0
    flow_max_lpm: float = 60.0
    supply_min_c: float = 18.0                    # the bypass valve keeps the coolant at least this warm
    radiator_approach_k: float = 3.0              # coolant leaves the radiator this much warmer than the panel
    bypass_max_pct: float = 95.0
    core_heat_capacity_j_per_k: float = 6.0e5
    radiator_time_constant_s: float = 900.0
    throttle_c: float = 85.0
    throttle_release_c: float = 80.0
    shutdown_c: float = 95.0
    shutdown_release_c: float = 70.0
    throttle_load: float = 0.6                    # fraction of the compute load kept while throttling


@dataclass
class ThermalHistory:
    core_temp_c: np.ndarray
    radiator_temp_c: np.ndarray
    coolant_flow_lpm: np.ndarray
    bypass_valve_pct: np.ndarray
    throttle_mode: list                           # "NOMINAL" / "THROTTLE" / "SHUTDOWN" per step
    load_fraction: np.ndarray                     # share of the nominal compute load actually running


def coolant_flow_lpm(heat_w: float, cfg: ThermalLoopConfig) -> float:
    """Pump flow that makes the coolant warm by `coolant_rise_target_k` at this heat load, within the pump's limits."""
    kg_s = heat_w / (cfg.coolant_rise_target_k * cfg.coolant_cp_j_per_kg_k)
    return float(np.clip(kg_s / cfg.coolant_kg_per_l * 60.0, cfg.flow_min_lpm, cfg.flow_max_lpm))


def bypass_fraction(radiator_c: float, coolant_rise_k: float, cfg: ThermalLoopConfig) -> float:
    """
    Share of the coolant diverted around the radiator so the mixed supply is no colder than `supply_min_c`.
    Mixing the radiator outlet (panel + approach) with the warm return: supply = outlet + b/(1-b) * rise, so b/(1-b) = (supply_min - outlet)/rise.
    """
    outlet = radiator_c + cfg.radiator_approach_k
    x = max(0.0, (cfg.supply_min_c - outlet) / max(coolant_rise_k, 1e-6))
    return min(x / (1.0 + x), cfg.bypass_max_pct / 100.0)


def radiator_equilibrium_c(heat_w: float, absorbed_solar_w, area_m2: float, emissivity: float, space_k: float):
    t4 = (heat_w + np.asarray(absorbed_solar_w, dtype=float)) / (emissivity * oe.STEFAN_BOLTZMANN * area_m2) + space_k ** 4
    return t4 ** 0.25 - 273.15


def simulate_loop(absorbed_solar_w: np.ndarray, step_s: float, thermal: oe.ThermalConfig, cfg: ThermalLoopConfig | None = None) -> ThermalHistory:
    """
    March the loop through the whole scenario. `absorbed_solar_w` is the sunlight absorbed by the radiator at each step (from the orbit and
    eclipse engines, `SatelliteTrack.radiator_solar_load_w`); the waste heat and radiator sizing come from `thermal` (the Conditions panel).
    """
    cfg = cfg or ThermalLoopConfig()
    n = len(absorbed_solar_w)
    p_nominal = thermal.internal_dissipation_w
    core, rad, flow, byp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    load = np.ones(n)
    modes = [NOMINAL] * n
    mode = NOMINAL
    t_rad = float(radiator_equilibrium_c(p_nominal, absorbed_solar_w[0], thermal.radiator_area_m2, thermal.emissivity, thermal.space_temperature_k))
    flow0 = coolant_flow_lpm(p_nominal, cfg)
    rise0 = p_nominal / (flow0 / 60.0 * cfg.coolant_kg_per_l * cfg.coolant_cp_j_per_kg_k)
    supply0 = max(t_rad + cfg.radiator_approach_k, cfg.supply_min_c)
    t_core = supply0 + rise0 / 2.0 + p_nominal * cfg.plate_resistance_k_per_w            # start in steady state
    alpha = 1.0 - math.exp(-step_s / cfg.radiator_time_constant_s)
    for i in range(n):
        frac = 1.0 if mode == NOMINAL else (cfg.throttle_load if mode == THROTTLE else 0.0)
        p = p_nominal * frac
        t_eq = float(radiator_equilibrium_c(p, absorbed_solar_w[i], thermal.radiator_area_m2, thermal.emissivity, thermal.space_temperature_k))
        t_rad += alpha * (t_eq - t_rad)
        q = max(coolant_flow_lpm(p, cfg), cfg.flow_min_lpm)
        mdot_cp = q / 60.0 * cfg.coolant_kg_per_l * cfg.coolant_cp_j_per_kg_k
        rise = p / mdot_cp
        b = bypass_fraction(t_rad, rise, cfg)
        supply = t_rad + cfg.radiator_approach_k + (b / (1.0 - b)) * rise
        t_plate = supply + rise / 2.0                                                   # mean coolant temperature in the cold plate
        # core: C dT/dt = P - (T - T_plate)/R   (explicit Euler; the step is far below the time constant C*R)
        sub = max(1, int(math.ceil(step_s / (0.25 * cfg.core_heat_capacity_j_per_k * cfg.plate_resistance_k_per_w))))
        h = step_s / sub
        for _ in range(sub):
            t_core += h * (p - (t_core - t_plate) / cfg.plate_resistance_k_per_w) / cfg.core_heat_capacity_j_per_k
        core[i], rad[i], flow[i], byp[i], load[i], modes[i] = t_core, t_rad, q, 100.0 * b, frac, mode
        if mode == NOMINAL and t_core >= cfg.throttle_c:
            mode = THROTTLE
        elif mode == THROTTLE:
            mode = SHUTDOWN if t_core >= cfg.shutdown_c else (NOMINAL if t_core <= cfg.throttle_release_c else THROTTLE)
        elif mode == SHUTDOWN and t_core <= cfg.shutdown_release_c:
            mode = THROTTLE
        if mode == NOMINAL and t_core >= cfg.shutdown_c:
            mode = SHUTDOWN
    return ThermalHistory(core, rad, flow, byp, modes, load)
