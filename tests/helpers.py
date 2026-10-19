"""Shared test helpers."""
import simulation as sim
from laser_link_engine import RelayConfig


def direct_cfg(**kw) -> "sim.SimulationConfig":
    """A scenario with the GEO relay network OFF: only the direct ground-to-satellite beam (what most single-hop tests want)."""
    cfg = sim.SimulationConfig(**kw)
    if "relay" not in kw:
        cfg.relay = RelayConfig(enabled=False)
    return cfg
