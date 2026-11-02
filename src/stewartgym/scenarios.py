"""Physics conditions: the nominal plant, the randomization range, and the shifted test cases."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PhysicsParams:
    ball_mass_scale: float = 1.0
    platform_mass_scale: float = 1.0
    sliding_friction_scale: float = 1.0
    rolling_friction_scale: float = 1.0
    latency_s: float = 0.02
    sensor_noise_m: float = 0.0005
    leg_offset_m: float = 0.0


NOMINAL = PhysicsParams()

# Log-uniform for multiplicative quantities, uniform for the rest. Each scenario below tagged
# "edge" sits on the boundary of this box and each tagged "beyond" sits outside it.
RANDOMIZATION = {
    "ball_mass_scale": ("log", 0.5, 2.0),
    "platform_mass_scale": ("log", 0.6, 1.7),
    "sliding_friction_scale": ("log", 0.25, 1.5),
    "rolling_friction_scale": ("log", 0.25, 10.0),
    "latency_s": ("uniform", 0.0, 0.06),
    "sensor_noise_m": ("uniform", 0.0, 0.002),
    "leg_offset_m": ("uniform", 0.0, 0.0015),
}


def sample_randomized(rng: np.random.Generator, level: float = 1.0) -> PhysicsParams:
    """Draw physics from the randomization box, shrunk toward nominal when level < 1."""
    values = {}
    for name, (kind, low, high) in RANDOMIZATION.items():
        centre = getattr(NOMINAL, name)
        if kind == "log":
            lo = np.exp(np.log(centre) + level * (np.log(low) - np.log(centre)))
            hi = np.exp(np.log(centre) + level * (np.log(high) - np.log(centre)))
            values[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            lo = centre + level * (low - centre)
            hi = centre + level * (high - centre)
            values[name] = float(rng.uniform(lo, hi))
    return PhysicsParams(**values)


@dataclass(frozen=True)
class Scenario:
    name: str
    params: PhysicsParams
    group: str
    severity: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("nominal", NOMINAL, "nominal", "nominal"),
    Scenario(
        "mass_edge",
        replace(NOMINAL, ball_mass_scale=2.0, platform_mass_scale=1.7),
        "mass",
        "edge",
    ),
    Scenario(
        "mass_beyond",
        replace(NOMINAL, ball_mass_scale=4.0, platform_mass_scale=3.0),
        "mass",
        "beyond",
    ),
    Scenario(
        "friction_edge",
        replace(NOMINAL, sliding_friction_scale=0.25, rolling_friction_scale=10.0),
        "friction",
        "edge",
    ),
    Scenario(
        "friction_beyond",
        replace(NOMINAL, sliding_friction_scale=0.1, rolling_friction_scale=40.0),
        "friction",
        "beyond",
    ),
    Scenario("latency_edge", replace(NOMINAL, latency_s=0.06), "latency", "edge"),
    Scenario("latency_beyond", replace(NOMINAL, latency_s=0.10), "latency", "beyond"),
    Scenario("noise_edge", replace(NOMINAL, sensor_noise_m=0.002), "noise", "edge"),
    Scenario("noise_beyond", replace(NOMINAL, sensor_noise_m=0.005), "noise", "beyond"),
    Scenario("offset_edge", replace(NOMINAL, leg_offset_m=0.0015), "actuator offset", "edge"),
    Scenario("offset_beyond", replace(NOMINAL, leg_offset_m=0.004), "actuator offset", "beyond"),
    Scenario(
        "combined_edge",
        PhysicsParams(
            ball_mass_scale=2.0,
            platform_mass_scale=1.7,
            sliding_friction_scale=0.25,
            rolling_friction_scale=10.0,
            latency_s=0.06,
            sensor_noise_m=0.002,
            leg_offset_m=0.0015,
        ),
        "combined",
        "edge",
    ),
    Scenario(
        "combined_beyond",
        PhysicsParams(
            ball_mass_scale=4.0,
            platform_mass_scale=3.0,
            sliding_friction_scale=0.1,
            rolling_friction_scale=40.0,
            latency_s=0.10,
            sensor_noise_m=0.005,
            leg_offset_m=0.004,
        ),
        "combined",
        "beyond",
    ),
)

SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
