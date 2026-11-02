"""Scoring: run a controller over seeded episodes and reduce each one to control metrics.

Every controller gets the same episodes: for a given seed the start, target, kick, sensor noise
and leg offsets are identical, so results can be compared episode by episode. Scores use the
true ball position, which controllers never see.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from stewartgym.env import CONTROL_DT, EPISODE_STEPS, BallBalanceEnv
from stewartgym.pid import CascadedPID, PIDGains
from stewartgym.policy import NumpyPolicy
from stewartgym.scenarios import PhysicsParams

SETTLE_BAND_M = 0.005
KICK_STEP = 300
STEADY_WINDOW = (200, 300)
PHASE_SECONDS = KICK_STEP * CONTROL_DT
RECOVERY_SECONDS = (EPISODE_STEPS - KICK_STEP) * CONTROL_DT

TEST_SEED_BASE = 20_000_000
TUNE_SEED_BASE = 10_000_000
SELECT_SEED_BASE = 15_000_000


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, env: BallBalanceEnv, observation: np.ndarray) -> np.ndarray: ...


class PIDController:
    def __init__(self, gains: PIDGains) -> None:
        self.pid = CascadedPID(gains)

    def reset(self) -> None:
        self.pid.reset()

    def act(self, env: BallBalanceEnv, observation: np.ndarray) -> np.ndarray:
        return self.pid.act(env.measured_xy, env.target_xy)


class PolicyController:
    def __init__(self, path: str | Path) -> None:
        self.policy = NumpyPolicy.load(path)

    def reset(self) -> None:
        pass

    def act(self, env: BallBalanceEnv, observation: np.ndarray) -> np.ndarray:
        return self.policy.act(observation)


@dataclass(frozen=True)
class ControllerSpec:
    """A picklable recipe, because workers build their own controllers after spawn."""

    name: str
    family: str
    seed: int = 0
    gains: PIDGains | None = None
    path: str | None = None

    def build(self) -> Controller:
        if self.gains is not None:
            return PIDController(self.gains)
        return PolicyController(self.path)


def pid_spec(results: Path, label: str) -> ControllerSpec:
    """Load gains chosen by scripts/tune_pid.py. `label` is "nominal" or "robust"."""
    record = json.loads((results / f"pid_tuning_{label}.json").read_text())
    gains = {key: value for key, value in record["chosen"].items() if key != "score"}
    return ControllerSpec(f"pid_{label}", f"pid_{label}", gains=PIDGains(**gains))


def policy_spec(policies: Path, label: str, seed: int) -> ControllerSpec:
    """Point at a saved policy. `label` is "nominal" or "dr"."""
    name = f"ppo_{label}_seed{seed}"
    return ControllerSpec(name, f"ppo_{label}", seed=seed, path=str(policies / f"{name}.npz"))


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_return: float
    dropped: bool
    survived_steps: int
    settled: bool
    settling_time_s: float
    steady_state_error_m: float
    kick_peak_m: float
    recovered: bool
    recovery_time_s: float
    action_rate_rms: float


def episode_metrics(
    distance: np.ndarray, action_changes: np.ndarray, episode_return: float, dropped: bool
) -> EpisodeMetrics:
    """Reduce one distance trace to metrics.

    Settling and recovery are the first time the ball enters the band and stays there to the
    end of that phase. A ball that fails to settle is scored at the full phase length, which
    understates the true value, so `settled` always travels with it.
    """
    steps = len(distance)
    step_phase = distance[:KICK_STEP]
    outside = np.flatnonzero(step_phase > SETTLE_BAND_M)
    last_outside = int(outside[-1]) if len(outside) else -1
    reached_kick = steps >= KICK_STEP
    settled = reached_kick and last_outside < KICK_STEP - 1
    settling = (last_outside + 1) * CONTROL_DT if settled else PHASE_SECONDS

    window = distance[STEADY_WINDOW[0] : STEADY_WINDOW[1]]
    steady = float(window.mean()) if reached_kick else float("nan")

    after = distance[KICK_STEP:]
    completed = steps >= EPISODE_STEPS and not dropped
    peak = float(after.max()) if len(after) and not dropped else float("nan")
    outside = np.flatnonzero(after > SETTLE_BAND_M)
    last_outside = int(outside[-1]) if len(outside) else -1
    recovered = completed and last_outside < len(after) - 1
    recovery = (last_outside + 1) * CONTROL_DT if recovered else RECOVERY_SECONDS

    rate = float(np.sqrt(np.mean(np.sum(action_changes**2, axis=1)))) if steps else 0.0
    return EpisodeMetrics(
        episode_return=episode_return,
        dropped=dropped,
        survived_steps=steps,
        settled=settled,
        settling_time_s=float(settling),
        steady_state_error_m=steady,
        kick_peak_m=peak,
        recovered=recovered,
        recovery_time_s=float(recovery),
        action_rate_rms=rate,
    )


@dataclass
class Trace:
    distance: np.ndarray
    action_changes: np.ndarray
    ball_xy: np.ndarray
    target_xy: np.ndarray
    episode_return: float
    dropped: bool


def run_episode(
    env: BallBalanceEnv,
    controller: Controller,
    seed: int,
    params: PhysicsParams | None = None,
    randomize_level: float | None = None,
    test_episode: bool = True,
    on_step: Callable[[BallBalanceEnv, int], None] | None = None,
) -> Trace:
    observation = env.reset(
        seed, params, randomize_level=randomize_level, test_episode=test_episode
    )
    controller.reset()
    distance = np.empty(EPISODE_STEPS)
    changes = np.empty((EPISODE_STEPS, 2))
    path = np.empty((EPISODE_STEPS, 2))
    total = 0.0
    dropped = False
    count = 0
    for step in range(EPISODE_STEPS):
        action = controller.act(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        distance[step] = info["distance"]
        changes[step] = info["action_change"]
        path[step] = info["ball_xy"]
        count = step + 1
        if on_step is not None:
            on_step(env, step)
        if terminated or truncated:
            dropped = terminated
            break
    return Trace(
        distance[:count], changes[:count], path[:count], env.target_xy.copy(), total, dropped
    )


def _run_chunk(task: tuple) -> list[dict]:
    spec, scenario_name, params, seeds = task
    env = shared_env()
    controller = spec.build()
    rows = []
    for seed in seeds:
        trace = run_episode(env, controller, seed, params)
        metrics = episode_metrics(
            trace.distance, trace.action_changes, trace.episode_return, trace.dropped
        )
        rows.append(
            {
                "controller": spec.name,
                "family": spec.family,
                "training_seed": spec.seed,
                "scenario": scenario_name,
                "episode_seed": seed,
                **asdict(metrics),
            }
        )
    return rows


_ENV: BallBalanceEnv | None = None


def shared_env() -> BallBalanceEnv:
    global _ENV
    if _ENV is None:
        _ENV = BallBalanceEnv()
    return _ENV


def evaluate_matrix(
    specs: list[ControllerSpec],
    scenarios: list[tuple[str, PhysicsParams]],
    episodes: int,
    workers: int = 8,
    chunk: int = 25,
) -> list[dict]:
    """Every controller on every scenario on the same episode seeds."""
    seeds = [TEST_SEED_BASE + i for i in range(episodes)]
    tasks = [
        (spec, name, params, seeds[start : start + chunk])
        for spec in specs
        for name, params in scenarios
        for start in range(0, episodes, chunk)
    ]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_run_chunk, tasks, chunksize=1)
    return [row for part in rows for row in part]
