"""Numerical PID tuning: global Sobol search, then Nelder-Mead, then selection on fresh seeds.

The objective is mean episode return, the exact quantity PPO maximises, on the same episode
distribution. Each candidate is scored on the same fixed seeds (common random numbers), which
makes the objective deterministic and lets a simplex method work on it.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from stewartgym.evaluation import (
    SELECT_SEED_BASE,
    TUNE_SEED_BASE,
    PIDController,
    run_episode,
    shared_env,
)
from stewartgym.pid import PIDGains, loop_shaped_gains

NAMES = ("kp_pos", "v_max", "kp_vel", "ki_vel", "kd_vel", "filter_hz", "i_deadband")
LOWER = np.array([0.5, 0.1, 1.0, 0.05, 0.001, 2.0, 0.0001])
UPPER = np.array([15.0, 1.5, 40.0, 60.0, 0.6, 24.0, 0.008])


def to_gains(log_values: np.ndarray) -> PIDGains:
    clipped = np.clip(log_values, np.log(LOWER), np.log(UPPER))
    return PIDGains(**dict(zip(NAMES, map(float, np.exp(clipped)), strict=True)))


def to_vector(gains: PIDGains) -> np.ndarray:
    values = np.array([getattr(gains, name) for name in NAMES], dtype=float)
    return np.log(np.clip(values, LOWER, UPPER))


def _episode_return(task: tuple[PIDGains, int, bool]) -> float:
    gains, seed, randomize = task
    trace = run_episode(
        shared_env(),
        PIDController(gains),
        seed,
        randomize_level=1.0 if randomize else None,
        test_episode=False,
    )
    return trace.episode_return


@dataclass
class TuningConfig:
    randomize: bool = False
    sobol_candidates: int = 128
    sobol_episodes: int = 32
    refine_starts: int = 2
    refine_evaluations: int = 150
    refine_episodes: int = 32
    select_episodes: int = 256
    seed: int = 0
    workers: int = 8


@dataclass
class TuningRecord:
    config: dict
    evaluations: list[dict] = field(default_factory=list)
    finalists: list[dict] = field(default_factory=list)
    chosen: dict = field(default_factory=dict)
    seconds: float = 0.0
    baseline: dict = field(default_factory=dict)


def tune(config: TuningConfig) -> TuningRecord:
    started = time.perf_counter()
    record = TuningRecord(config=vars(config).copy())

    with mp.get_context("spawn").Pool(config.workers) as pool:

        def score(gains: PIDGains, episodes: int, base: int) -> float:
            tasks = [(gains, base + i, config.randomize) for i in range(episodes)]
            return float(np.mean(pool.map(_episode_return, tasks, chunksize=4)))

        analytic = loop_shaped_gains()
        analytic_score = score(analytic, config.sobol_episodes, TUNE_SEED_BASE)
        record.evaluations.append({"stage": "analytic", "score": analytic_score, **vars(analytic)})

        sampler = qmc.Sobol(len(NAMES), scramble=True, seed=config.seed)
        unit = sampler.random(config.sobol_candidates)
        logs = np.log(LOWER) + unit * (np.log(UPPER) - np.log(LOWER))
        scored = [(analytic_score, to_vector(analytic))]
        for vector in logs:
            value = score(to_gains(vector), config.sobol_episodes, TUNE_SEED_BASE)
            scored.append((value, vector))
            record.evaluations.append({"stage": "sobol", "score": value, **vars(to_gains(vector))})
        scored.sort(key=lambda item: -item[0])

        finalists = [to_gains(vector) for _, vector in scored[: config.refine_starts]]
        for start in range(config.refine_starts):
            refined = _refine(scored[start][1], f"nelder_mead_{start}", score, record, config)
            finalists.append(refined)

        # Refinement can overfit its seeds, so the winner is picked on seeds it never saw.
        best = None
        for gains in finalists:
            value = score(gains, config.select_episodes, SELECT_SEED_BASE)
            record.finalists.append({"score": value, **vars(gains)})
            if best is None or value > best[0]:
                best = (value, gains)
        record.chosen = {"score": best[0], **vars(best[1])}
        record.baseline = {
            "analytic_score": score(analytic, config.select_episodes, SELECT_SEED_BASE),
            "hand_defaults_score": score(PIDGains(), config.select_episodes, SELECT_SEED_BASE),
        }

    record.seconds = time.perf_counter() - started
    return record


def _refine(
    start_vector: np.ndarray,
    stage: str,
    score,
    record: TuningRecord,
    config: TuningConfig,
) -> PIDGains:
    def cost(vector: np.ndarray) -> float:
        gains = to_gains(vector)
        value = score(gains, config.refine_episodes, TUNE_SEED_BASE)
        record.evaluations.append({"stage": stage, "score": value, **vars(gains)})
        return -value

    result = minimize(
        cost,
        start_vector,
        method="Nelder-Mead",
        options={
            "maxfev": config.refine_evaluations,
            "initial_simplex": _simplex(start_vector),
            "xatol": 1e-3,
            "fatol": 0.05,
        },
    )
    return to_gains(result.x)


def _simplex(centre: np.ndarray, spread: float = 0.25) -> np.ndarray:
    """A 7-dimensional simplex whose edges change each gain by about a quarter in log space."""
    simplex = np.tile(centre, (len(centre) + 1, 1))
    simplex[1:] += spread * np.eye(len(centre))
    return simplex
