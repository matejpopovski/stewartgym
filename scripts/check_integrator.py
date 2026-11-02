"""Test the explanation for PID-R's friction failure: rerun it with the integrator off."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from stewartgym.evaluation import TEST_SEED_BASE, PIDController, pid_spec, run_episode, shared_env
from stewartgym.scenarios import SCENARIO_BY_NAME

RESULTS = Path(__file__).resolve().parent.parent / "results"
EPISODES = 200


def steady_error(gains, params) -> float:
    env = shared_env()
    errors = [
        run_episode(env, PIDController(gains), TEST_SEED_BASE + i, params).distance[200:300].mean()
        for i in range(EPISODES)
    ]
    return float(np.mean(errors))


def main() -> None:
    gains = pid_spec(RESULTS, "robust").gains
    params = SCENARIO_BY_NAME["friction_beyond"].params
    result = {
        "scenario": "friction_beyond",
        "episodes": EPISODES,
        "steady_error_m_pid_r": steady_error(gains, params),
        "steady_error_m_pid_r_without_integrator": steady_error(replace(gains, ki_vel=0.0), params),
    }
    (RESULTS / "integrator_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
