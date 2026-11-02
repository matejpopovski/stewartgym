import numpy as np
import pytest

from stewartgym.env import EPISODE_STEPS, BallBalanceEnv
from stewartgym.evaluation import (
    KICK_STEP,
    PHASE_SECONDS,
    RECOVERY_SECONDS,
    ControllerSpec,
    PIDController,
    episode_metrics,
    evaluate_matrix,
    run_episode,
)
from stewartgym.pid import PIDGains
from stewartgym.scenarios import NOMINAL, RANDOMIZATION, SCENARIO_BY_NAME, SCENARIOS

ZERO = np.zeros((EPISODE_STEPS, 2))


def metrics_for(distance, dropped=False):
    return episode_metrics(np.asarray(distance), ZERO[: len(distance)], 0.0, dropped)


def test_settling_time_is_the_last_exit_from_the_band():
    distance = np.full(EPISODE_STEPS, 0.001)
    distance[:50] = 0.05
    distance[120] = 0.02
    result = metrics_for(distance)
    assert result.settled
    assert result.settling_time_s == pytest.approx(121 * 0.02)


def test_a_ball_that_never_settles_is_flagged_and_capped():
    result = metrics_for(np.full(EPISODE_STEPS, 0.02))
    assert not result.settled
    assert result.settling_time_s == pytest.approx(PHASE_SECONDS)
    assert not result.recovered
    assert result.recovery_time_s == pytest.approx(RECOVERY_SECONDS)


def test_kick_metrics_measure_only_the_second_phase():
    distance = np.full(EPISODE_STEPS, 0.001)
    distance[KICK_STEP : KICK_STEP + 40] = 0.03
    result = metrics_for(distance)
    assert result.kick_peak_m == pytest.approx(0.03)
    assert result.recovered
    assert result.recovery_time_s == pytest.approx(40 * 0.02)
    assert result.settling_time_s == 0.0
    assert result.steady_state_error_m == pytest.approx(0.001)


def test_drop_before_the_kick_leaves_later_metrics_undefined():
    result = metrics_for(np.full(100, 0.05), dropped=True)
    assert result.dropped
    assert not result.settled
    assert np.isnan(result.steady_state_error_m)
    assert np.isnan(result.kick_peak_m)
    assert not result.recovered


def test_drop_after_the_kick_counts_as_unrecovered():
    distance = np.full(KICK_STEP + 30, 0.001)
    result = metrics_for(distance, dropped=True)
    assert result.settled
    assert not result.recovered
    assert np.isnan(result.kick_peak_m)


def test_two_controllers_see_identical_episodes():
    env = BallBalanceEnv()
    seen = []
    for gains in (PIDGains(), PIDGains(kp_pos=1.0)):
        params = SCENARIO_BY_NAME["offset_edge"].params
        trace = run_episode(env, PIDController(gains), 77, params)
        seen.append((env.episode, env.leg_offsets.copy(), trace.distance[0]))
    assert seen[0][0] == seen[1][0]
    assert seen[0][1] == pytest.approx(seen[1][1])
    # Different gains must still give different closed-loop results.
    trace_a = run_episode(env, PIDController(PIDGains()), 77, NOMINAL)
    trace_b = run_episode(env, PIDController(PIDGains(kp_pos=1.0)), 77, NOMINAL)
    assert not np.allclose(trace_a.distance[100:], trace_b.distance[100:])


def test_matrix_runs_every_pair_on_the_same_seeds():
    specs = [
        ControllerSpec("a", "pid", gains=PIDGains()),
        ControllerSpec("b", "pid", gains=PIDGains(kp_pos=1.0)),
    ]
    scenarios = [("nominal", NOMINAL), ("latency", SCENARIO_BY_NAME["latency_edge"].params)]
    rows = evaluate_matrix(specs, scenarios, episodes=3, workers=2, chunk=2)
    assert len(rows) == 12
    sets = {
        (r["controller"], r["scenario"]): sorted(
            x["episode_seed"]
            for x in rows
            if x["controller"] == r["controller"] and x["scenario"] == r["scenario"]
        )
        for r in rows
    }
    assert len({tuple(value) for value in sets.values()}) == 1


def inside_box(params):
    for name, (_, low, high) in RANDOMIZATION.items():
        if not low - 1e-12 <= getattr(params, name) <= high + 1e-12:
            return False
    return True


def test_edge_scenarios_sit_inside_the_training_box_and_beyond_ones_outside():
    for scenario in SCENARIOS:
        if scenario.severity == "edge":
            assert inside_box(scenario.params), scenario.name
        if scenario.severity == "beyond":
            assert not inside_box(scenario.params), scenario.name
