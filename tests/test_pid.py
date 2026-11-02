import numpy as np
import pytest

from stewartgym.env import EPISODE_STEPS, BallBalanceEnv, sample_test_episode
from stewartgym.pid import ACCEL_PER_ACTION, CascadedPID, PIDGains, loop_shaped_gains
from stewartgym.scenarios import NOMINAL, PhysicsParams


def test_proportional_paths_produce_the_expected_output():
    pid = CascadedPID(PIDGains(kp_pos=2.0, v_max=1.0, kp_vel=5.0, ki_vel=0.0, filter_hz=50.0))
    action = pid.act(np.zeros(2), np.array([0.01, 0.0]))
    assert action[0] == pytest.approx(2.0 * 0.01 * 5.0 / ACCEL_PER_ACTION)
    assert action[1] == pytest.approx(0.0)


def test_speed_setpoint_is_limited_along_the_error_direction():
    pid = CascadedPID(PIDGains(kp_pos=100.0, v_max=0.1, kp_vel=1.0, ki_vel=0.0))
    action = pid.act(np.zeros(2), np.array([0.3, 0.4]))
    assert np.linalg.norm(action * ACCEL_PER_ACTION) == pytest.approx(0.1)
    assert action[1] / action[0] == pytest.approx(0.4 / 0.3)


def test_integrator_holds_while_the_output_is_saturated():
    gains = PIDGains(kp_pos=50.0, v_max=10.0, kp_vel=50.0, ki_vel=20.0)
    pid = CascadedPID(gains)
    for _ in range(50):
        pid.act(np.zeros(2), np.array([0.1, 0.0]))
    assert pid._integral[0] == pytest.approx(0.0)


def test_integrator_rests_inside_the_deadband():
    pid = CascadedPID(PIDGains(kp_pos=5.0, kp_vel=1.0, ki_vel=5.0, i_deadband=0.01))
    for _ in range(20):
        pid.act(np.zeros(2), np.array([0.005, 0.0]))
    assert pid._integral == pytest.approx(np.zeros(2))
    for _ in range(20):
        pid.act(np.zeros(2), np.array([0.02, 0.0]))
    assert pid._integral[0] > 0.0


def test_reset_clears_state():
    pid = CascadedPID(PIDGains())
    pid.act(np.zeros(2), np.ones(2) * 0.01)
    pid.act(np.ones(2) * 0.01, np.ones(2) * 0.01)
    pid.reset()
    assert pid._integral == pytest.approx(np.zeros(2))
    assert pid._velocity == pytest.approx(np.zeros(2))


def test_loop_shaping_refuses_a_crossover_the_delay_cannot_support():
    with pytest.raises(ValueError):
        loop_shaped_gains(crossover_hz=6.0, delay_s=0.1)


def run(gains, params, seed=11):
    env = BallBalanceEnv()
    env.reset(seed, params, sample_test_episode(np.random.default_rng(seed)))
    pid = CascadedPID(gains)
    distance = []
    for _ in range(EPISODE_STEPS):
        _, _, terminated, truncated, info = env.step(pid.act(env.measured_xy, env.target_xy))
        distance.append(info["distance"])
        if terminated or truncated:
            break
    return np.array(distance)


def test_closed_loop_regulates_the_ball_to_the_target():
    distance = run(PIDGains(), NOMINAL)
    assert len(distance) == EPISODE_STEPS
    assert distance[200:300].max() < 0.01


def test_integral_action_removes_an_actuator_offset():
    params = PhysicsParams(leg_offset_m=0.002)
    with_integral = run(PIDGains(), params)
    without = run(PIDGains(ki_vel=0.0), params)
    assert with_integral[250:300].mean() < without[250:300].mean()
