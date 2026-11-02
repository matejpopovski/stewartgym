import numpy as np
import pytest

from stewartgym.env import (
    CONTROL_DT,
    EPISODE_STEPS,
    LEG_SLEW_LIMIT_M_S,
    OBS_DIM,
    PLATE_RADIUS,
    SUBSTEPS,
    TILT_LIMIT_RAD,
    BallBalanceEnv,
    EpisodeSpec,
    reward_terms,
    sample_test_episode,
)
from stewartgym.kinematics import euler_from_matrix
from stewartgym.pid import CascadedPID, PIDGains
from stewartgym.scenarios import NOMINAL, PhysicsParams, sample_randomized

QUIET = PhysicsParams(latency_s=0.0, sensor_noise_m=0.0)
STILL = EpisodeSpec((0.0, 0.0), (0.0, 0.0), 10**9, (0.0, 0.0))


@pytest.fixture(scope="module")
def env():
    return BallBalanceEnv()


def platform_rpy(env):
    rot = env.data.xmat[env._platform_body].reshape(3, 3)
    return euler_from_matrix(rot)


def test_positive_action_pushes_the_ball_in_that_direction(env):
    for action, axis in (([1.0, 0.0], 0), ([0.0, 1.0], 1)):
        env.reset(1, QUIET, STILL)
        for _ in range(20):
            env.step(np.array(action))
        assert env.ball_xy()[axis] > 0.01


def test_full_action_reaches_the_tilt_limit(env):
    env.reset(1, QUIET, EpisodeSpec((-0.1, 0.0), (0.0, 0.0), 10**9, (0.0, 0.0)))
    for _ in range(15):
        env.step(np.array([1.0, 0.0]))
    assert platform_rpy(env)[1] == pytest.approx(TILT_LIMIT_RAD, rel=0.03)


def test_latency_delays_the_platform_response(env):
    def first_motion(latency):
        env.reset(2, PhysicsParams(latency_s=latency, sensor_noise_m=0.0), STILL)
        history = []
        for _ in range(12):
            env.step(np.array([0.5, 0.0]))
            history.append(abs(platform_rpy(env)[1]))
        return next(i for i, value in enumerate(history) if value > 0.004)

    assert first_motion(0.1) - first_motion(0.0) == pytest.approx(0.1 / CONTROL_DT, abs=1)


def test_slew_limit_caps_the_commanded_leg_rate(env):
    env.reset(3, QUIET, STILL)
    before = env._commanded.copy()
    env.step(np.array([1.0, 1.0]))
    change = np.abs(env._commanded - before).max()
    assert change <= LEG_SLEW_LIMIT_M_S * CONTROL_DT + 1e-12
    assert env.model.opt.timestep * SUBSTEPS == pytest.approx(CONTROL_DT)


def test_leg_offset_tilts_the_plate_with_no_command(env):
    env.reset(4, PhysicsParams(leg_offset_m=0.004, sensor_noise_m=0.0), STILL)
    assert np.abs(env.leg_offsets) == pytest.approx(np.full(6, 0.004))
    assert np.linalg.norm(platform_rpy(env)[:2]) > 1e-3


def test_physics_scaling_applies_and_resets_cleanly(env):
    ball, platform = env._ball_body, env._platform_body
    env.reset(0, NOMINAL, STILL)
    base_mass = env.model.body_mass[ball]
    base_platform = env.model.body_mass[platform]
    base_friction = env.model.pair_friction[env._pair].copy()
    params = PhysicsParams(
        ball_mass_scale=2.0,
        platform_mass_scale=3.0,
        sliding_friction_scale=0.5,
        rolling_friction_scale=4.0,
    )
    env.reset(0, params, STILL)
    assert env.model.body_mass[ball] == pytest.approx(2.0 * base_mass)
    assert env.model.body_mass[platform] == pytest.approx(3.0 * base_platform)
    assert env.model.pair_friction[env._pair][0] == pytest.approx(0.5 * base_friction[0])
    assert env.model.pair_friction[env._pair][3] == pytest.approx(4.0 * base_friction[3])
    env.reset(0, NOMINAL, STILL)
    assert env.model.body_mass[ball] == pytest.approx(base_mass)
    assert env.model.pair_friction[env._pair] == pytest.approx(base_friction)


def test_same_seed_gives_identical_noise_for_different_controllers(env):
    def measurements(actions):
        env.reset(99, NOMINAL, STILL)
        recorded = []
        for action in actions:
            noise_free = env.ball_xy()
            recorded.append(env.measured_xy - noise_free)
            env.step(action)
        return np.array(recorded)

    still = measurements([np.zeros(2)] * 30)
    moving = measurements([np.array([0.4, -0.3])] * 30)
    # The first reading predates any action, and later ones differ only through the ball.
    assert still[0] == pytest.approx(moving[0])
    assert np.allclose(still, moving, atol=1e-9)


def test_runs_are_reproducible(env):
    def rollout():
        env.reset(5, NOMINAL, sample_test_episode(np.random.default_rng(5)))
        pid = CascadedPID(PIDGains())
        trace = []
        for _ in range(100):
            env.step(pid.act(env.measured_xy, env.target_xy))
            trace.append(env.ball_xy().copy())
        return np.array(trace)

    assert np.array_equal(rollout(), rollout())


def test_ball_leaving_the_plate_terminates_with_a_penalty(env):
    env.reset(6, QUIET, EpisodeSpec((0.13, 0.0), (0.0, 0.0), 10**9, (0.0, 0.0)))
    for _ in range(200):
        _, reward, terminated, truncated, _ = env.step(np.array([1.0, 0.0]))
        if terminated:
            break
    assert terminated
    assert not truncated
    assert reward < -10.0
    assert np.linalg.norm(env.ball_xy()) > PLATE_RADIUS


def test_time_limit_is_a_truncation_not_a_termination(env):
    env.reset(7, QUIET, STILL)
    for _ in range(EPISODE_STEPS):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert truncated
    assert not terminated


def test_observation_shape_and_history_order(env):
    env.reset(8, QUIET, STILL)
    obs = env.reset(8, QUIET, STILL)
    assert obs.shape == (OBS_DIM,)
    action = np.array([0.25, -0.5])
    obs, *_ = env.step(action)
    assert obs[2:4] == pytest.approx(action)
    assert obs[6:8] == pytest.approx(np.zeros(2))


def test_controllers_cannot_see_the_true_ball_position(env):
    env.reset(9, PhysicsParams(sensor_noise_m=0.01), STILL)
    obs, *_ = env.step(np.zeros(2))
    true_error = env.target_xy - env.ball_xy()
    assert np.linalg.norm(obs[:2] - true_error) > 1e-4


def test_reward_peaks_at_the_target_and_penalises_chatter():
    assert reward_terms(0.0, np.zeros(2)) == pytest.approx(1.25)
    assert reward_terms(0.01, np.zeros(2)) < reward_terms(0.0, np.zeros(2))
    assert reward_terms(0.0, np.ones(2)) < reward_terms(0.0, np.zeros(2))


def test_randomized_physics_stay_in_the_box_and_collapse_to_nominal_at_level_zero():
    rng = np.random.default_rng(0)
    assert sample_randomized(rng, 0.0).latency_s == pytest.approx(NOMINAL.latency_s)
    assert sample_randomized(rng, 0.0).ball_mass_scale == pytest.approx(1.0)
    for _ in range(200):
        params = sample_randomized(rng)
        assert 0.5 <= params.ball_mass_scale <= 2.0
        assert 0.0 <= params.latency_s <= 0.06
