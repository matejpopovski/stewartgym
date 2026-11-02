"""Ball-balancing task on the Stewart platform.

Both controllers act through the same interface: a 2-vector in [-1, 1] that says which way to
push the ball, mapped to a platform tilt, through the IK, to six leg-length commands. Keeping
the IK on the controller side of the plant means neither controller has to rediscover the
kinematics, and any difference in the results comes from feedback, not from the interface.

Latency, the slew limit and the leg offsets live inside the plant, so a controller can only
ever observe them through the ball.
"""

from __future__ import annotations

import importlib.resources
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from stewartgym.kinematics import GEOMETRY, rotation
from stewartgym.scenarios import NOMINAL, PhysicsParams, sample_randomized

MODEL_PATH = str(importlib.resources.files("stewartgym") / "assets" / "stewart.xml")

CONTROL_DT = 0.02
SUBSTEPS = 10
TILT_LIMIT_RAD = np.deg2rad(8.0)
LEG_COMMAND_LIMIT_M = 0.029
LEG_SLEW_LIMIT_M_S = 0.25
PLATE_RADIUS = 0.15
PLATE_TOP_Z = 0.008
BALL_RADIUS = 0.02
EPISODE_STEPS = 500
HISTORY_FRAMES = 10
SLOW_FILTER_TAU_S = 1.5
SETTLE_SUBSTEPS = 150

OBS_DIM = 4 * HISTORY_FRAMES + 6

DROP_PENALTY = 20.0
ACTION_RATE_WEIGHT = 0.2


def reward_terms(distance: float, action_change: np.ndarray) -> float:
    """Sharp peak to reward precision near the target, broad term to give a gradient far away."""
    tracking = np.exp(-distance / 0.02) + 0.25 * np.exp(-distance / 0.08)
    return float(tracking - ACTION_RATE_WEIGHT * np.dot(action_change, action_change))


@dataclass(frozen=True)
class EpisodeSpec:
    start_xy: tuple[float, float]
    target_xy: tuple[float, float]
    kick_step: int
    kick_velocity_xy: tuple[float, float]


def _polar(rng: np.random.Generator, low: float, high: float) -> tuple[float, float]:
    radius = rng.uniform(low, high)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return float(radius * np.cos(angle)), float(radius * np.sin(angle))


def sample_training_episode(rng: np.random.Generator, level: float = 1.0) -> EpisodeSpec:
    """Start radius, target spread and kick size grow with the curriculum level."""
    return EpisodeSpec(
        start_xy=_polar(rng, 0.0, 0.04 + 0.08 * level),
        target_xy=_polar(rng, 0.0, 0.06 * level),
        kick_step=int(rng.integers(150, 400)),
        kick_velocity_xy=_polar(rng, 0.0, 0.35 * level),
    )


def sample_test_episode(rng: np.random.Generator) -> EpisodeSpec:
    """Step response from rest for 6 s, then a 0.3 m/s kick, then 4 s to recover."""
    return EpisodeSpec(
        start_xy=_polar(rng, 0.06, 0.12),
        target_xy=_polar(rng, 0.0, 0.06),
        kick_step=300,
        kick_velocity_xy=_polar(rng, 0.3, 0.3),
    )


class BallBalanceEnv:
    """One simulated rig. Owns its MjModel because physics scaling mutates it per episode."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        model = self.model

        self._platform_body = model.body("platform").id
        self._ball_body = model.body("ball").id
        ball_joint = model.joint("ball")
        self._ball_qpos = ball_joint.qposadr[0]
        self._ball_dof = ball_joint.dofadr[0]
        self._pair = model.pair("ball_plate").id
        self._target_site = model.site("target").id
        self._nominal_mass = model.body_mass.copy()
        self._nominal_inertia = model.body_inertia.copy()
        self._nominal_friction = model.pair_friction[self._pair].copy()

        pose = GEOMETRY.home_pose
        self._home_lengths = GEOMETRY.leg_lengths(pose)
        self._platform_anchors = GEOMETRY.platform
        self._base_anchors = GEOMETRY.base
        self._home_origin = pose[:3]

        self.params = NOMINAL
        self._applied_params: PhysicsParams | None = None
        self._slew_per_substep = LEG_SLEW_LIMIT_M_S * model.opt.timestep

        self.episode: EpisodeSpec | None = None
        self.leg_offsets = np.zeros(6)
        self.step_count = 0
        self.last_action = np.zeros(2)
        self.measured_xy = np.zeros(2)
        self._history = np.zeros((HISTORY_FRAMES, 4))
        self._slow_error = np.zeros(2)
        self._slow_action = np.zeros(2)
        self._slow_alpha = CONTROL_DT / (SLOW_FILTER_TAU_S + CONTROL_DT)
        self._noise_rng = np.random.default_rng(0)
        self._commanded = np.zeros(6)
        self._pending: deque[np.ndarray] = deque()

    def _apply_params(self, params: PhysicsParams) -> None:
        model = self.model
        self.params = params
        if params == self._applied_params:
            return
        ball, platform = self._ball_body, self._platform_body
        model.body_mass[:] = self._nominal_mass
        model.body_inertia[:] = self._nominal_inertia
        model.body_mass[ball] *= params.ball_mass_scale
        model.body_inertia[ball] *= params.ball_mass_scale
        model.body_mass[platform] *= params.platform_mass_scale
        model.body_inertia[platform] *= params.platform_mass_scale

        friction = self._nominal_friction.copy()
        friction[:2] *= params.sliding_friction_scale
        friction[3:] *= params.rolling_friction_scale
        model.pair_friction[self._pair] = friction

        # Constraint softness is defined relative to quantities cached from the masses, so
        # scaling masses without this would quietly change the closure stiffness too.
        mujoco.mj_setConst(model, self.data)
        self._applied_params = params

    def reset(
        self,
        seed: int | np.random.SeedSequence,
        params: PhysicsParams | None = None,
        episode: EpisodeSpec | None = None,
        randomize_level: float | None = None,
        curriculum_level: float = 1.0,
        test_episode: bool = False,
    ) -> np.ndarray:
        """Start an episode whose randomness depends on `seed` alone.

        Episode layout, sensor noise and leg offsets draw from separate child streams, so two
        controllers given one seed see the same sensor noise however they behave.
        """
        sequence = (
            seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        )
        episode_stream, noise_stream, offset_stream, physics_stream = sequence.spawn(4)
        if randomize_level is not None:
            params = sample_randomized(np.random.default_rng(physics_stream), randomize_level)
        self._apply_params(NOMINAL if params is None else params)
        if episode is None:
            episode_rng = np.random.default_rng(episode_stream)
            episode = (
                sample_test_episode(episode_rng)
                if test_episode
                else sample_training_episode(episode_rng, curriculum_level)
            )
        self.episode = episode
        self._noise_rng = np.random.default_rng(noise_stream)
        signs = np.random.default_rng(offset_stream).choice([-1.0, 1.0], size=6)
        self.leg_offsets = signs * self.params.leg_offset_m

        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        model.site_pos[self._target_site][:2] = episode.target_xy

        self._commanded = self.leg_offsets.copy()
        delay_substeps = int(round(self.params.latency_s / model.opt.timestep))
        self._pending = deque(self.leg_offsets.copy() for _ in range(delay_substeps))
        data.ctrl[:] = self._commanded

        parked = data.qpos[self._ball_qpos : self._ball_qpos + 3].copy()
        parked[2] += 1.0
        for _ in range(SETTLE_SUBSTEPS):
            data.qpos[self._ball_qpos : self._ball_qpos + 3] = parked
            data.qvel[self._ball_dof : self._ball_dof + 6] = 0.0
            mujoco.mj_step(model, data)

        origin = data.xpos[self._platform_body]
        rot = data.xmat[self._platform_body].reshape(3, 3)
        local = np.array([*episode.start_xy, PLATE_TOP_Z + BALL_RADIUS])
        data.qpos[self._ball_qpos : self._ball_qpos + 3] = origin + rot @ local
        data.qvel[self._ball_dof : self._ball_dof + 6] = 0.0
        data.time = 0.0
        mujoco.mj_forward(model, data)

        self.step_count = 0
        self.last_action = np.zeros(2)
        self._slow_action = np.zeros(2)
        self._measure()
        error = self.target_xy - self.measured_xy
        self._slow_error = error.copy()
        self._history[:] = np.concatenate([error, np.zeros(2)])
        return self._observation()

    @property
    def target_xy(self) -> np.ndarray:
        return np.asarray(self.episode.target_xy)

    def ball_xy(self) -> np.ndarray:
        """True position in the plate frame: for scoring, never given to controllers."""
        rot = self.data.xmat[self._platform_body].reshape(3, 3)
        offset = self.data.xpos[self._ball_body] - self.data.xpos[self._platform_body]
        return (rot.T @ offset)[:2]

    def ball_height_above_plate(self) -> float:
        rot = self.data.xmat[self._platform_body].reshape(3, 3)
        offset = self.data.xpos[self._ball_body] - self.data.xpos[self._platform_body]
        return float((rot.T @ offset)[2] - PLATE_TOP_Z)

    def _measure(self) -> None:
        noise = self._noise_rng.normal(0.0, 1.0, 2) * self.params.sensor_noise_m
        self.measured_xy = self.ball_xy() + noise

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [self._history.ravel(), self._slow_error, self._slow_action, self.target_xy]
        )

    def leg_commands(self, action: np.ndarray) -> np.ndarray:
        """Action to slide command. Pitch moves the ball along +x and negative roll along +y."""
        pose_rot = rotation(-action[1] * TILT_LIMIT_RAD, action[0] * TILT_LIMIT_RAD)
        vectors = self._home_origin + self._platform_anchors @ pose_rot.T - self._base_anchors
        lengths = np.sqrt(np.sum(vectors * vectors, axis=1))
        return lengths - self._home_lengths

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        model, data = self.model, self.data
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        target = np.clip(
            self.leg_commands(action) + self.leg_offsets, -LEG_COMMAND_LIMIT_M, LEG_COMMAND_LIMIT_M
        )

        if self.step_count == self.episode.kick_step:
            data.qvel[self._ball_dof : self._ball_dof + 2] += self.episode.kick_velocity_xy

        pending, slew = self._pending, self._slew_per_substep
        commanded = self._commanded
        for _ in range(SUBSTEPS):
            pending.append(target)
            delayed = pending.popleft()
            commanded = commanded + np.clip(delayed - commanded, -slew, slew)
            data.ctrl[:] = commanded
            mujoco.mj_step(model, data)
        self._commanded = commanded

        self.step_count += 1
        ball = self.ball_xy()
        distance = float(np.linalg.norm(ball - self.target_xy))
        fell = float(np.linalg.norm(ball)) > PLATE_RADIUS or self.ball_height_above_plate() < 0.0
        change = action - self.last_action
        reward = reward_terms(distance, change)
        if fell:
            reward -= DROP_PENALTY

        self.last_action = action
        self._measure()
        error = self.target_xy - self.measured_xy
        alpha = self._slow_alpha
        self._slow_error += alpha * (error - self._slow_error)
        self._slow_action += alpha * (action - self._slow_action)
        self._history[1:] = self._history[:-1]
        self._history[0] = np.concatenate([error, action])

        truncated = self.step_count >= EPISODE_STEPS and not fell
        info = {"ball_xy": ball, "distance": distance, "action_change": change}
        return self._observation(), reward, fell, truncated, info
