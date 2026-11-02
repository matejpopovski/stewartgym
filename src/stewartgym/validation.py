"""Numerical checks of the MJCF model against the independent analytic kinematics.

A closed-chain model can look right and still be wrong: a mistyped anchor shows up as a slightly
stressed mechanism, not as an error. So each check here compares the simulator with something
that does not depend on it, and returns the measured value for tests and reports to use.
"""

from __future__ import annotations

import importlib.resources

import mujoco
import numpy as np

from stewartgym.env import EPISODE_STEPS, BallBalanceEnv
from stewartgym.kinematics import GEOMETRY, euler_from_matrix, rotation, tilt_to_leg_commands
from stewartgym.pid import CascadedPID, PIDGains
from stewartgym.scenarios import NOMINAL

MODEL_PATH = str(importlib.resources.files("stewartgym") / "assets" / "stewart.xml")
LEGS = range(6)


def load_chain(gravity: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The mechanism with the ball removed, so every DoF in the model belongs to the chain."""
    spec = mujoco.MjSpec.from_file(MODEL_PATH)
    spec.delete(spec.pair("ball_plate"))
    spec.delete(spec.body("ball"))
    model = spec.compile()
    if not gravity:
        model.opt.gravity[:] = 0.0
    return model, mujoco.MjData(model)


def closure_gaps(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Distance between each leg tip and its platform anchor, in metres."""
    return np.array(
        [
            np.linalg.norm(
                data.site_xpos[model.site(f"leg{i}_tip").id]
                - data.site_xpos[model.site(f"plat{i}").id]
            )
            for i in LEGS
        ]
    )


def slide_positions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.array([data.qpos[model.joint(f"leg{i}_slide").qposadr[0]] for i in LEGS])


def platform_pose(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body = model.body("platform").id
    return np.concatenate(
        [data.xpos[body], euler_from_matrix(data.xmat[body].reshape(3, 3))],
    )


def settle(
    model: mujoco.MjModel, data: mujoco.MjData, commands: np.ndarray, steps: int = 1500
) -> None:
    data.ctrl[:] = commands
    for _ in range(steps):
        mujoco.mj_step(model, data)


def equality_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense closure Jacobian, one row per scalar constraint."""
    mujoco.mj_forward(model, data)
    rows = data.ne
    if mujoco.mj_isSparse(model):
        dense = np.zeros((data.nefc, model.nv))
        mujoco.mju_sparse2dense(
            dense, data.efc_J, data.efc_J_rownnz, data.efc_J_rowadr, data.efc_J_colind
        )
    else:
        dense = data.efc_J.reshape(data.nefc, model.nv)
    return dense[:rows].copy()


def sample_pose(rng: np.random.Generator) -> np.ndarray:
    """A pose inside the working volume, wider than the task needs in every axis."""
    pose = GEOMETRY.home_pose
    pose[:2] += rng.uniform(-0.008, 0.008, 2)
    pose[2] += rng.uniform(-0.01, 0.01)
    pose[3:5] = rng.uniform(np.deg2rad(-8.0), np.deg2rad(8.0), 2)
    pose[5] = rng.uniform(np.deg2rad(-4.0), np.deg2rad(4.0))
    return pose


def leg_commands_for(pose: np.ndarray) -> np.ndarray:
    return GEOMETRY.leg_lengths(pose) - GEOMETRY.home_lengths


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    relative = rotation(*a[3:6]).T @ rotation(*b[3:6])
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def check_mobility(samples: int = 25, seed: int = 0) -> dict[str, float]:
    """Closure rank and the residual mobility of the mechanism across the workspace.

    Grubler's count for a 6-UPS gives 6. The rank test confirms it in the model as built, and
    the locked-slide test confirms the six slides really do control all six freedoms.
    """
    model, data = load_chain(gravity=False)
    rng = np.random.default_rng(seed)
    slides = [model.joint(f"leg{i}_slide").dofadr[0] for i in LEGS]
    ranks, locked_ranks, worst_condition = [], [], 0.0
    for _ in range(samples):
        mujoco.mj_resetData(model, data)
        settle(model, data, leg_commands_for(sample_pose(rng)))
        jac = equality_jacobian(model, data)
        ranks.append(np.linalg.matrix_rank(jac, tol=1e-9))
        lock = np.zeros((6, model.nv))
        lock[np.arange(6), slides] = 1.0
        locked_ranks.append(np.linalg.matrix_rank(np.vstack([jac, lock]), tol=1e-9))
        ik_jac = GEOMETRY.jacobian(platform_pose(model, data))
        worst_condition = max(worst_condition, float(np.linalg.cond(ik_jac)))
    links = 2 + 2 * 6
    joints = 6 * 3
    freedoms = 6 * (2 + 1 + 3)
    return {
        "tree_dof": int(model.nv),
        "closure_rows": int(jac.shape[0]),
        "grubler_mobility": 6 * (links - joints - 1) + freedoms,
        "min_closure_rank": int(min(ranks)),
        "max_closure_rank": int(max(ranks)),
        "min_rank_with_slides_locked": int(min(locked_ranks)),
        "worst_ik_jacobian_condition": worst_condition,
    }


def check_nullspace_matches_ik(samples: int = 25, seed: int = 1) -> dict[str, float]:
    """Free motions of the simulated chain must obey the analytic IK Jacobian.

    The null space of MuJoCo's closure Jacobian is every velocity the assembled chain allows.
    Mapping each null vector through the analytic J must reproduce its slide rates, which ties
    the constraint rows in the engine to a formula derived with no reference to the engine.
    """
    model, data = load_chain(gravity=False)
    rng = np.random.default_rng(seed)
    slides = [model.joint(f"leg{i}_slide").dofadr[0] for i in LEGS]
    platform_dof = model.joint("platform").dofadr[0]
    body = model.body("platform").id
    worst = 0.0
    for _ in range(samples):
        mujoco.mj_resetData(model, data)
        settle(model, data, leg_commands_for(sample_pose(rng)))
        jac = equality_jacobian(model, data)
        _, _, vt = np.linalg.svd(jac)
        null = vt[np.linalg.matrix_rank(jac, tol=1e-9) :]
        rot = data.xmat[body].reshape(3, 3)
        ik_jac = GEOMETRY.jacobian(platform_pose(model, data))
        for velocity in null:
            linear = velocity[platform_dof : platform_dof + 3]
            angular = rot @ velocity[platform_dof + 3 : platform_dof + 6]
            predicted = ik_jac @ np.concatenate([linear, angular])
            actual = velocity[slides]
            worst = max(worst, float(np.max(np.abs(predicted - actual))))
    return {"worst_slide_rate_error_per_unit_velocity": worst}


def check_ik_against_sim(samples: int = 40, seed: int = 2) -> dict[str, float]:
    """Command poses through the IK and measure what the simulated platform does.

    Run with gravity off for the pure kinematic error, and on for the error that a real load
    adds through the finite servo stiffness.
    """
    results = {}
    for label, gravity in (("zero_g", False), ("gravity", True)):
        model, data = load_chain(gravity=gravity)
        rng = np.random.default_rng(seed)
        position, angle, slide_vs_ik, fk = [], [], [], []
        for _ in range(samples):
            mujoco.mj_resetData(model, data)
            pose = sample_pose(rng)
            settle(model, data, leg_commands_for(pose))
            achieved = platform_pose(model, data)
            position.append(np.linalg.norm(achieved[:3] - pose[:3]))
            angle.append(_rotation_error(achieved, pose))
            slides = slide_positions(model, data)
            lengths = GEOMETRY.home_lengths + slides
            slide_vs_ik.append(np.max(np.abs(GEOMETRY.leg_lengths(achieved) - lengths)))
            recovered = GEOMETRY.forward(lengths)
            fk.append(np.linalg.norm(recovered[:3] - achieved[:3]))
        results[f"{label}_max_position_error_m"] = float(max(position))
        results[f"{label}_max_angle_error_rad"] = float(max(angle))
        results[f"{label}_max_leg_length_vs_ik_m"] = float(max(slide_vs_ik))
        results[f"{label}_max_fk_position_error_m"] = float(max(fk))
    return results


def check_finite_difference_jacobian(seed: int = 3, step: float = 1e-4) -> dict[str, float]:
    """Step one leg in the simulator and compare the platform motion with J^-1 at that pose."""
    model, data = load_chain(gravity=False)
    pose = sample_pose(np.random.default_rng(seed))
    commands = leg_commands_for(pose)
    settle(model, data, commands, steps=2500)
    start = platform_pose(model, data)
    inverse = np.linalg.inv(GEOMETRY.jacobian(start))
    state = mujoco.MjData(model)
    worst = 0.0
    for leg in LEGS:
        mujoco.mj_copyData(state, model, data)
        moved = commands.copy()
        moved[leg] += step
        settle(model, state, moved, steps=1500)
        measured = platform_pose(model, state)[:3] - start[:3]
        predicted = (inverse[:, leg] * step)[:3]
        worst = max(worst, float(np.max(np.abs(measured - predicted))))
    return {"step_m": step, "worst_platform_position_error_m": worst}


def check_closure_under_motion(seconds: float = 6.0, seed: int = 4) -> dict[str, float]:
    """Drive all legs hard and confirm the loop stays closed while reaction forces are high."""
    model, data = load_chain()
    rng = np.random.default_rng(seed)
    frequencies = rng.uniform(0.8, 3.5, 6)
    phases = rng.uniform(0.0, 2.0 * np.pi, 6)
    worst = 0.0
    peak_rate = 0.0
    slides = [model.joint(f"leg{i}_slide").dofadr[0] for i in LEGS]
    for step in range(int(seconds / model.opt.timestep)):
        time = step * model.opt.timestep
        data.ctrl[:] = 0.012 * np.sin(2.0 * np.pi * frequencies * time + phases)
        mujoco.mj_step(model, data)
        worst = max(worst, float(closure_gaps(model, data).max()))
        peak_rate = max(peak_rate, float(np.max(np.abs(data.qvel[slides]))))
    return {"max_gap_m": worst, "peak_slide_speed_m_s": peak_rate}


def check_passive_energy(seconds: float = 0.25, speed: float = 0.06) -> dict[str, float]:
    """Coast along a closure-respecting velocity with no drive, damping or gravity.

    Kinetic energy is then the only energy in the system. Soft closures may bleed a little of
    it, but a rise would mean the constraint solver is injecting energy into the loop.
    """
    model, data = load_chain(gravity=False)
    model.dof_damping[:] = 0.0
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_resetData(model, data)
    jac = equality_jacobian(model, data)
    _, _, vt = np.linalg.svd(jac)
    # Take the null vector that moves the platform most, which is a heave or sway mode.
    platform_dof = model.joint("platform").dofadr[0]
    null = vt[np.linalg.matrix_rank(jac, tol=1e-9) :]
    velocity = max(null, key=lambda v: np.linalg.norm(v[platform_dof : platform_dof + 3]))
    data.qvel[:] = velocity * speed / np.linalg.norm(velocity[platform_dof : platform_dof + 3])
    mujoco.mj_forward(model, data)
    first = float(data.energy.sum())
    highest = lowest = first
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        current = float(data.energy.sum())
        highest, lowest = max(highest, current), min(lowest, current)
    return {
        "initial_energy_j": first,
        "relative_rise": (highest - first) / first,
        "relative_loss": (first - lowest) / first,
    }


def _roll_on_fixed_tilt(
    tilt: float, sliding: float | None, rolling: float | None
) -> tuple[float, float]:
    """Fit the ball's acceleration down a plate held still at `tilt`. Returns (fit, tilt)."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    pair = model.pair("ball_plate").id
    if sliding is not None:
        model.pair_friction[pair][:2] = sliding
    if rolling is None:
        # MuJoCo floors friction coefficients at 1e-5, so a rolling coefficient of zero is not
        # really zero. Dropping to condim 3 is the only way to remove rolling resistance.
        model.pair_dim[pair] = 3
    else:
        model.pair_friction[pair][3:] = rolling
    ball_q = model.joint("ball").qposadr[0]
    ball_v = model.joint("ball").dofadr[0]
    platform = model.body("platform").id
    ball = model.body("ball").id
    parked = data.qpos[ball_q : ball_q + 3] + np.array([0.0, 0.0, 1.0])
    data.ctrl[:] = tilt_to_leg_commands(0.0, tilt)
    for _ in range(1500):
        data.qpos[ball_q : ball_q + 3] = parked
        data.qvel[ball_v : ball_v + 6] = 0.0
        mujoco.mj_step(model, data)
    rot = data.xmat[platform].reshape(3, 3)
    actual_tilt = float(np.arctan2(rot[0, 2], rot[2, 2]))
    data.qpos[ball_q : ball_q + 3] = data.xpos[platform] + rot @ np.array([-0.09, 0.0, 0.028])
    data.qvel[ball_v : ball_v + 6] = 0.0
    times, along = [], []
    for step in range(250):
        mujoco.mj_step(model, data)
        offset = rot.T @ (data.xpos[ball] - data.xpos[platform])
        if offset[0] > 0.12:
            break
        if step >= 25:
            times.append(step * model.opt.timestep)
            along.append(offset[0])
    return float(2.0 * np.polyfit(times, along, 2)[0]), actual_tilt


def check_ball_contact(tilt_deg: float = 4.0, slip_tilt_deg: float = 8.0) -> dict[str, float]:
    """Compare ball acceleration on a held tilt with closed-form results.

    Three cases: pure rolling, 5/7 g sin(t); rolling resistance as modelled, which subtracts
    (mu_r / r) cos(t) inside the bracket; and low friction, where a ball beyond the slip limit
    tan(t) > 7/2 mu slides at g (sin(t) - mu cos(t)). A sliding point mass would give g sin(t)
    in the first case, so this catches a contact model that fails to roll.
    """
    g, radius = 9.81, 0.02
    tilt = np.deg2rad(tilt_deg)

    pure, actual = _roll_on_fixed_tilt(tilt, None, None)
    pure_theory = 5.0 / 7.0 * g * np.sin(actual)

    mu_r = 2e-5
    resisted, actual = _roll_on_fixed_tilt(tilt, None, mu_r)
    resisted_theory = 5.0 / 7.0 * g * (np.sin(actual) - mu_r / radius * np.cos(actual))

    mu = 0.02
    slipping, actual = _roll_on_fixed_tilt(np.deg2rad(slip_tilt_deg), mu, None)
    slip_theory = g * (np.sin(actual) - mu * np.cos(actual))

    return {
        "pure_rolling_fit_m_s2": pure,
        "pure_rolling_theory_m_s2": float(pure_theory),
        "pure_rolling_relative_error": float(abs(pure - pure_theory) / pure_theory),
        "rolling_resistance_fit_m_s2": resisted,
        "rolling_resistance_theory_m_s2": float(resisted_theory),
        "rolling_resistance_relative_error": float(
            abs(resisted - resisted_theory) / resisted_theory
        ),
        "slip_fit_m_s2": slipping,
        "slip_theory_m_s2": float(slip_theory),
        "slip_relative_error": float(abs(slipping - slip_theory) / slip_theory),
    }


def _tilt_response(timestep: float, seconds: float) -> np.ndarray:
    """Ball path in the plate frame under a fixed open-loop tilt sequence."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    ball = model.body("ball").id
    platform = model.body("platform").id
    ball_q = model.joint("ball").qposadr[0]
    data.qpos[ball_q] = -0.03
    mujoco.mj_forward(model, data)
    path = []
    steps_per_sample = round(0.02 / timestep)
    for sample in range(int(seconds / 0.02)):
        time = sample * 0.02
        tilt = 0.03 * np.sin(2.0 * np.pi * 1.2 * time)
        data.ctrl[:] = tilt_to_leg_commands(0.4 * tilt, tilt)
        for _ in range(steps_per_sample):
            mujoco.mj_step(model, data)
        rot = data.xmat[platform].reshape(3, 3)
        path.append((rot.T @ (data.xpos[ball] - data.xpos[platform]))[:2])
    return np.array(path)


def check_timestep_convergence(seconds: float = 3.0) -> dict[str, float]:
    """How far the 2 ms simulation used for control sits from a 0.25 ms reference."""
    reference = _tilt_response(0.00025, seconds)
    errors = {}
    for timestep in (0.002, 0.001):
        path = _tilt_response(timestep, seconds)
        errors[f"max_path_error_at_{timestep * 1000:g}ms_m"] = float(
            np.max(np.linalg.norm(path - reference, axis=1))
        )
    errors["ball_path_extent_m"] = float(np.max(np.abs(reference)))
    return errors


def check_closure_in_task(seed: int = 21) -> dict[str, float]:
    """Largest loop gap in a closed-loop episode, including the kick and the slew-limited moves."""
    env = BallBalanceEnv()
    env.reset(seed, NOMINAL, test_episode=True)
    pid = CascadedPID(PIDGains())
    worst = 0.0
    for _ in range(EPISODE_STEPS):
        env.step(pid.act(env.measured_xy, env.target_xy))
        worst = max(worst, float(closure_gaps(env.model, env.data).max()))
    return {"max_gap_m": worst}


def run_all() -> dict[str, dict[str, float]]:
    return {
        "mobility": check_mobility(),
        "nullspace_vs_ik": check_nullspace_matches_ik(),
        "ik_vs_sim": check_ik_against_sim(),
        "finite_difference_jacobian": check_finite_difference_jacobian(),
        "closure_under_motion": check_closure_under_motion(),
        "closure_in_task": check_closure_in_task(),
        "passive_energy": check_passive_energy(),
        "ball_contact": check_ball_contact(),
        "timestep_convergence": check_timestep_convergence(),
    }
