import numpy as np
import pytest

from stewartgym.kinematics import GEOMETRY, euler_from_matrix, rotation, tilt_to_leg_commands
from stewartgym.validation import sample_pose


def test_home_legs_are_equal_and_match_the_mjcf_length():
    assert GEOMETRY.home_lengths == pytest.approx(np.full(6, 0.224066), abs=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_rotation_round_trips_through_euler(seed):
    angles = np.random.default_rng(seed).uniform(-1.2, 1.2, 3)
    assert euler_from_matrix(rotation(*angles)) == pytest.approx(angles)


def test_rotation_is_orthonormal():
    rot = rotation(0.3, -0.2, 0.1)
    assert rot @ rot.T == pytest.approx(np.eye(3))
    assert np.linalg.det(rot) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(10))
def test_forward_inverts_inverse_kinematics(seed):
    pose = sample_pose(np.random.default_rng(seed))
    recovered = GEOMETRY.forward(GEOMETRY.leg_lengths(pose))
    assert recovered == pytest.approx(pose, abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_jacobian_matches_finite_differences(seed):
    pose = sample_pose(np.random.default_rng(seed))
    analytic = GEOMETRY.jacobian(pose)
    numeric = np.zeros((6, 6))
    h = 1e-6
    rot = rotation(*pose[3:6])
    for axis in range(3):
        moved = pose.copy()
        moved[axis] += h
        numeric[:, axis] = (GEOMETRY.leg_lengths(moved) - GEOMETRY.leg_lengths(pose)) / h

        twist = np.zeros(3)
        twist[axis] = h
        skew = np.array(
            [[0, -twist[2], twist[1]], [twist[2], 0, -twist[0]], [-twist[1], twist[0], 0]]
        )
        moved = pose.copy()
        moved[3:6] = euler_from_matrix((np.eye(3) + skew) @ rot)
        numeric[:, axis + 3] = (GEOMETRY.leg_lengths(moved) - GEOMETRY.leg_lengths(pose)) / h
    assert numeric == pytest.approx(analytic, abs=1e-5)


def test_tilt_commands_are_zero_at_level_and_antisymmetric_to_first_order():
    assert tilt_to_leg_commands(0.0, 0.0) == pytest.approx(np.zeros(6))
    up = tilt_to_leg_commands(0.0, 1e-4)
    down = tilt_to_leg_commands(0.0, -1e-4)
    assert up == pytest.approx(-down, abs=1e-8)


def test_workspace_fits_inside_the_actuator_stroke():
    commands = [
        tilt_to_leg_commands(a * np.deg2rad(8.0), b * np.deg2rad(8.0))
        for a in (-1, 1)
        for b in (-1, 1)
    ]
    assert np.max(np.abs(commands)) < 0.029
