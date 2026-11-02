"""Analytic kinematics of the 6-UPS Stewart platform.

Inverse kinematics is closed form: a pose fixes where each platform anchor sits, and a leg
length is the distance to its base anchor. Forward kinematics has no closed form for the
general 6-6 platform, so it is a Newton iteration on the IK Jacobian. Nothing in here calls
MuJoCo: these functions are the independent model that the simulator gets checked against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_RADIUS = 0.18
PLATFORM_RADIUS = 0.11
BASE_ANCHOR_Z = 0.06
HOME_HEIGHT = 0.26
BASE_ANGLES_DEG = (-15.0, 15.0, 105.0, 135.0, 225.0, 255.0)
PLATFORM_ANGLES_DEG = (-45.0, 45.0, 75.0, 165.0, 195.0, 285.0)


def _ring(radius: float, angles_deg: tuple[float, ...], z: float) -> np.ndarray:
    angles = np.deg2rad(angles_deg)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(6, z)])


def rotation(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """World-from-platform rotation, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def euler_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rotation`, valid away from pitch = +/-90 degrees."""
    return np.array(
        [
            np.arctan2(rot[2, 1], rot[2, 2]),
            np.arctan2(-rot[2, 0], np.hypot(rot[2, 1], rot[2, 2])),
            np.arctan2(rot[1, 0], rot[0, 0]),
        ]
    )


@dataclass(frozen=True)
class StewartGeometry:
    """Anchor layout. Pose vectors are (x, y, z, roll, pitch, yaw) of the platform frame."""

    base: np.ndarray = field(
        default_factory=lambda: _ring(BASE_RADIUS, BASE_ANGLES_DEG, BASE_ANCHOR_Z)
    )
    platform: np.ndarray = field(
        default_factory=lambda: _ring(PLATFORM_RADIUS, PLATFORM_ANGLES_DEG, 0.0)
    )

    @property
    def home_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, HOME_HEIGHT, 0.0, 0.0, 0.0])

    @property
    def home_lengths(self) -> np.ndarray:
        return self.leg_lengths(self.home_pose)

    def leg_vectors(self, pose: np.ndarray) -> np.ndarray:
        rot = rotation(*pose[3:6])
        return pose[:3] + self.platform @ rot.T - self.base

    def leg_lengths(self, pose: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.leg_vectors(pose), axis=1)

    def jacobian(self, pose: np.ndarray) -> np.ndarray:
        """J in dL = J (dp, dw), where dw is a world-frame small rotation.

        Each row is the leg's unit vector and the moment of that unit vector about the
        platform origin: the Plucker line of the leg, the standard parallel-robot result.
        """
        rot = rotation(*pose[3:6])
        vectors = self.leg_vectors(pose)
        units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        arms = self.platform @ rot.T
        return np.hstack([units, np.cross(arms, units)])

    def forward(
        self,
        lengths: np.ndarray,
        guess: np.ndarray | None = None,
        tol: float = 1e-12,
        max_iter: int = 30,
    ) -> np.ndarray:
        """Solve leg lengths for the pose by Newton iteration from a nearby guess.

        The 6-6 platform has up to 40 assembly modes, so the result is whichever branch the
        guess sits on. Starting at home is correct for every pose this project reaches.
        """
        pose = self.home_pose.copy() if guess is None else np.array(guess, dtype=float)
        for _ in range(max_iter):
            residual = self.leg_lengths(pose) - lengths
            if np.max(np.abs(residual)) < tol:
                return pose
            step = np.linalg.solve(self.jacobian(pose), -residual)
            pose[:3] += step[:3]
            small = step[3:]
            angle = np.linalg.norm(small)
            if angle > 0:
                axis = small / angle
                skew = np.array(
                    [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
                )
                delta = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
                pose[3:6] = euler_from_matrix(delta @ rotation(*pose[3:6]))
        raise RuntimeError("forward kinematics did not converge")


GEOMETRY = StewartGeometry()


def tilt_to_leg_commands(
    roll: float, pitch: float, geometry: StewartGeometry = GEOMETRY
) -> np.ndarray:
    """Slide displacements, relative to home, that tilt the platform in place."""
    pose = geometry.home_pose
    pose[3:5] = roll, pitch
    return geometry.leg_lengths(pose) - geometry.home_lengths
