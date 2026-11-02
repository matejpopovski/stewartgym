"""Cascaded PID baseline: position loop outside, velocity loop inside, one copy per axis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from stewartgym.env import BALL_RADIUS, CONTROL_DT, TILT_LIMIT_RAD

GRAVITY = 9.81
# A rolling solid sphere accelerates at 5/7 g sin(tilt), so this is the gain from a normalized
# action to ball acceleration near level. Scaling by it keeps the gains in physical units.
ACCEL_PER_ACTION = 5.0 / 7.0 * GRAVITY * np.sin(TILT_LIMIT_RAD)


@dataclass(frozen=True)
class PIDGains:
    """Gains in physical units. kp_vel is 1/s, as the loop's output is an acceleration."""

    kp_pos: float = 3.0
    v_max: float = 0.4
    kp_vel: float = 6.0
    ki_vel: float = 2.0
    kd_vel: float = 0.0
    filter_hz: float = 8.0
    i_deadband: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class CascadedPID:
    """P on position gives a velocity setpoint; PID on velocity gives a ball acceleration.

    The integrator stops when the output saturates (conditional integration) and, optionally,
    when the ball is inside `i_deadband`. Without the second rule rolling resistance traps the
    ball off-target and the integrator ramps until it breaks free, which gives stick-slip hunting.
    """

    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self.reset()

    def reset(self) -> None:
        self._first = True
        self._previous = np.zeros(2)
        self._velocity = np.zeros(2)
        self._previous_velocity_error = np.zeros(2)
        self._integral = np.zeros(2)

    def act(self, measured_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        g = self.gains
        if self._first:
            self._previous = measured_xy.copy()
            self._first = False
        rate = (measured_xy - self._previous) / CONTROL_DT
        self._previous = measured_xy.copy()
        alpha = 1.0 - np.exp(-2.0 * np.pi * g.filter_hz * CONTROL_DT)
        self._velocity += alpha * (rate - self._velocity)

        error = target_xy - measured_xy
        speed_setpoint = g.kp_pos * error
        norm = np.linalg.norm(speed_setpoint)
        if norm > g.v_max:
            speed_setpoint *= g.v_max / norm

        velocity_error = speed_setpoint - self._velocity
        derivative = (velocity_error - self._previous_velocity_error) / CONTROL_DT
        self._previous_velocity_error = velocity_error

        accel = g.kp_vel * velocity_error + g.kd_vel * derivative + self._integral
        action = accel / ACCEL_PER_ACTION
        saturated = np.abs(action) >= 1.0
        outside_band = np.linalg.norm(error) > g.i_deadband
        self._integral += (
            g.ki_vel * velocity_error * CONTROL_DT * (~saturated if outside_band else False)
        )
        return np.clip(action, -1.0, 1.0)


def loop_shaped_gains(
    crossover_hz: float = 1.6, delay_s: float = 0.045, filter_hz: float = 8.0
) -> PIDGains:
    """Analytic starting point from the linearised plant, used to seed the numerical tuner.

    Around level the ball is a double integrator. The inner loop closes velocity at
    `crossover_hz`, and the position loop sits a factor of four below it, which keeps the
    cascade separated and leaves phase margin for the delay given in `delay_s`.
    """
    omega = 2.0 * np.pi * crossover_hz
    margin_loss = omega * delay_s + np.arctan(omega / (2.0 * np.pi * filter_hz))
    if margin_loss > np.pi / 3.0:
        raise ValueError("crossover too high for this delay: less than 60 degrees of margin left")
    return PIDGains(
        kp_pos=omega / 4.0,
        v_max=0.4,
        kp_vel=omega,
        ki_vel=omega * omega / 16.0,
        kd_vel=0.0,
        filter_hz=filter_hz,
        i_deadband=BALL_RADIUS / 20.0,
    )
