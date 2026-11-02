import numpy as np
import pytest

from stewartgym.pid import PIDGains
from stewartgym.tuning import LOWER, NAMES, UPPER, _simplex, to_gains, to_vector


def test_gains_round_trip_through_the_search_space():
    gains = PIDGains(
        kp_pos=4.0, v_max=0.6, kp_vel=9.0, ki_vel=3.0, kd_vel=0.05, filter_hz=9.0, i_deadband=0.001
    )
    assert to_vector(to_gains(to_vector(gains))) == pytest.approx(to_vector(gains))


def test_out_of_range_values_are_clipped_to_the_box():
    gains = to_gains(np.full(len(NAMES), 50.0))
    assert gains.kp_pos == pytest.approx(UPPER[0])
    gains = to_gains(np.full(len(NAMES), -50.0))
    assert gains.i_deadband == pytest.approx(LOWER[-1])


def test_simplex_has_one_more_vertex_than_there_are_gains():
    simplex = _simplex(np.zeros(len(NAMES)))
    assert simplex.shape == (len(NAMES) + 1, len(NAMES))
    assert np.linalg.matrix_rank(simplex[1:] - simplex[0]) == len(NAMES)
