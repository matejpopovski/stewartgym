"""The closed chain, checked numerically against analytic kinematics and closed-form dynamics."""

import mujoco
import numpy as np
import pytest

from stewartgym import validation as v
from stewartgym.kinematics import BASE_ANCHOR_Z, GEOMETRY, HOME_HEIGHT


@pytest.fixture(scope="module")
def chain():
    model, data = v.load_chain()
    mujoco.mj_forward(model, data)
    return model, data


def test_mjcf_anchors_agree_with_the_kinematic_model(chain):
    model, data = chain
    for leg in range(6):
        base = model.body(f"leg{leg}_cyl").pos
        assert base == pytest.approx(GEOMETRY.base[leg], abs=1e-6)
        anchor = model.site(f"plat{leg}").pos
        assert anchor == pytest.approx(GEOMETRY.platform[leg], abs=1e-6)
    assert model.body("platform").pos[2] == pytest.approx(HOME_HEIGHT)
    assert model.body("leg0_cyl").pos[2] == pytest.approx(BASE_ANCHOR_Z)


def test_the_compiled_model_starts_assembled(chain):
    model, data = chain
    assert v.closure_gaps(model, data).max() < 2e-6


def test_closures_are_declared_for_all_six_legs(chain):
    model, _ = chain
    assert model.neq == 6
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_CONNECT)
    assert model.nu == 6


def test_mobility_is_six_across_the_workspace():
    result = v.check_mobility()
    assert result["tree_dof"] == 24
    assert result["closure_rows"] == 18
    assert result["grubler_mobility"] == 6
    assert result["min_closure_rank"] == result["max_closure_rank"] == 18
    assert result["tree_dof"] - result["max_closure_rank"] == 6


def test_locking_the_six_slides_removes_all_mobility():
    result = v.check_mobility(samples=10)
    assert result["min_rank_with_slides_locked"] == 24
    assert result["worst_ik_jacobian_condition"] < 100.0


def test_closure_null_space_obeys_the_analytic_jacobian():
    assert v.check_nullspace_matches_ik()["worst_slide_rate_error_per_unit_velocity"] < 1e-5


def test_ik_reproduces_the_simulated_pose_without_load():
    result = v.check_ik_against_sim(samples=15)
    assert result["zero_g_max_position_error_m"] < 2e-6
    assert result["zero_g_max_angle_error_rad"] < 2e-6
    assert result["zero_g_max_leg_length_vs_ik_m"] < 2e-6
    assert result["zero_g_max_fk_position_error_m"] < 2e-6


def test_gravity_error_stays_within_servo_compliance():
    result = v.check_ik_against_sim(samples=15)
    assert result["gravity_max_position_error_m"] < 1e-3
    assert result["gravity_max_angle_error_rad"] < 2e-3
    # Length against pose still holds under load, only the servo sag differs from the command.
    assert result["gravity_max_leg_length_vs_ik_m"] < 3e-5


def test_single_leg_steps_match_the_inverse_jacobian():
    assert v.check_finite_difference_jacobian()["worst_platform_position_error_m"] < 2e-7


def test_loop_stays_closed_under_hard_motion():
    result = v.check_closure_under_motion()
    assert result["peak_slide_speed_m_s"] > 0.25
    assert result["max_gap_m"] < 1e-3


def test_loop_gap_is_negligible_during_a_task_episode():
    assert v.check_closure_in_task()["max_gap_m"] < 5e-4


def test_constraint_solver_does_not_inject_energy():
    result = v.check_passive_energy()
    assert result["initial_energy_j"] > 1e-3
    assert result["relative_rise"] < 1e-5
    assert result["relative_loss"] < 1e-3


def test_ball_rolls_and_slips_as_closed_form_mechanics_predicts():
    result = v.check_ball_contact()
    assert result["pure_rolling_relative_error"] < 0.01
    assert result["rolling_resistance_relative_error"] < 0.01
    assert result["slip_relative_error"] < 0.01


def test_a_broken_anchor_is_detected():
    """Guards the checks themselves: a 1 mm typo in the MJCF must fail assembly."""
    spec = mujoco.MjSpec.from_file(v.MODEL_PATH)
    spec.site("plat3").pos = spec.site("plat3").pos + np.array([0.001, 0.0, 0.0])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert v.closure_gaps(model, data).max() > 9e-4


def test_control_timestep_is_converged_for_this_task():
    result = v.check_timestep_convergence()
    assert result["ball_path_extent_m"] < 0.15
    assert result["max_path_error_at_2ms_m"] < 1e-3
