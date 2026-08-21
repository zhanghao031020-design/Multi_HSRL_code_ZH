import numpy as np
import pytest

from hsrl_sim.inverse_lowdim import detect_near_optimal_solutions, retrieve_bimodal_state
from hsrl_sim.mie_forward import compute_aerosol_optics
from hsrl_sim.optical_observation import (
    lognormal_sigma_for_relative_error,
    make_optical_observation,
)
from hsrl_sim.schemas import AerosolState


WAVELENGTHS_M = np.array([355e-9, 532e-9, 1064e-9])
RADIUS_GRID_M = np.geomspace(0.01e-6, 15.0e-6, 256)


def make_state():
    return AerosolState(
        fine_volume=1.2e-12,
        coarse_volume=3.0e-12,
        fine_rv_m=0.11e-6,
        coarse_rv_m=0.95e-6,
        scenario_id="smoke",
        truth_id="truth-0",
    )


def test_base_is_strict_subvector_of_plus_observation():
    state = make_state()
    optics = compute_aerosol_optics(state, WAVELENGTHS_M, RADIUS_GRID_M)
    observation = make_optical_observation(
        optics,
        relative_error=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        seed=123,
    )

    assert np.array_equal(observation.optical_base, observation.optical_plus[:-1])
    assert observation.covariance_base.shape == (5, 5)
    assert observation.covariance_plus.shape == (6, 6)


def test_base_noise_is_identical_across_alpha1064_error_scan():
    state = make_state()
    optics = compute_aerosol_optics(state, WAVELENGTHS_M, RADIUS_GRID_M)
    low_error = make_optical_observation(
        optics,
        relative_error=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05]),
        seed=123,
    )
    high_error = make_optical_observation(
        optics,
        relative_error=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.70]),
        seed=123,
    )

    assert np.array_equal(low_error.optical_base, high_error.optical_base)
    assert not np.array_equal(low_error.optical_plus, high_error.optical_plus)


def test_lognormal_mapping_preserves_declared_relative_standard_deviation():
    for relative_error in (0.05, 0.10, 0.30, 0.70):
        sigma_log = lognormal_sigma_for_relative_error(relative_error)
        coefficient_of_variation = np.sqrt(np.exp(sigma_log**2) - 1.0)
        assert coefficient_of_variation == pytest.approx(relative_error)


def test_full_correlation_is_preserved_in_declared_covariance():
    state = make_state()
    optics = compute_aerosol_optics(state, WAVELENGTHS_M, RADIUS_GRID_M)
    correlation = np.eye(6)
    correlation[0, 3] = correlation[3, 0] = 0.35
    correlation[1, 4] = correlation[4, 1] = 0.35
    correlation[2, 5] = correlation[5, 2] = 0.35
    observation = make_optical_observation(
        optics,
        relative_error=np.full(6, 0.1),
        correlation=correlation,
        seed=17,
    )

    standard_deviation = np.sqrt(np.diag(observation.covariance_plus))
    recovered = observation.covariance_plus / np.outer(
        standard_deviation, standard_deviation
    )

    assert recovered == pytest.approx(correlation)


def test_correlated_base_noise_is_identical_across_alpha_error_scan():
    state = make_state()
    optics = compute_aerosol_optics(state, WAVELENGTHS_M, RADIUS_GRID_M)
    correlation = np.full((6, 6), 0.10)
    np.fill_diagonal(correlation, 1.0)
    low = make_optical_observation(
        optics,
        relative_error=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.05]),
        correlation=correlation,
        seed=99,
    )
    high = make_optical_observation(
        optics,
        relative_error=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.70]),
        correlation=correlation,
        seed=99,
    )

    assert np.array_equal(low.optical_base, high.optical_base)


def test_noiseless_plus_retrieval_recovers_four_parameter_state():
    state = make_state()
    optics = compute_aerosol_optics(state, WAVELENGTHS_M, RADIUS_GRID_M)
    observation = make_optical_observation(
        optics,
        relative_error=np.full(6, 0.05),
        seed=7,
        add_noise=False,
    )

    result = retrieve_bimodal_state(
        observation.optical_plus,
        observation.covariance_plus,
        wavelengths_m=WAVELENGTHS_M,
        radius_grid_m=RADIUS_GRID_M,
        template_state=state,
        arm="plus",
        multistart=4,
        seed=7,
    )

    assert result.converged
    assert result.state_estimate.fine_volume == pytest.approx(state.fine_volume, rel=0.08)
    assert result.state_estimate.coarse_volume == pytest.approx(state.coarse_volume, rel=0.08)
    assert result.state_estimate.fine_rv_m == pytest.approx(state.fine_rv_m, rel=0.08)
    assert result.state_estimate.coarse_rv_m == pytest.approx(state.coarse_rv_m, rel=0.08)
    covariance = np.asarray(result.log_parameter_covariance)
    assert covariance.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-10)


def test_default_initialization_is_independent_of_truth_parameters():
    state_a = make_state()
    state_b = AerosolState(
        fine_volume=4.0e-12,
        coarse_volume=8.0e-13,
        fine_rv_m=0.18e-6,
        coarse_rv_m=1.8e-6,
        scenario_id="different",
        truth_id="truth-1",
    )

    result_a = retrieve_bimodal_state(
        np.full(6, np.nan),
        np.eye(6),
        wavelengths_m=WAVELENGTHS_M,
        radius_grid_m=RADIUS_GRID_M,
        template_state=state_a,
        arm="plus",
        multistart=2,
        seed=1,
    )
    result_b = retrieve_bimodal_state(
        np.full(6, np.nan),
        np.eye(6),
        wavelengths_m=WAVELENGTHS_M,
        radius_grid_m=RADIUS_GRID_M,
        template_state=state_b,
        arm="plus",
        multistart=2,
        seed=1,
    )

    assert result_a.initial_parameters == pytest.approx(result_b.initial_parameters)


def test_failed_retrieval_is_explicit():
    state = make_state()
    optics = compute_aerosol_optics(state, WAVELENGTHS_M, RADIUS_GRID_M)
    observation = make_optical_observation(optics, relative_error=np.full(6, 0.1), seed=1)
    result = retrieve_bimodal_state(
        np.full(6, np.nan),
        observation.covariance_plus,
        wavelengths_m=WAVELENGTHS_M,
        radius_grid_m=RADIUS_GRID_M,
        template_state=state,
        arm="plus",
        multistart=2,
        seed=1,
    )

    assert not result.converged
    assert "non_finite_observation" in result.quality_flags
    assert result.log_parameter_covariance is None


def test_near_optimal_solutions_require_similar_cost_and_distinct_parameters():
    parameters = np.log(
        np.array([
            [1.0e-12, 2.0e-12, 0.10e-6, 1.0e-6],
            [1.0e-12, 4.0e-12, 0.10e-6, 2.0e-6],
            [1.0e-12, 2.0e-12, 0.10e-6, 1.0e-6],
        ])
    )
    chi2 = np.array([1.0, 3.0, 20.0])

    count, maximum_log_spread = detect_near_optimal_solutions(parameters, chi2)

    assert count == 2
    assert maximum_log_spread > np.log(1.25)
