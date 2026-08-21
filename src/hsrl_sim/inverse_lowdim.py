from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from .mie_forward import compute_aerosol_optics
from .optical_observation import optical_vector
from .schemas import AerosolState


Arm = Literal["base", "plus"]


@dataclass(frozen=True)
class RetrievalResult:
    state_estimate: AerosolState
    converged: bool
    residual_norm: float
    objective: float
    quality_flags: tuple[str, ...]
    selected_start: int
    n_starts: int
    initial_parameters: tuple[float, float, float, float]
    near_optimal_count: int
    maximum_log_parameter_spread: float
    boundary_hit: bool
    log_parameter_covariance: tuple[tuple[float, ...], ...] | None


DEFAULT_LOWER = np.array([1e-14, 1e-14, 0.02e-6, 0.20e-6], dtype=float)
DEFAULT_UPPER = np.array([5e-10, 5e-10, 0.50e-6, 5.00e-6], dtype=float)
DEFAULT_INITIAL = np.array([1.5e-12, 1.5e-12, 0.10e-6, 1.0e-6], dtype=float)


def detect_near_optimal_solutions(
    log_parameters: np.ndarray,
    chi2: np.ndarray,
    delta_chi2: float = 4.0,
) -> tuple[int, float]:
    """Count near-optimal starts and report their largest log-parameter spread."""

    parameters = np.asarray(log_parameters, dtype=float)
    objective = np.asarray(chi2, dtype=float)
    if parameters.ndim != 2 or parameters.shape[0] != objective.size:
        raise ValueError("log_parameters rows must match chi2 values")
    finite = np.all(np.isfinite(parameters), axis=1) & np.isfinite(objective)
    if not np.any(finite):
        return 0, float("nan")
    threshold = float(np.min(objective[finite]) + delta_chi2)
    selected = parameters[finite & (objective <= threshold)]
    if len(selected) < 2:
        return len(selected), 0.0
    maximum_spread = max(
        float(np.max(np.abs(selected[left] - selected[right])))
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    )
    return len(selected), maximum_spread


def _prediction(parameters, template_state, wavelengths_m, radius_grid_m, arm):
    state = AerosolState(
        fine_volume=float(parameters[0]),
        coarse_volume=float(parameters[1]),
        fine_rv_m=float(parameters[2]),
        coarse_rv_m=float(parameters[3]),
        fine_sigma_g=template_state.fine_sigma_g,
        coarse_sigma_g=template_state.coarse_sigma_g,
        fine_refractive_index=template_state.fine_refractive_index,
        coarse_refractive_index=template_state.coarse_refractive_index,
        relative_humidity=template_state.relative_humidity,
        scenario_id=template_state.scenario_id,
        truth_id=template_state.truth_id,
    )
    vector = optical_vector(compute_aerosol_optics(state, wavelengths_m, radius_grid_m))
    return vector if arm == "plus" else vector[:-1]


def retrieve_bimodal_state(
    observed: np.ndarray,
    covariance: np.ndarray,
    wavelengths_m: np.ndarray,
    radius_grid_m: np.ndarray,
    template_state: AerosolState,
    arm: Arm,
    multistart: int = 8,
    seed: int = 0,
    lower: np.ndarray = DEFAULT_LOWER,
    upper: np.ndarray = DEFAULT_UPPER,
) -> RetrievalResult:
    """Fit four released bimodal parameters with whitened bounded least squares."""

    observed = np.asarray(observed, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    expected_size = 6 if arm == "plus" else 5
    if observed.shape != (expected_size,) or covariance.shape != (expected_size, expected_size):
        raise ValueError(f"{arm} observation must be {expected_size}-element with matching covariance")
    if np.any(~np.isfinite(observed)):
        initial = np.clip(DEFAULT_INITIAL, lower, upper)
        return RetrievalResult(template_state, False, float("nan"), float("nan"), ("non_finite_observation",), -1, multistart, tuple(float(x) for x in initial), 0, float("nan"), False, None)

    covariance = 0.5 * (covariance + covariance.T)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        initial = np.clip(DEFAULT_INITIAL, lower, upper)
        return RetrievalResult(template_state, False, float("nan"), float("nan"), ("non_positive_definite_covariance",), -1, multistart, tuple(float(x) for x in initial), 0, float("nan"), False, None)

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != (4,) or upper.shape != (4,) or np.any(lower <= 0) or np.any(upper <= lower):
        raise ValueError("retrieval bounds must be positive 4-vectors with upper > lower")

    rng = np.random.default_rng(seed)
    center = np.clip(DEFAULT_INITIAL, lower, upper)
    starts = [np.clip(center, lower, upper)]
    log_lower = np.log(lower)
    log_upper = np.log(upper)
    for _ in range(max(0, multistart - 1)):
        starts.append(np.exp(rng.uniform(log_lower, log_upper)))

    best = None
    best_index = -1
    optimizer_results = []
    for index, start in enumerate(starts):
        def residual(log_parameters):
            prediction = _prediction(np.exp(log_parameters), template_state, wavelengths_m, radius_grid_m, arm)
            return solve_triangular(chol, observed - prediction, lower=True)

        result = least_squares(
            residual,
            np.log(start),
            bounds=(log_lower, log_upper),
            max_nfev=250,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        optimizer_results.append(result)
        if (
            best is None
            or (result.success and not best.success)
            or (result.success == best.success and result.cost < best.cost)
        ):
            best = result
            best_index = index

    assert best is not None
    estimate = np.exp(best.x)
    state_estimate = AerosolState(
        fine_volume=float(estimate[0]),
        coarse_volume=float(estimate[1]),
        fine_rv_m=float(estimate[2]),
        coarse_rv_m=float(estimate[3]),
        fine_sigma_g=template_state.fine_sigma_g,
        coarse_sigma_g=template_state.coarse_sigma_g,
        fine_refractive_index=template_state.fine_refractive_index,
        coarse_refractive_index=template_state.coarse_refractive_index,
        relative_humidity=template_state.relative_humidity,
        scenario_id=template_state.scenario_id,
        truth_id=template_state.truth_id,
    )
    successful = [result for result in optimizer_results if result.success]
    candidates = successful or optimizer_results
    near_optimal_count, maximum_log_spread = detect_near_optimal_solutions(
        np.asarray([result.x for result in candidates]),
        np.asarray([2.0 * result.cost for result in candidates]),
    )
    log_range = log_upper - log_lower
    normalized_distance = np.minimum(best.x - log_lower, log_upper - best.x) / log_range
    boundary_hit = bool(np.any(normalized_distance < 0.005))
    flags = []
    if not best.success:
        flags.append("optimizer_not_converged")
    if boundary_hit:
        flags.append("parameter_at_bound")
    if near_optimal_count >= 2 and maximum_log_spread > np.log(1.25):
        flags.append("multiple_near_optimal_solutions")
    information = best.jac.T @ best.jac
    information_condition = np.linalg.cond(information)
    if not np.isfinite(information_condition) or information_condition > 1e12:
        flags.append("ill_conditioned_local_covariance")
    log_parameter_covariance = np.linalg.pinv(information, hermitian=True)
    return RetrievalResult(
        state_estimate,
        bool(best.success),
        float(np.linalg.norm(best.fun)),
        float(2.0 * best.cost),
        tuple(flags),
        best_index,
        len(starts),
        tuple(float(x) for x in center),
        near_optimal_count,
        maximum_log_spread,
        boundary_hit,
        tuple(tuple(float(value) for value in row) for row in log_parameter_covariance),
    )
