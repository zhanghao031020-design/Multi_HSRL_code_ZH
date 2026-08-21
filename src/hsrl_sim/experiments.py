from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from .inverse_lowdim import RetrievalResult, retrieve_bimodal_state
from .io import source_digest, source_revision, stable_config_hash, write_rows
from .metrics import paired_product_metrics
from .mie_forward import DEFAULT_WAVELENGTHS_M, compute_aerosol_optics
from .optical_observation import make_optical_observation, optical_vector
from .distributions import aerosol_products
from .reporting import write_optical_report, write_signal_report
from .schemas import AerosolState
from .discriminator import DiscriminatorConfig
from .signal_forward import (
    CHANNEL_NAMES,
    InstrumentConfig,
    SignalNoiseConfig,
    SignalObservation,
    simulate_signal_observation,
)


@dataclass(frozen=True)
class ExperimentResult:
    retrieval_records: list[dict[str, object]]
    summary_records: list[dict[str, object]]
    arm_a_records: list[dict[str, object]]


@dataclass(frozen=True)
class SignalExperimentResult:
    retrieval_records: list[dict[str, object]]
    paired_retrieval_records: list[dict[str, object]]
    summary_records: list[dict[str, object]]


def build_optical_correlation(
    common_correlation: float = 0.10,
    same_wavelength_correlation: float = 0.35,
) -> np.ndarray:
    """Build a documented 6x6 cross-channel correlation matrix."""

    if not -1.0 < common_correlation < 1.0:
        raise ValueError("common_correlation must be between -1 and 1")
    if not -1.0 < same_wavelength_correlation < 1.0:
        raise ValueError("same_wavelength_correlation must be between -1 and 1")
    correlation = np.full((6, 6), common_correlation, dtype=float)
    np.fill_diagonal(correlation, 1.0)
    for beta_index, alpha_index in ((0, 3), (1, 4), (2, 5)):
        correlation[beta_index, alpha_index] = same_wavelength_correlation
        correlation[alpha_index, beta_index] = same_wavelength_correlation
    eigenvalues = np.linalg.eigvalsh(correlation)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("configured optical correlation is not positive semidefinite")
    return correlation


def _make_truth_states(
    seed: int,
    n_truth: int,
    model_mismatch: bool = False,
) -> list[AerosolState]:
    rng = np.random.default_rng(seed)
    scenarios = (
        ("urban_fine", 1.0e-12, 3.0e-13, 0.08e-6, 0.70e-6),
        ("smoke_mixed", 1.5e-12, 1.5e-12, 0.11e-6, 0.95e-6),
        ("marine_coarse", 4.0e-13, 5.0e-12, 0.12e-6, 1.40e-6),
    )
    states = []
    for index in range(n_truth):
        scenario, fine_v, coarse_v, fine_r, coarse_r = scenarios[index % len(scenarios)]
        nuisance = {}
        if model_mismatch:
            nuisance = {
                "fine_sigma_g": 1.60 + 0.03 * (index % 3),
                "coarse_sigma_g": 1.80 + 0.04 * (index % 3),
                "fine_refractive_index": 1.46 + 0.01j * (index % 2),
                "coarse_refractive_index": 1.53 + 0.01j * ((index + 1) % 2),
            }
        states.append(
            AerosolState(
                fine_volume=float(fine_v * np.exp(rng.normal(0.0, 0.25))),
                coarse_volume=float(coarse_v * np.exp(rng.normal(0.0, 0.25))),
                fine_rv_m=float(fine_r * np.exp(rng.normal(0.0, 0.08))),
                coarse_rv_m=float(coarse_r * np.exp(rng.normal(0.0, 0.10))),
                scenario_id=scenario,
                truth_id=f"truth-{index:04d}",
                **nuisance,
            )
        )
    return states


def _prediction_for_log_parameters(log_parameters, template_state, wavelengths_m, radius_grid_m, arm):
    parameters = np.exp(log_parameters)
    state = AerosolState(
        fine_volume=float(parameters[0]),
        coarse_volume=float(parameters[1]),
        fine_rv_m=float(parameters[2]),
        coarse_rv_m=float(parameters[3]),
        fine_sigma_g=template_state.fine_sigma_g,
        coarse_sigma_g=template_state.coarse_sigma_g,
        fine_refractive_index=template_state.fine_refractive_index,
        coarse_refractive_index=template_state.coarse_refractive_index,
        scenario_id=template_state.scenario_id,
        truth_id=template_state.truth_id,
    )
    vector = optical_vector(compute_aerosol_optics(state, wavelengths_m, radius_grid_m))
    return vector if arm == "plus" else vector[:-1]


def _arm_a_diagnostics(state, optics, wavelengths_m, radius_grid_m, relative_errors):
    truth = optical_vector(optics)
    log_parameters = np.log([state.fine_volume, state.coarse_volume, state.fine_rv_m, state.coarse_rv_m])
    records = []
    for arm in ("base", "plus"):
        vector = truth if arm == "plus" else truth[:-1]
        relative = np.asarray(relative_errors, dtype=float)
        covariance = np.diag((vector * relative[: len(vector)]) ** 2)
        chol = np.linalg.cholesky(covariance)
        jacobian = np.empty((len(vector), 4))
        for column in range(4):
            step = 1e-4
            plus_parameters = log_parameters.copy()
            minus_parameters = log_parameters.copy()
            plus_parameters[column] += step
            minus_parameters[column] -= step
            f_plus = _prediction_for_log_parameters(plus_parameters, state, wavelengths_m, radius_grid_m, arm)
            f_minus = _prediction_for_log_parameters(minus_parameters, state, wavelengths_m, radius_grid_m, arm)
            jacobian[:, column] = np.linalg.solve(chol, (f_plus - f_minus) / (2.0 * step))
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        fisher = jacobian.T @ jacobian
        eigvals = np.linalg.eigvalsh(fisher)
        records.append({
            "truth_id": state.truth_id,
            "scenario_id": state.scenario_id,
            "arm_id": arm,
            "condition_number": float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf"),
            "min_fisher_eigenvalue": float(eigvals[0]),
            "singular_values": json.dumps([float(value) for value in singular_values]),
            "jacobian": json.dumps(jacobian.tolist()),
            "fisher_matrix": json.dumps(fisher.tolist()),
            "local_covariance": json.dumps(np.linalg.pinv(fisher, hermitian=True).tolist()),
            "finite_difference_step": 1e-4,
        })
    return records


def _state_from_parameters(parameters, template_state):
    return AerosolState(
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


def _retrieval_template_for(state: AerosolState) -> AerosolState:
    """Return nominal nuisance parameters without copying truth nuisance values."""

    return AerosolState(
        fine_volume=1.5e-12,
        coarse_volume=1.5e-12,
        fine_rv_m=0.10e-6,
        coarse_rv_m=1.0e-6,
        scenario_id=state.scenario_id,
        truth_id=state.truth_id,
    )


def _product_interval(
    result: RetrievalResult,
    product_id: str,
    radius_grid_m: np.ndarray,
    z_value: float = 1.96,
    max_interval_factor: float = 10.0,
) -> tuple[float, float, bool]:
    """Propagate local log-parameter covariance to a positive product interval."""

    if not result.converged or result.log_parameter_covariance is None:
        return float("nan"), float("nan"), False
    center_parameters = np.array([
        result.state_estimate.fine_volume,
        result.state_estimate.coarse_volume,
        result.state_estimate.fine_rv_m,
        result.state_estimate.coarse_rv_m,
    ])
    center_log = np.log(center_parameters)
    center_product = aerosol_products(result.state_estimate, radius_grid_m)[product_id]
    gradient = np.empty(4)
    step = 1e-4
    for index in range(4):
        plus = center_log.copy()
        minus = center_log.copy()
        plus[index] += step
        minus[index] -= step
        plus_product = aerosol_products(
            _state_from_parameters(np.exp(plus), result.state_estimate), radius_grid_m
        )[product_id]
        minus_product = aerosol_products(
            _state_from_parameters(np.exp(minus), result.state_estimate), radius_grid_m
        )[product_id]
        gradient[index] = (np.log(plus_product) - np.log(minus_product)) / (2.0 * step)
    covariance = np.asarray(result.log_parameter_covariance, dtype=float)
    variance = max(0.0, float(gradient @ covariance @ gradient))
    log_half_width = min(50.0, z_value * np.sqrt(variance))
    informative = bool(np.exp(log_half_width) <= max_interval_factor)
    return (
        float(center_product * np.exp(-log_half_width)),
        float(center_product * np.exp(log_half_width)),
        informative,
    )


def run_optical_smoke(
    output_dir: Path,
    seed: int = 20260820,
    n_truth: int = 5,
    replicates: int = 2,
    alpha_errors: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
    multistart: int = 4,
    radius_points: int = 192,
    radius_min_um: float = 0.01,
    radius_max_um: float = 15.0,
    coverage_target: float = 0.90,
    max_interval_factor: float = 10.0,
    common_correlation: float = 0.10,
    same_wavelength_correlation: float = 0.35,
    model_mismatch: bool = False,
) -> ExperimentResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wavelengths_m = DEFAULT_WAVELENGTHS_M.copy()
    radius_grid_m = np.geomspace(radius_min_um * 1e-6, radius_max_um * 1e-6, radius_points)
    optical_correlation = build_optical_correlation(
        common_correlation, same_wavelength_correlation
    )
    config = {
        "seed": seed,
        "n_truth": n_truth,
        "replicates": replicates,
        "alpha_errors": list(alpha_errors),
        "multistart": multistart,
        "radius_points": radius_points,
        "radius_min_um": radius_min_um,
        "radius_max_um": radius_max_um,
        "coverage_target": coverage_target,
        "max_interval_factor": max_interval_factor,
        "common_correlation": common_correlation,
        "same_wavelength_correlation": same_wavelength_correlation,
        "model_mismatch": model_mismatch,
        "wavelengths_nm": [355, 532, 1064],
        "retrieval": "bimodal_four_parameter",
    }
    config_hash = stable_config_hash(config)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
    run_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "miepython": version("miepython"),
        "config_hash": config_hash,
        "source_revision": source_revision(),
        "source_digest": source_digest(),
        "scope": "optical-level Arm A/B",
        "forward_model": "bimodal lognormal spherical Mie",
        "noise_model": "correlated multivariate lognormal relative optical error",
        "correlation_model": {
            "common_correlation": common_correlation,
            "same_wavelength_correlation": same_wavelength_correlation,
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    states = _make_truth_states(seed, n_truth, model_mismatch=model_mismatch)
    truth_rows = []
    observation_rows = []
    retrieval_records = []
    arm_a_records = []
    for truth_index, state in enumerate(states):
        retrieval_template = _retrieval_template_for(state)
        optics = compute_aerosol_optics(state, wavelengths_m, radius_grid_m)
        truth_vector = optical_vector(optics)
        truth_rows.append({
            "truth_id": state.truth_id,
            "scenario_id": state.scenario_id,
            "fine_volume": state.fine_volume,
            "coarse_volume": state.coarse_volume,
            "fine_rv_m": state.fine_rv_m,
            "coarse_rv_m": state.coarse_rv_m,
            **aerosol_products(state, radius_grid_m),
        })
        arm_a_records.extend(_arm_a_diagnostics(state, optics, wavelengths_m, radius_grid_m, np.full(6, 0.1)))
        for replicate in range(replicates):
            for alpha_error in alpha_errors:
                observation_seed = int(seed + truth_index * 100000 + replicate * 1000)
                relative_errors = np.full(6, 0.10)
                relative_errors[-1] = alpha_error
                observation = make_optical_observation(
                    optics,
                    relative_errors,
                    observation_seed,
                    add_noise=True,
                    correlation=optical_correlation,
                )
                if alpha_error == alpha_errors[0]:
                    cached_base_result = retrieve_bimodal_state(
                        observation.optical_base,
                        observation.covariance_base,
                        wavelengths_m,
                        radius_grid_m,
                        retrieval_template,
                        "base",
                        multistart=multistart,
                        seed=observation_seed,
                    )
                base_result = cached_base_result
                plus_result = retrieve_bimodal_state(
                    observation.optical_plus,
                    observation.covariance_plus,
                    wavelengths_m,
                    radius_grid_m,
                    retrieval_template,
                    "plus",
                    multistart=multistart,
                    seed=observation_seed,
                )
                common = {
                    "truth_id": state.truth_id,
                    "scenario_id": state.scenario_id,
                    "replicate_id": replicate,
                    "alpha1064_relative_error": alpha_error,
                    "seed": observation_seed,
                    "config_hash": config_hash,
                }
                observation_rows.append({
                    **common,
                    "optical_plus": json.dumps(observation.optical_plus.tolist()),
                    "optical_base": json.dumps(observation.optical_base.tolist()),
                    "covariance_plus": json.dumps(observation.covariance_plus.tolist()),
                    "covariance_base": json.dumps(observation.covariance_base.tolist()),
                    "noise_model_id": observation.noise_model_id,
                })
                truth_products = aerosol_products(state, radius_grid_m)
                base_products = aerosol_products(base_result.state_estimate, radius_grid_m)
                plus_products = aerosol_products(plus_result.state_estimate, radius_grid_m)
                for product_id in truth_products:
                    base_lower, base_upper, base_informative = _product_interval(
                        base_result, product_id, radius_grid_m,
                        max_interval_factor=max_interval_factor,
                    )
                    plus_lower, plus_upper, plus_informative = _product_interval(
                        plus_result, product_id, radius_grid_m,
                        max_interval_factor=max_interval_factor,
                    )
                    retrieval_records.append({
                        **common,
                        "estimate_base": base_products[product_id] if base_result.converged else float("nan"),
                        "estimate_plus": plus_products[product_id] if plus_result.converged else float("nan"),
                        "truth": truth_products[product_id],
                        "converged_base": base_result.converged,
                        "converged_plus": plus_result.converged,
                        "residual_base": base_result.residual_norm,
                        "residual_plus": plus_result.residual_norm,
                        "quality_flags_base": ";".join(base_result.quality_flags),
                        "quality_flags_plus": ";".join(plus_result.quality_flags),
                        "selected_start_base": base_result.selected_start,
                        "selected_start_plus": plus_result.selected_start,
                        "objective_base": base_result.objective,
                        "objective_plus": plus_result.objective,
                        "n_starts_base": base_result.n_starts,
                        "n_starts_plus": plus_result.n_starts,
                        "multiple_solutions_base": "multiple_near_optimal_solutions" in base_result.quality_flags,
                        "multiple_solutions_plus": "multiple_near_optimal_solutions" in plus_result.quality_flags,
                        "near_optimal_count_base": base_result.near_optimal_count,
                        "near_optimal_count_plus": plus_result.near_optimal_count,
                        "maximum_log_parameter_spread_base": base_result.maximum_log_parameter_spread,
                        "maximum_log_parameter_spread_plus": plus_result.maximum_log_parameter_spread,
                        "boundary_hit_base": base_result.boundary_hit,
                        "boundary_hit_plus": plus_result.boundary_hit,
                        "interval_lower_base": base_lower,
                        "interval_upper_base": base_upper,
                        "interval_lower_plus": plus_lower,
                        "interval_upper_plus": plus_upper,
                        "interval_width_base": base_upper - base_lower,
                        "interval_width_plus": plus_upper - plus_lower,
                        "interval_informative_base": base_informative,
                        "interval_informative_plus": plus_informative,
                        "covered_base": bool(base_lower <= truth_products[product_id] <= base_upper),
                        "covered_plus": bool(plus_lower <= truth_products[product_id] <= plus_upper),
                        "product_id": product_id,
                    })

    summary_records = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in retrieval_records})
    for product_id in ("Vf", "Vc", "Vf_over_Vc", "reff_total", "reff_coarse"):
        for scenario_id in ("all", *scenario_ids):
            for alpha_error in alpha_errors:
                selected = [
                    row for row in retrieval_records
                    if row["product_id"] == product_id
                    and row["alpha1064_relative_error"] == alpha_error
                    and (scenario_id == "all" or row["scenario_id"] == scenario_id)
                ]
                metrics = paired_product_metrics(
                    selected,
                    bootstrap_samples=500,
                    seed=seed + round(alpha_error * 1000) + len(product_id) + len(scenario_id),
                )
                summary_records.append({
                    "scenario_id": scenario_id,
                    "product_id": product_id,
                    "alpha1064_relative_error": alpha_error,
                    "rmse_base": metrics.rmse_base,
                    "rmse_plus": metrics.rmse_plus,
                    "gain": metrics.gain,
                    "bias_base": metrics.bias_base,
                    "bias_plus": metrics.bias_plus,
                    "bootstrap_ci_low": metrics.bootstrap_ci_low,
                    "bootstrap_ci_high": metrics.bootstrap_ci_high,
                    "failure_rate_base": metrics.failure_rate_base,
                    "failure_rate_plus": metrics.failure_rate_plus,
                    "multiple_solution_rate_base": metrics.multiple_solution_rate_base,
                    "multiple_solution_rate_plus": metrics.multiple_solution_rate_plus,
                    "coverage_base": metrics.coverage_base,
                    "coverage_plus": metrics.coverage_plus,
                    "mean_interval_width_base": metrics.mean_interval_width_base,
                    "mean_interval_width_plus": metrics.mean_interval_width_plus,
                    "informative_interval_rate_base": metrics.informative_interval_rate_base,
                    "informative_interval_rate_plus": metrics.informative_interval_rate_plus,
                    "coverage_target": coverage_target,
                    "n_total": metrics.n_total,
                    "n_paired_success": metrics.n_paired_success,
                    "config_hash": config_hash,
                })

    write_rows(output_dir / "truth.csv", truth_rows)
    write_rows(output_dir / "observations.csv", observation_rows)
    write_rows(output_dir / "retrievals.csv", retrieval_records)
    write_rows(output_dir / "summary.csv", summary_records)
    write_rows(output_dir / "arm_a.csv", arm_a_records)
    write_optical_report(output_dir, summary_records, arm_a_records, config)
    return ExperimentResult(retrieval_records, summary_records, arm_a_records)


def _signal_config_dict(
    instrument: InstrumentConfig,
    noise: SignalNoiseConfig,
    signal_levels: tuple[float, ...],
    *,
    seed: int,
    n_truth: int,
    replicates: int,
    multistart: int,
    radius_points: int,
    radius_min_um: float,
    radius_max_um: float,
    model_mismatch: bool,
) -> dict[str, object]:
    discriminator = instrument.discriminator
    return {
        "seed": seed,
        "n_truth": n_truth,
        "replicates": replicates,
        "signal_levels": list(signal_levels),
        "radius_points": radius_points,
        "radius_min_um": radius_min_um,
        "radius_max_um": radius_max_um,
        "multistart": multistart,
        "model_mismatch": model_mismatch,
        "wavelengths_nm": list(instrument.wavelengths_nm),
        "retrieval": "bimodal_four_parameter",
        "instrument": {
            "range_m": instrument.range_m,
            "bin_width_m": instrument.bin_width_m,
            "laser_energy_relative": list(instrument.laser_energy_relative),
            "system_gain": list(instrument.system_gain),
            "overlap": list(instrument.overlap),
            "background_rate": list(instrument.background_rate),
            "dead_time_s": list(instrument.dead_time_s),
            "counting_time_s": instrument.counting_time_s,
            "count_scale": instrument.count_scale,
            "discriminator": {
                "molecular_transmission": list(discriminator.molecular_transmission),
                "aerosol_transmission": list(discriminator.aerosol_transmission),
                "cross_talk": discriminator.cross_talk.tolist(),
                "frequency_offset_hz": list(discriminator.frequency_offset_hz),
                "frequency_acceptance_width_hz": discriminator.frequency_acceptance_width_hz,
            },
        },
        "noise": {
            "poisson": noise.poisson,
            "gain_relative_error": noise.gain_relative_error,
            "laser_energy_relative_error": noise.laser_energy_relative_error,
            "discriminator_relative_error": noise.discriminator_relative_error,
            "discriminator_cross_talk_relative_error": noise.discriminator_cross_talk_relative_error,
            "min_molecular_expected_counts": noise.min_molecular_expected_counts,
        },
        "raw_channel_names": list(CHANNEL_NAMES),
        "scope": "single-range-gate signal-level Arm C",
    }


def _instrument_for_signal_level(
    instrument: InstrumentConfig,
    molecular_1064_transmission_scale: float,
) -> InstrumentConfig:
    if molecular_1064_transmission_scale <= 0 or not np.isfinite(molecular_1064_transmission_scale):
        raise ValueError("molecular_1064_transmission_scale must be finite and positive")
    discriminator = instrument.discriminator
    molecular = list(discriminator.molecular_transmission)
    molecular[2] *= molecular_1064_transmission_scale
    if molecular[2] > 1.0:
        raise ValueError("signal level would make 1064 molecular transmission exceed one")
    level_discriminator = DiscriminatorConfig(
        molecular_transmission=tuple(molecular),
        aerosol_transmission=discriminator.aerosol_transmission,
        cross_talk=discriminator.cross_talk.copy(),
        frequency_offset_hz=discriminator.frequency_offset_hz,
        frequency_acceptance_width_hz=discriminator.frequency_acceptance_width_hz,
    )
    return replace(instrument, discriminator=level_discriminator)


def _signal_observation_row(
    observation: SignalObservation,
    common: dict[str, object],
    background_counts: np.ndarray,
) -> dict[str, object]:
    alpha1064_expected = float(observation.expected_counts[5])
    alpha1064_signal = max(0.0, alpha1064_expected - float(background_counts[5]))
    return {
        **common,
        "raw_signal": json.dumps(observation.raw_signal.tolist()),
        "raw_counts": json.dumps(observation.raw_counts.tolist()),
        "expected_counts": json.dumps(observation.expected_counts.tolist()),
        "background_counts": json.dumps(np.asarray(background_counts, dtype=float).tolist()),
        "gain_multiplier": json.dumps(observation.gain_multiplier.tolist()),
        "laser_energy_multiplier": json.dumps(
            observation.laser_energy_multiplier.tolist()
        ),
        "discriminator_multiplier": json.dumps(
            observation.discriminator_multiplier.tolist()
        ),
        "cross_talk_multiplier": json.dumps(
            observation.cross_talk_multiplier.tolist()
        ),
        "optical_base": json.dumps(observation.optical_base.tolist()),
        "optical_plus": json.dumps(observation.optical_plus.tolist()),
        "truth_plus": json.dumps(observation.truth_plus.tolist()),
        "covariance_counts": json.dumps(observation.covariance_counts.tolist()),
        "covariance_base": json.dumps(observation.covariance_base.tolist()),
        "covariance_plus": json.dumps(observation.covariance_plus.tolist()),
        "channel_names": "|".join(CHANNEL_NAMES),
        "alpha1064_molecular_expected_counts": alpha1064_expected,
        "alpha1064_molecular_snr": float(alpha1064_signal / np.sqrt(max(alpha1064_expected, 1.0))),
        "noise_model_id": observation.noise_model_id,
        "quality_flags": ";".join(observation.quality_flags),
        "seed": observation.seed,
    }


def _signal_retrieval_row(
    result: RetrievalResult,
    arm: str,
    product_id: str,
    truth_value: float,
    truth_products: dict[str, float],
    radius_grid_m: np.ndarray,
    common: dict[str, object],
    max_interval_factor: float,
) -> dict[str, object]:
    lower, upper, informative = _product_interval(
        result,
        product_id,
        radius_grid_m,
        max_interval_factor=max_interval_factor,
    )
    estimate = (
        aerosol_products(result.state_estimate, radius_grid_m)[product_id]
        if result.converged
        else float("nan")
    )
    return {
        **common,
        "arm_id": arm,
        "product_id": product_id,
        "estimate": estimate,
        "truth": truth_value,
        "error": float(estimate - truth_value) if np.isfinite(estimate) else float("nan"),
        "converged": result.converged,
        "residual_norm": result.residual_norm,
        "objective": result.objective,
        "quality_flags": ";".join(result.quality_flags),
        "selected_start": result.selected_start,
        "n_starts": result.n_starts,
        "multiple_solutions": "multiple_near_optimal_solutions" in result.quality_flags,
        "near_optimal_count": result.near_optimal_count,
        "boundary_hit": result.boundary_hit,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_width": upper - lower,
        "interval_informative": informative,
        "covered": bool(lower <= truth_value <= upper),
    }


def run_signal_mc(
    output_dir: Path,
    seed: int = 20260821,
    n_truth: int = 5,
    replicates: int = 2,
    signal_levels: tuple[float, ...] = (0.5, 1.0, 2.0),
    multistart: int = 4,
    radius_points: int = 192,
    radius_min_um: float = 0.01,
    radius_max_um: float = 15.0,
    poisson: bool = True,
    gain_relative_error: float = 0.02,
    laser_energy_relative_error: float = 0.01,
    discriminator_relative_error: float = 0.02,
    discriminator_cross_talk_relative_error: float = 0.02,
    min_molecular_expected_counts: float = 20.0,
    model_mismatch: bool = True,
    coverage_target: float = 0.90,
    max_interval_factor: float = 10.0,
    instrument: InstrumentConfig | None = None,
) -> SignalExperimentResult:
    """Run paired raw-signal Arm C at several 1064 molecular signal levels."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not signal_levels or any(level <= 0 or not np.isfinite(level) for level in signal_levels):
        raise ValueError("signal_levels must contain finite positive values")
    instrument = instrument or InstrumentConfig()
    noise = SignalNoiseConfig(
        poisson=poisson,
        gain_relative_error=gain_relative_error,
        laser_energy_relative_error=laser_energy_relative_error,
        discriminator_relative_error=discriminator_relative_error,
        discriminator_cross_talk_relative_error=discriminator_cross_talk_relative_error,
        min_molecular_expected_counts=min_molecular_expected_counts,
    )
    signal_levels = tuple(float(level) for level in signal_levels)
    config = _signal_config_dict(
        instrument,
        noise,
        signal_levels,
        seed=seed,
        n_truth=n_truth,
        replicates=replicates,
        multistart=multistart,
        radius_points=radius_points,
        radius_min_um=radius_min_um,
        radius_max_um=radius_max_um,
        model_mismatch=model_mismatch,
    )
    config["coverage_target"] = coverage_target
    config["max_interval_factor"] = max_interval_factor
    config_hash = stable_config_hash(config)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "miepython": version("miepython"),
        "config_hash": config_hash,
        "source_revision": source_revision(),
        "source_digest": source_digest(),
        "scope": "single-range-gate signal-level Arm C",
        "forward_model": "bimodal lognormal spherical Mie plus Rayleigh and parameterized discriminator",
        "noise_model": "Poisson counts plus lognormal gain, laser-energy and discriminator perturbations",
        "raw_channel_names": list(CHANNEL_NAMES),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    wavelengths_m = DEFAULT_WAVELENGTHS_M.copy()
    radius_grid_m = np.geomspace(radius_min_um * 1e-6, radius_max_um * 1e-6, radius_points)
    states = _make_truth_states(seed, n_truth, model_mismatch=model_mismatch)
    truth_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    retrieval_records: list[dict[str, object]] = []
    paired_records: list[dict[str, object]] = []
    for truth_index, state in enumerate(states):
        optics = compute_aerosol_optics(state, wavelengths_m, radius_grid_m)
        truth_products = aerosol_products(state, radius_grid_m)
        truth_rows.append({
            "truth_id": state.truth_id,
            "scenario_id": state.scenario_id,
            "fine_volume": state.fine_volume,
            "coarse_volume": state.coarse_volume,
            "fine_rv_m": state.fine_rv_m,
            "coarse_rv_m": state.coarse_rv_m,
            **truth_products,
        })
        retrieval_template = _retrieval_template_for(state)
        for replicate in range(replicates):
            for level_index, signal_level in enumerate(signal_levels):
                level_instrument = _instrument_for_signal_level(instrument, signal_level)
                observation_seed = int(seed + truth_index * 100000 + replicate * 1000 + level_index)
                observation = simulate_signal_observation(
                    optics,
                    level_instrument,
                    seed=observation_seed,
                    noise=noise,
                )
                observation_id = f"{state.truth_id}-rep{replicate:03d}-level{level_index:03d}"
                common = {
                    "observation_id": observation_id,
                    "truth_id": state.truth_id,
                    "scenario_id": state.scenario_id,
                    "replicate_id": replicate,
                    "molecular_1064_transmission_scale": signal_level,
                    "seed": observation_seed,
                    "config_hash": config_hash,
                    "raw_observation_shared": True,
                }
                observation_rows.append(
                    _signal_observation_row(
                        observation,
                        common,
                        np.asarray(level_instrument.background_rate)
                        * level_instrument.counting_time_s,
                    )
                )
                base_result = retrieve_bimodal_state(
                    observation.optical_base,
                    observation.covariance_base,
                    wavelengths_m,
                    radius_grid_m,
                    retrieval_template,
                    "base",
                    multistart=multistart,
                    seed=observation_seed,
                )
                plus_result = retrieve_bimodal_state(
                    observation.optical_plus,
                    observation.covariance_plus,
                    wavelengths_m,
                    radius_grid_m,
                    retrieval_template,
                    "plus",
                    multistart=multistart,
                    seed=observation_seed,
                )
                paired_common = {
                    **common,
                    "arm_id": "paired_base_plus",
                    "observation_quality_flags": ";".join(observation.quality_flags),
                }
                for product_id, truth_value in truth_products.items():
                    base_row = _signal_retrieval_row(
                        base_result, "base", product_id, truth_value, truth_products,
                        radius_grid_m, common, max_interval_factor,
                    )
                    plus_row = _signal_retrieval_row(
                        plus_result, "plus", product_id, truth_value, truth_products,
                        radius_grid_m, common, max_interval_factor,
                    )
                    retrieval_records.extend((base_row, plus_row))
                    base_estimate = base_row["estimate"]
                    plus_estimate = plus_row["estimate"]
                    paired_records.append({
                        **paired_common,
                        "product_id": product_id,
                        "estimate_base": base_estimate,
                        "estimate_plus": plus_estimate,
                        "truth": truth_value,
                        "converged_base": base_result.converged,
                        "converged_plus": plus_result.converged,
                        "residual_base": base_result.residual_norm,
                        "residual_plus": plus_result.residual_norm,
                        "quality_flags_base": ";".join(base_result.quality_flags),
                        "quality_flags_plus": ";".join(plus_result.quality_flags),
                        "multiple_solutions_base": base_row["multiple_solutions"],
                        "multiple_solutions_plus": plus_row["multiple_solutions"],
                        "interval_lower_base": base_row["interval_lower"],
                        "interval_upper_base": base_row["interval_upper"],
                        "interval_lower_plus": plus_row["interval_lower"],
                        "interval_upper_plus": plus_row["interval_upper"],
                        "interval_width_base": base_row["interval_width"],
                        "interval_width_plus": plus_row["interval_width"],
                        "interval_informative_base": base_row["interval_informative"],
                        "interval_informative_plus": plus_row["interval_informative"],
                        "covered_base": base_row["covered"],
                        "covered_plus": plus_row["covered"],
                    })

    summary_records: list[dict[str, object]] = []
    scenario_ids = sorted({str(row["scenario_id"]) for row in paired_records})
    for product_id in ("Vf", "Vc", "Vf_over_Vc", "reff_total", "reff_coarse"):
        for scenario_id in ("all", *scenario_ids):
            for signal_level in signal_levels:
                selected = [
                    row for row in paired_records
                    if row["product_id"] == product_id
                    and float(row["molecular_1064_transmission_scale"]) == signal_level
                    and (scenario_id == "all" or row["scenario_id"] == scenario_id)
                ]
                metrics = paired_product_metrics(
                    selected,
                    bootstrap_samples=500,
                    seed=seed + int(signal_level * 1000) + len(product_id) + len(scenario_id),
                )
                summary_records.append({
                    "scenario_id": scenario_id,
                    "product_id": product_id,
                    "molecular_1064_transmission_scale": signal_level,
                    "rmse_base": metrics.rmse_base,
                    "rmse_plus": metrics.rmse_plus,
                    "gain": metrics.gain,
                    "bias_base": metrics.bias_base,
                    "bias_plus": metrics.bias_plus,
                    "bootstrap_ci_low": metrics.bootstrap_ci_low,
                    "bootstrap_ci_high": metrics.bootstrap_ci_high,
                    "failure_rate_base": metrics.failure_rate_base,
                    "failure_rate_plus": metrics.failure_rate_plus,
                    "multiple_solution_rate_base": metrics.multiple_solution_rate_base,
                    "multiple_solution_rate_plus": metrics.multiple_solution_rate_plus,
                    "coverage_base": metrics.coverage_base,
                    "coverage_plus": metrics.coverage_plus,
                    "mean_interval_width_base": metrics.mean_interval_width_base,
                    "mean_interval_width_plus": metrics.mean_interval_width_plus,
                    "informative_interval_rate_base": metrics.informative_interval_rate_base,
                    "informative_interval_rate_plus": metrics.informative_interval_rate_plus,
                    "coverage_target": coverage_target,
                    "n_total": metrics.n_total,
                    "n_paired_success": metrics.n_paired_success,
                    "config_hash": config_hash,
                })

    write_rows(output_dir / "truth.csv", truth_rows)
    write_rows(output_dir / "observations.csv", observation_rows)
    write_rows(output_dir / "retrievals.csv", retrieval_records)
    write_rows(output_dir / "paired_retrievals.csv", paired_records)
    write_rows(output_dir / "summary.csv", summary_records)
    write_signal_report(output_dir, summary_records, observation_rows, config)
    return SignalExperimentResult(retrieval_records, paired_records, summary_records)
