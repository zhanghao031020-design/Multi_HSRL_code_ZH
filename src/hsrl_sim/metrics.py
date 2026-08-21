from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class PairedMetrics:
    rmse_base: float
    rmse_plus: float
    gain: float
    bias_base: float
    bias_plus: float
    failure_rate_base: float
    failure_rate_plus: float
    n_total: int
    n_paired_success: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    multiple_solution_rate_base: float
    multiple_solution_rate_plus: float
    coverage_base: float
    coverage_plus: float
    mean_interval_width_base: float
    mean_interval_width_plus: float
    informative_interval_rate_base: float
    informative_interval_rate_plus: float


def paired_gain(estimate_base, estimate_plus, truth) -> float:
    """Return paired fractional RMSE reduction from base to plus."""

    base = np.asarray(estimate_base, dtype=float)
    plus = np.asarray(estimate_plus, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if base.shape != plus.shape or base.shape != truth.shape or base.size == 0:
        raise ValueError("paired estimates and truth must have equal non-empty shapes")
    rmse_base = float(np.sqrt(np.mean((base - truth) ** 2)))
    rmse_plus = float(np.sqrt(np.mean((plus - truth) ** 2)))
    if rmse_base == 0:
        return 0.0 if rmse_plus == 0 else float("-inf")
    return float((rmse_base - rmse_plus) / rmse_base)


def paired_product_metrics(
    records: Iterable[Mapping[str, object]],
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> PairedMetrics:
    """Summarize paired records without hiding failures."""

    rows = list(records)
    if not rows:
        raise ValueError("records must not be empty")
    success = [
        row for row in rows
        if bool(row["converged_base"])
        and bool(row["converged_plus"])
        and np.isfinite(float(row["estimate_base"]))
        and np.isfinite(float(row["estimate_plus"]))
    ]
    failure_rate_base = float(np.mean([not bool(row["converged_base"]) for row in rows]))
    failure_rate_plus = float(np.mean([not bool(row["converged_plus"]) for row in rows]))
    multiple_solution_rate_base = float(
        np.mean([bool(row.get("multiple_solutions_base", False)) for row in rows])
    )
    multiple_solution_rate_plus = float(
        np.mean([bool(row.get("multiple_solutions_plus", False)) for row in rows])
    )
    base_success = [row for row in rows if bool(row["converged_base"])]
    plus_success = [row for row in rows if bool(row["converged_plus"])]
    coverage_base = float(np.mean([bool(row.get("covered_base", False)) for row in base_success])) if base_success else float("nan")
    coverage_plus = float(np.mean([bool(row.get("covered_plus", False)) for row in plus_success])) if plus_success else float("nan")
    base_widths = [float(row.get("interval_width_base", float("nan"))) for row in base_success]
    plus_widths = [float(row.get("interval_width_plus", float("nan"))) for row in plus_success]
    base_widths = [value for value in base_widths if np.isfinite(value)]
    plus_widths = [value for value in plus_widths if np.isfinite(value)]
    mean_interval_width_base = float(np.mean(base_widths)) if base_widths else float("nan")
    mean_interval_width_plus = float(np.mean(plus_widths)) if plus_widths else float("nan")
    informative_interval_rate_base = float(
        np.mean([bool(row.get("interval_informative_base", False)) for row in rows])
    )
    informative_interval_rate_plus = float(
        np.mean([bool(row.get("interval_informative_plus", False)) for row in rows])
    )
    if not success:
        return PairedMetrics(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), failure_rate_base, failure_rate_plus, len(rows), 0, float("nan"), float("nan"), multiple_solution_rate_base, multiple_solution_rate_plus, coverage_base, coverage_plus, mean_interval_width_base, mean_interval_width_plus, informative_interval_rate_base, informative_interval_rate_plus)

    base = np.asarray([float(row["estimate_base"]) for row in success])
    plus = np.asarray([float(row["estimate_plus"]) for row in success])
    truth = np.asarray([float(row["truth"]) for row in success])
    rmse_base = float(np.sqrt(np.mean((base - truth) ** 2)))
    rmse_plus = float(np.sqrt(np.mean((plus - truth) ** 2)))
    gain = paired_gain(base, plus, truth)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in success:
        grouped.setdefault(str(row["truth_id"]), []).append(row)
    truth_ids = tuple(grouped)
    rng = np.random.default_rng(seed)
    bootstrap_gains = []
    for _ in range(bootstrap_samples):
        selected_ids = rng.choice(truth_ids, size=len(truth_ids), replace=True)
        sampled = [row for truth_id in selected_ids for row in grouped[str(truth_id)]]
        bootstrap_gains.append(paired_gain([float(row["estimate_base"]) for row in sampled], [float(row["estimate_plus"]) for row in sampled], [float(row["truth"]) for row in sampled]))
    finite_gains = np.asarray(bootstrap_gains, dtype=float)
    finite_gains = finite_gains[np.isfinite(finite_gains)]
    ci_low, ci_high = (np.quantile(finite_gains, [0.025, 0.975]) if finite_gains.size else (float("nan"), float("nan")))
    return PairedMetrics(rmse_base, rmse_plus, gain, float(np.mean(base - truth)), float(np.mean(plus - truth)), failure_rate_base, failure_rate_plus, len(rows), len(success), float(ci_low), float(ci_high), multiple_solution_rate_base, multiple_solution_rate_plus, coverage_base, coverage_plus, mean_interval_width_base, mean_interval_width_plus, informative_interval_rate_base, informative_interval_rate_plus)
