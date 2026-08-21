from __future__ import annotations

import numpy as np

from .schemas import AerosolState


def modal_volume_distribution(
    radius_grid_m: np.ndarray,
    total_volume: float,
    median_radius_m: float,
    sigma_g: float,
) -> np.ndarray:
    """Return dV/d r for a lognormal mode, normalized to ``total_volume``."""

    radius = np.asarray(radius_grid_m, dtype=float)
    if np.any(radius <= 0) or np.any(np.diff(radius) <= 0):
        raise ValueError("radius_grid_m must be strictly increasing and positive")
    if total_volume <= 0 or median_radius_m <= 0 or sigma_g <= 1.0:
        raise ValueError("volume, median radius and sigma_g must be positive; sigma_g > 1")

    log_sigma = np.log(sigma_g)
    log_radius = np.log(radius / median_radius_m)
    volume_per_log_radius = total_volume * np.exp(
        -(log_radius**2) / (2.0 * log_sigma**2)
    ) / (np.sqrt(2.0 * np.pi) * log_sigma)
    return volume_per_log_radius / radius


def bimodal_volume_distribution(
    state: AerosolState, radius_grid_m: np.ndarray
) -> np.ndarray:
    """Return the sum of fine and coarse dV/d r distributions."""

    return modal_volume_distribution(
        radius_grid_m, state.fine_volume, state.fine_rv_m, state.fine_sigma_g
    ) + modal_volume_distribution(
        radius_grid_m, state.coarse_volume, state.coarse_rv_m, state.coarse_sigma_g
    )


def effective_radius_m(radius_grid_m: np.ndarray, volume_distribution: np.ndarray) -> float:
    """Compute effective radius from a volume-per-radius distribution."""

    radius = np.asarray(radius_grid_m, dtype=float)
    distribution = np.asarray(volume_distribution, dtype=float)
    if radius.shape != distribution.shape or np.any(radius <= 0):
        raise ValueError("radius and volume distribution must have matching positive grids")
    total_volume = np.trapezoid(distribution, radius)
    area_weighted_denominator = np.trapezoid(distribution / radius, radius)
    if total_volume <= 0 or area_weighted_denominator <= 0:
        raise ValueError("volume distribution must have positive volume and area moments")
    return float(total_volume / area_weighted_denominator)


def aerosol_products(state: AerosolState, radius_grid_m: np.ndarray) -> dict[str, float]:
    """Return the registered low-dimensional products for one aerosol state."""

    fine_distribution = modal_volume_distribution(
        radius_grid_m, state.fine_volume, state.fine_rv_m, state.fine_sigma_g
    )
    coarse_distribution = modal_volume_distribution(
        radius_grid_m, state.coarse_volume, state.coarse_rv_m, state.coarse_sigma_g
    )
    total_distribution = fine_distribution + coarse_distribution
    fine_volume = float(state.fine_volume)
    coarse_volume = float(state.coarse_volume)
    return {
        "Vf": fine_volume,
        "Vc": coarse_volume,
        "Vf_over_Vc": float(fine_volume / coarse_volume),
        "reff_total": effective_radius_m(radius_grid_m, total_distribution),
        "reff_coarse": effective_radius_m(radius_grid_m, coarse_distribution),
    }
