from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import miepython
import numpy as np

from .distributions import modal_volume_distribution
from .molecular import molecular_backscatter_m_inv_sr_inv, molecular_extinction_m_inv
from .schemas import AerosolState, OpticalProperties


DEFAULT_WAVELENGTHS_M = np.array([355e-9, 532e-9, 1064e-9], dtype=float)


def differential_backscatter_from_qback(radius_m: float, qback: float) -> float:
    """Convert miepython Qback to d sigma/d Omega at 180 degrees.

    miepython defines Qback as the total backscatter cross-section divided by
    pi*r^2. The lidar differential cross-section is therefore r^2*Qback/4,
    with units m^2 sr^-1.
    """

    if radius_m <= 0 or qback < 0:
        raise ValueError("radius must be positive and qback must be non-negative")
    return float(radius_m**2 * qback / 4.0)


@lru_cache(maxsize=32)
def _cached_efficiency_table(
    refractive_index_real: float,
    refractive_index_imag: float,
    wavelengths_key: tuple[float, ...],
    radius_key: tuple[float, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Cache vectorized Mie efficiencies for a fixed optical grid and index."""

    refractive_index = complex(refractive_index_real, refractive_index_imag)
    radius = np.asarray(radius_key, dtype=float)
    qext_table = []
    qback_table = []
    for wavelength_m in wavelengths_key:
        x = 2.0 * np.pi * radius / wavelength_m
        qext, _, qback, _ = miepython.efficiencies_mx(refractive_index, x)
        qext_table.append(np.asarray(qext, dtype=float))
        qback_table.append(np.asarray(qback, dtype=float))
    return tuple(qext_table), tuple(qback_table)


def _integrate_modal_optics(
    volume_distribution: np.ndarray,
    radius: np.ndarray,
    qext: np.ndarray,
    qback: np.ndarray,
) -> tuple[float, float]:
    number_distribution = volume_distribution / ((4.0 / 3.0) * np.pi * radius**3)
    alpha_integrand = number_distribution * np.pi * radius**2 * qext
    beta_integrand = number_distribution * radius**2 * qback / 4.0
    return float(np.trapezoid(alpha_integrand, radius)), float(np.trapezoid(beta_integrand, radius))


def compute_aerosol_optics(
    state: AerosolState,
    wavelengths_m: Iterable[float],
    radius_grid_m: np.ndarray,
) -> OpticalProperties:
    """Compute aerosol and molecular optics for the requested wavelengths."""

    wavelengths = np.asarray(tuple(wavelengths_m), dtype=float)
    radius = np.asarray(radius_grid_m, dtype=float)
    if wavelengths.ndim != 1 or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_m must be a one-dimensional positive sequence")
    if radius.ndim != 1 or np.any(radius <= 0) or np.any(np.diff(radius) <= 0):
        raise ValueError("radius_grid_m must be strictly increasing and positive")

    fine_volume_distribution = modal_volume_distribution(
        radius, state.fine_volume, state.fine_rv_m, state.fine_sigma_g
    )
    coarse_volume_distribution = modal_volume_distribution(
        radius, state.coarse_volume, state.coarse_rv_m, state.coarse_sigma_g
    )
    fine_qext, fine_qback = _cached_efficiency_table(
        state.fine_refractive_index.real,
        state.fine_refractive_index.imag,
        tuple(float(value) for value in wavelengths),
        tuple(float(value) for value in radius),
    )
    coarse_qext, coarse_qback = _cached_efficiency_table(
        state.coarse_refractive_index.real,
        state.coarse_refractive_index.imag,
        tuple(float(value) for value in wavelengths),
        tuple(float(value) for value in radius),
    )
    aerosol = []
    for fine_ext, fine_back, coarse_ext, coarse_back in zip(
        fine_qext, fine_qback, coarse_qext, coarse_qback
    ):
        fine_alpha, fine_beta = _integrate_modal_optics(
            fine_volume_distribution, radius, fine_ext, fine_back
        )
        coarse_alpha, coarse_beta = _integrate_modal_optics(
            coarse_volume_distribution, radius, coarse_ext, coarse_back
        )
        aerosol.append((fine_alpha + coarse_alpha, fine_beta + coarse_beta))
    alpha_aerosol = tuple(value[0] for value in aerosol)
    beta_aerosol = tuple(value[1] for value in aerosol)
    alpha_molecular = tuple(molecular_extinction_m_inv(wavelength) for wavelength in wavelengths)
    beta_molecular = tuple(
        molecular_backscatter_m_inv_sr_inv(wavelength) for wavelength in wavelengths
    )
    lidar_ratio = tuple(alpha / beta for alpha, beta in zip(alpha_aerosol, beta_aerosol))

    return OpticalProperties(
        wavelengths_m=tuple(wavelengths),
        alpha_aerosol_m_inv=alpha_aerosol,
        beta_aerosol_m_inv_sr_inv=beta_aerosol,
        alpha_molecular_m_inv=alpha_molecular,
        beta_molecular_m_inv_sr_inv=beta_molecular,
        lidar_ratio=lidar_ratio,
    )
