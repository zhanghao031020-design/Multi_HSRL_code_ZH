from __future__ import annotations

import numpy as np


REFERENCE_WAVELENGTH_M = 355e-9
REFERENCE_MOLECULAR_BETA_M_INV_SR_INV = 2.5e-6
RAYLEIGH_ALPHA_TO_BETA = 8.0 * np.pi / 3.0


def molecular_backscatter_m_inv_sr_inv(
    wavelength_m: float,
    reference_beta_m_inv_sr_inv: float = REFERENCE_MOLECULAR_BETA_M_INV_SR_INV,
) -> float:
    """Return a standard-atmosphere Rayleigh backscatter reference.

    This first-stage model uses a fixed molecular number-density scale. The
    wavelength dependence is the Rayleigh lambda^-4 law; pressure and
    temperature profiles are explicit extension points for the profile stage.
    """

    if wavelength_m <= 0 or reference_beta_m_inv_sr_inv <= 0:
        raise ValueError("wavelength and reference beta must be positive")
    return float(reference_beta_m_inv_sr_inv * (REFERENCE_WAVELENGTH_M / wavelength_m) ** 4)


def molecular_extinction_m_inv(wavelength_m: float) -> float:
    """Return the molecular extinction coefficient in m^-1."""

    return float(RAYLEIGH_ALPHA_TO_BETA * molecular_backscatter_m_inv_sr_inv(wavelength_m))
