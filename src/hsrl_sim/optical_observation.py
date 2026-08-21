from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import OpticalProperties


@dataclass(frozen=True)
class OpticalObservation:
    """Paired optical observations for the base and enhanced arms."""

    optical_plus: np.ndarray
    optical_base: np.ndarray
    covariance_plus: np.ndarray
    covariance_base: np.ndarray
    truth_plus: np.ndarray
    seed: int
    noise_model_id: str
    quality_flags: tuple[str, ...] = ()


def optical_vector(optics: OpticalProperties) -> np.ndarray:
    """Return [beta355, beta532, beta1064, alpha355, alpha532, alpha1064]."""

    return np.asarray(
        [*optics.beta_aerosol_m_inv_sr_inv, *optics.alpha_aerosol_m_inv],
        dtype=float,
    )


def _relative_error_vector(relative_error: float | np.ndarray) -> np.ndarray:
    values = np.asarray(relative_error, dtype=float)
    if values.ndim == 0:
        values = np.full(6, float(values))
    if values.shape != (6,) or np.any(values <= 0) or np.any(~np.isfinite(values)):
        raise ValueError("relative_error must contain six finite positive values")
    return values


def lognormal_sigma_for_relative_error(relative_error: float | np.ndarray) -> np.ndarray:
    """Map a coefficient of variation to its mean-preserving lognormal sigma."""

    relative = np.asarray(relative_error, dtype=float)
    if np.any(relative <= 0) or np.any(~np.isfinite(relative)):
        raise ValueError("relative_error must be finite and positive")
    return np.sqrt(np.log1p(relative**2))


def make_optical_observation(
    optics: OpticalProperties,
    relative_error: float | np.ndarray,
    seed: int,
    add_noise: bool = True,
    correlation: np.ndarray | None = None,
) -> OpticalObservation:
    """Sample a positive optical observation with a declared full covariance."""

    truth = optical_vector(optics)
    if truth.shape != (6,) or np.any(truth <= 0) or np.any(~np.isfinite(truth)):
        raise ValueError("optical truth vector must contain six finite positive values")
    relative = _relative_error_vector(relative_error)
    if correlation is None:
        correlation = np.eye(6)
    correlation = np.asarray(correlation, dtype=float)
    if correlation.shape != (6, 6) or np.any(~np.isfinite(correlation)):
        raise ValueError("correlation must be a finite 6x6 matrix")
    correlation = 0.5 * (correlation + correlation.T)
    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("correlation diagonal must equal one")
    if np.min(np.linalg.eigvalsh(correlation)) < -1e-10:
        raise ValueError("correlation must be positive semidefinite")
    standard_deviation = truth * relative
    covariance = np.outer(standard_deviation, standard_deviation) * correlation
    rng = np.random.default_rng(seed)
    if add_noise:
        log_covariance = np.log1p(correlation * np.outer(relative, relative))
        log_covariance = 0.5 * (log_covariance + log_covariance.T)
        try:
            log_cholesky = np.linalg.cholesky(log_covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "correlation and relative errors do not define a valid multivariate lognormal"
            ) from error
        standard_normal = rng.standard_normal(6)
        log_perturbation = (
            log_cholesky @ standard_normal - 0.5 * np.diag(log_covariance)
        )
        sample = truth * np.exp(log_perturbation)
    else:
        sample = truth.copy()

    return OpticalObservation(
        optical_plus=sample,
        optical_base=sample[:-1].copy(),
        covariance_plus=covariance,
        covariance_base=covariance[:-1, :-1].copy(),
        truth_plus=truth,
        seed=seed,
        noise_model_id="multivariate_lognormal_relative" if add_noise else "none",
    )
