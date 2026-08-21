from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .discriminator import DiscriminatorConfig, channel_response_matrix
from .molecular import molecular_backscatter_m_inv_sr_inv, molecular_extinction_m_inv
from .optical_observation import optical_vector
from .schemas import OpticalProperties


CHANNEL_NAMES = (
    "mixed_355",
    "molecular_355",
    "mixed_532",
    "molecular_532",
    "mixed_1064",
    "molecular_1064",
)
REFERENCE_GATE_WIDTH_M = 30.0


def _vector(value: float | tuple[float, ...] | np.ndarray, length: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(length, float(values))
    if values.shape != (length,) or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar or length-{length} vector")
    return values


@dataclass(frozen=True)
class InstrumentConfig:
    """Single-range-gate instrument and count-scaling configuration."""

    wavelengths_nm: tuple[int, ...] = (355, 532, 1064)
    range_m: float = 1000.0
    bin_width_m: float = 30.0
    laser_energy_relative: tuple[float, ...] = (1.0, 1.0, 1.0)
    system_gain: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    overlap: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    background_rate: tuple[float, ...] = (200.0, 100.0, 120.0, 60.0, 80.0, 40.0)
    dead_time_s: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    counting_time_s: float = 0.02
    count_scale: float = 3.0e10
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)

    def __post_init__(self) -> None:
        if tuple(self.wavelengths_nm) != (355, 532, 1064):
            raise ValueError("Arm C currently requires wavelengths_nm=(355, 532, 1064)")
        if self.range_m <= 0 or self.bin_width_m <= 0 or self.counting_time_s <= 0:
            raise ValueError("range, bin width and counting time must be positive")
        if self.count_scale <= 0 or not np.isfinite(self.count_scale):
            raise ValueError("count_scale must be finite and positive")
        energy = _vector(self.laser_energy_relative, 3, "laser_energy_relative")
        gain = _vector(self.system_gain, 6, "system_gain")
        overlap = _vector(self.overlap, 6, "overlap")
        background = _vector(self.background_rate, 6, "background_rate")
        dead_time = _vector(self.dead_time_s, 6, "dead_time_s")
        if np.any(energy <= 0) or np.any(gain <= 0):
            raise ValueError("laser energy and system gain must be positive")
        if np.any((overlap < 0) | (overlap > 1)):
            raise ValueError("overlap must be in [0, 1]")
        if np.any(background < 0) or np.any(dead_time < 0):
            raise ValueError("background and dead time must be nonnegative")
        object.__setattr__(self, "laser_energy_relative", tuple(float(v) for v in energy))
        object.__setattr__(self, "system_gain", tuple(float(v) for v in gain))
        object.__setattr__(self, "overlap", tuple(float(v) for v in overlap))
        object.__setattr__(self, "background_rate", tuple(float(v) for v in background))
        object.__setattr__(self, "dead_time_s", tuple(float(v) for v in dead_time))


@dataclass(frozen=True)
class SignalNoiseConfig:
    """Noise layers applied after deterministic signal forward modelling."""

    poisson: bool = True
    gain_relative_error: float | tuple[float, ...] = 0.0
    laser_energy_relative_error: float | tuple[float, ...] = 0.0
    discriminator_relative_error: float | tuple[float, ...] = 0.0
    discriminator_cross_talk_relative_error: float | tuple[float, ...] = 0.0
    min_molecular_expected_counts: float = 20.0


@dataclass(frozen=True)
class SignalExpectation:
    signal_counts: np.ndarray
    background_counts: np.ndarray
    expected_counts: np.ndarray
    attenuation: np.ndarray
    response_matrix: np.ndarray


@dataclass(frozen=True)
class SignalObservation:
    """Raw six-channel data and its single-gate optical transformation."""

    raw_signal: np.ndarray
    raw_counts: np.ndarray
    expected_counts: np.ndarray
    optical_base: np.ndarray
    optical_plus: np.ndarray
    covariance_base: np.ndarray
    covariance_plus: np.ndarray
    truth_plus: np.ndarray
    covariance_counts: np.ndarray
    gain_multiplier: np.ndarray
    laser_energy_multiplier: np.ndarray
    discriminator_multiplier: np.ndarray
    cross_talk_multiplier: np.ndarray
    seed: int
    noise_model_id: str
    quality_flags: tuple[str, ...] = ()


def _validate_optics(optics: OpticalProperties, instrument: InstrumentConfig) -> None:
    wavelengths_m = np.asarray(optics.wavelengths_m, dtype=float)
    expected_wavelengths_m = np.asarray(instrument.wavelengths_nm, dtype=float) * 1e-9
    if wavelengths_m.shape != (3,) or not np.allclose(wavelengths_m, expected_wavelengths_m):
        raise ValueError("optics wavelengths must match instrument wavelengths 355/532/1064 nm")
    arrays = (
        optics.alpha_aerosol_m_inv,
        optics.beta_aerosol_m_inv_sr_inv,
        optics.alpha_molecular_m_inv,
        optics.beta_molecular_m_inv_sr_inv,
    )
    if any(len(values) != 3 for values in arrays):
        raise ValueError("optical properties must contain three wavelengths")
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError("optical properties must be finite")


def _dead_time_forward(pre_dead_time_counts: np.ndarray, instrument: InstrumentConfig) -> np.ndarray:
    dead_time = np.asarray(instrument.dead_time_s, dtype=float)
    return pre_dead_time_counts / (1.0 + pre_dead_time_counts * dead_time / instrument.counting_time_s)


def expected_signal_counts(
    optics: OpticalProperties,
    instrument: InstrumentConfig,
    *,
    laser_energy_multiplier: np.ndarray | None = None,
    gain_multiplier: np.ndarray | None = None,
    discriminator_multiplier: np.ndarray | None = None,
    cross_talk_multiplier: np.ndarray | None = None,
) -> SignalExpectation:
    """Compute deterministic six-channel expected counts before sampling.

    The two-way attenuation is applied to both aerosol and molecular return
    within the range gate.  ``count_scale`` carries telescope/receiver
    constants; ``bin_width_m`` keeps gate-size sensitivity explicit.
    """

    _validate_optics(optics, instrument)
    energy = np.asarray(instrument.laser_energy_relative, dtype=float)
    gain = np.asarray(instrument.system_gain, dtype=float)
    overlap = np.asarray(instrument.overlap, dtype=float)
    if laser_energy_multiplier is not None:
        energy = energy * _vector(laser_energy_multiplier, 3, "laser_energy_multiplier")
    if gain_multiplier is not None:
        gain = gain * _vector(gain_multiplier, 6, "gain_multiplier")
    if discriminator_multiplier is not None:
        discriminator_scale = _vector(
            discriminator_multiplier, 3, "discriminator_multiplier"
        )
    else:
        discriminator_scale = np.ones(3)
    if cross_talk_multiplier is None:
        cross_talk = None
    else:
        cross_talk_scale = np.asarray(cross_talk_multiplier, dtype=float)
        if cross_talk_scale.shape != (3, 2) or np.any(~np.isfinite(cross_talk_scale)) or np.any(cross_talk_scale < 0.0):
            raise ValueError("cross_talk_multiplier must be a finite nonnegative (3, 2) array")
        cross_talk = instrument.discriminator.cross_talk * cross_talk_scale
    alpha_aerosol = np.asarray(optics.alpha_aerosol_m_inv, dtype=float)
    beta_aerosol = np.asarray(optics.beta_aerosol_m_inv_sr_inv, dtype=float)
    alpha_molecular = np.asarray(optics.alpha_molecular_m_inv, dtype=float)
    beta_molecular = np.asarray(optics.beta_molecular_m_inv_sr_inv, dtype=float)
    attenuation = np.exp(-2.0 * instrument.range_m * (alpha_aerosol + alpha_molecular))
    response = channel_response_matrix(
        instrument.discriminator,
        discriminator_scale,
        cross_talk_override=cross_talk,
    )
    components = np.empty(6, dtype=float)
    components[0::2] = attenuation * beta_aerosol
    components[1::2] = attenuation * beta_molecular
    response_signal = response @ components
    channel_scale = instrument.count_scale * instrument.bin_width_m
    channel_scale *= np.repeat(energy, 2) * gain * overlap
    signal_counts = channel_scale * response_signal
    background_counts = np.asarray(instrument.background_rate) * instrument.counting_time_s
    expected_counts = _dead_time_forward(signal_counts + background_counts, instrument)
    if np.any(~np.isfinite(expected_counts)) or np.any(expected_counts < 0):
        raise ValueError("expected counts must be finite and nonnegative")
    return SignalExpectation(
        signal_counts=signal_counts,
        background_counts=background_counts,
        expected_counts=expected_counts,
        attenuation=attenuation,
        response_matrix=response,
    )


def _lognormal_multipliers(
    rng: np.random.Generator,
    relative_error: float | tuple[float, ...],
    length: int,
    name: str,
) -> np.ndarray:
    relative = _vector(relative_error, length, name)
    if np.any(relative < 0):
        raise ValueError(f"{name} must be nonnegative")
    sigma = np.sqrt(np.log1p(relative**2))
    return rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)


def _count_covariance(
    optics: OpticalProperties,
    expectation: SignalExpectation,
    instrument: InstrumentConfig,
    noise: SignalNoiseConfig,
) -> np.ndarray:
    pre_dead_time_counts = expectation.signal_counts + expectation.background_counts
    dead_time = np.asarray(instrument.dead_time_s, dtype=float)
    dead_time_factor = 1.0 / (1.0 + pre_dead_time_counts * dead_time / instrument.counting_time_s) ** 2
    poisson_variance = expectation.expected_counts * dead_time_factor
    covariance = np.diag(poisson_variance if noise.poisson else np.zeros(6))
    signal = expectation.signal_counts
    propagated_signal = signal * dead_time_factor
    gain_error = _vector(noise.gain_relative_error, 6, "gain_relative_error")
    energy_error = _vector(
        noise.laser_energy_relative_error, 3, "laser_energy_relative_error"
    )
    discriminator_error = _vector(
        noise.discriminator_relative_error, 3, "discriminator_relative_error"
    )
    cross_talk_error = _vector(
        noise.discriminator_cross_talk_relative_error, 6,
        "discriminator_cross_talk_relative_error",
    ).reshape(3, 2)
    covariance += np.diag((propagated_signal * gain_error) ** 2)
    for wavelength_index in range(3):
        channel_slice = slice(2 * wavelength_index, 2 * wavelength_index + 2)
        energy_vector = np.zeros(6)
        energy_vector[channel_slice] = propagated_signal[channel_slice]
        covariance += np.outer(energy_vector, energy_vector) * energy_error[wavelength_index] ** 2
        discriminator_vector = np.zeros(6)
        discriminator_vector[channel_slice] = propagated_signal[channel_slice]
        covariance += (
            np.outer(discriminator_vector, discriminator_vector)
            * discriminator_error[wavelength_index] ** 2
        )
        for cross_talk_index in range(2):
            nominal_cross_talk = instrument.discriminator.cross_talk[
                wavelength_index, cross_talk_index
            ]
            if nominal_cross_talk == 0.0:
                continue
            perturbation = 1.0e-5
            multiplier = np.ones((3, 2), dtype=float)
            multiplier[wavelength_index, cross_talk_index] += perturbation
            perturbed = expected_signal_counts(
                optics,
                instrument,
                cross_talk_multiplier=multiplier,
            ).expected_counts
            derivative = (perturbed - expectation.expected_counts) / perturbation
            covariance += (
                np.outer(derivative, derivative)
                * cross_talk_error[wavelength_index, cross_talk_index] ** 2
            )
    return 0.5 * (covariance + covariance.T)


def _dead_time_inverse(raw_counts: np.ndarray, instrument: InstrumentConfig) -> tuple[np.ndarray, bool]:
    dead_time = np.asarray(instrument.dead_time_s, dtype=float)
    fraction = raw_counts * dead_time / instrument.counting_time_s
    if np.any(fraction >= 1.0):
        return np.full(6, np.nan), False
    return raw_counts / (1.0 - fraction), True


def _extract_optical_values(
    raw_counts: np.ndarray,
    covariance_counts: np.ndarray,
    optics: OpticalProperties,
    instrument: InstrumentConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    flags: list[str] = []
    raw_counts = np.asarray(raw_counts, dtype=float)
    covariance_counts = np.asarray(covariance_counts, dtype=float)
    if raw_counts.shape != (6,) or covariance_counts.shape != (6, 6):
        raise ValueError("raw_counts and covariance_counts must have shapes (6,) and (6, 6)")
    corrected_total, valid_dead_time = _dead_time_inverse(raw_counts, instrument)
    if not valid_dead_time:
        flags.append("dead_time_saturated")
        return np.full(6, np.nan), np.full((6, 6), np.nan), tuple(flags)
    corrected_signal = corrected_total - np.asarray(instrument.background_rate) * instrument.counting_time_s
    if np.any(corrected_signal <= 0.0) or np.any(~np.isfinite(corrected_signal)):
        flags.append("background_subtraction_nonpositive")
        return np.full(6, np.nan), np.full((6, 6), np.nan), tuple(flags)
    response = channel_response_matrix(instrument.discriminator)
    channel_scale = instrument.count_scale * instrument.bin_width_m
    channel_scale *= np.repeat(instrument.laser_energy_relative, 2)
    channel_scale *= np.asarray(instrument.system_gain) * np.asarray(instrument.overlap)
    components = np.empty(6, dtype=float)
    jacobian = np.zeros((6, 6), dtype=float)
    beta_molecular = np.asarray(optics.beta_molecular_m_inv_sr_inv, dtype=float)
    alpha_molecular = np.asarray(optics.alpha_molecular_m_inv, dtype=float)
    for wavelength_index in range(3):
        channel_slice = slice(2 * wavelength_index, 2 * wavelength_index + 2)
        block = response[channel_slice, channel_slice]
        if abs(np.linalg.det(block)) < 1e-14:
            flags.append("singular_discriminator_response")
            return np.full(6, np.nan), np.full((6, 6), np.nan), tuple(flags)
        normalized = corrected_signal[channel_slice] / channel_scale[channel_slice]
        component_slice = np.linalg.solve(block, normalized)
        components[channel_slice] = component_slice
        molecular_component = component_slice[1]
        if molecular_component <= 0.0 or not np.isfinite(molecular_component):
            flags.append(f"nonpositive_molecular_component_{instrument.wavelengths_nm[wavelength_index]}")
            return np.full(6, np.nan), np.full((6, 6), np.nan), tuple(flags)
        attenuation = molecular_component / beta_molecular[wavelength_index]
        if attenuation <= 0.0 or not np.isfinite(attenuation):
            flags.append(f"invalid_attenuation_{instrument.wavelengths_nm[wavelength_index]}")
            return np.full(6, np.nan), np.full((6, 6), np.nan), tuple(flags)
        beta_aerosol = component_slice[0] / attenuation
        alpha_total = -np.log(attenuation) / (2.0 * instrument.range_m)
        alpha_aerosol = alpha_total - alpha_molecular[wavelength_index]
        components[2 * wavelength_index] = beta_aerosol
        components[2 * wavelength_index + 1] = alpha_aerosol
        inverse_block = np.linalg.solve(block, np.eye(2))
        dead_time = instrument.dead_time_s[channel_slice]
        raw_derivative = 1.0 / (1.0 - raw_counts[channel_slice] * dead_time / instrument.counting_time_s) ** 2
        component_derivative = inverse_block @ np.diag(raw_derivative / channel_scale[channel_slice])
        d_beta_d_components = np.array(
            [beta_molecular[wavelength_index] / molecular_component,
             -component_slice[0] * beta_molecular[wavelength_index] / molecular_component**2]
        )
        d_alpha_d_components = np.array([0.0, -1.0 / (2.0 * instrument.range_m * molecular_component)])
        jacobian[2 * wavelength_index, channel_slice] = d_beta_d_components @ component_derivative
        jacobian[2 * wavelength_index + 1, channel_slice] = d_alpha_d_components @ component_derivative
    optical_order = np.array([0, 2, 4, 1, 3, 5])
    components = components[optical_order]
    jacobian = jacobian[optical_order]
    covariance = jacobian @ covariance_counts @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(~np.isfinite(components)) or np.any(~np.isfinite(covariance)):
        flags.append("nonfinite_optical_extraction")
        return np.full(6, np.nan), np.full((6, 6), np.nan), tuple(flags)
    return components, covariance, tuple(flags)


def simulate_signal_observation(
    optics: OpticalProperties,
    instrument: InstrumentConfig,
    seed: int,
    noise: SignalNoiseConfig | None = None,
) -> SignalObservation:
    """Sample raw six-channel counts and extract paired optical observations."""

    noise = noise or SignalNoiseConfig()
    rng = np.random.default_rng(seed)
    gain_multiplier = _lognormal_multipliers(
        rng, noise.gain_relative_error, 6, "gain_relative_error"
    )
    energy_multiplier = _lognormal_multipliers(
        rng, noise.laser_energy_relative_error, 3, "laser_energy_relative_error"
    )
    discriminator_multiplier = _lognormal_multipliers(
        rng, noise.discriminator_relative_error, 3, "discriminator_relative_error"
    )
    cross_talk_multiplier = _lognormal_multipliers(
        rng,
        noise.discriminator_cross_talk_relative_error,
        6,
        "discriminator_cross_talk_relative_error",
    ).reshape(3, 2)
    nominal_cross_talk = np.asarray(instrument.discriminator.cross_talk, dtype=float)
    aerosol_transmission = np.asarray(instrument.discriminator.aerosol_transmission, dtype=float)
    molecular_transmission = np.asarray(instrument.discriminator.molecular_transmission, dtype=float)
    cross_talk_limit = np.column_stack(
        (
            (1.0 - aerosol_transmission) / aerosol_transmission,
            (1.0 - molecular_transmission) / molecular_transmission,
        )
    )
    proposed_cross_talk = nominal_cross_talk * cross_talk_multiplier
    proposed_cross_talk = np.minimum(proposed_cross_talk, 0.999 * cross_talk_limit)
    cross_talk_multiplier = np.divide(
        proposed_cross_talk,
        nominal_cross_talk,
        out=np.zeros_like(proposed_cross_talk),
        where=nominal_cross_talk > 0.0,
    )
    actual_expectation = expected_signal_counts(
        optics,
        instrument,
        laser_energy_multiplier=energy_multiplier,
        gain_multiplier=gain_multiplier,
        discriminator_multiplier=discriminator_multiplier,
        cross_talk_multiplier=cross_talk_multiplier,
    )
    nominal_expectation = expected_signal_counts(optics, instrument)
    if noise.poisson and np.any(np.asarray(instrument.dead_time_s) > 0.0):
        dead_time = np.asarray(instrument.dead_time_s, dtype=float)
        pre_dead_time = actual_expectation.signal_counts + actual_expectation.background_counts
        fano = 1.0 / (1.0 + pre_dead_time * dead_time / instrument.counting_time_s) ** 2
        raw_counts = np.rint(
            np.maximum(
                0.0,
                rng.normal(actual_expectation.expected_counts, np.sqrt(actual_expectation.expected_counts * fano)),
            )
        ).astype(np.int64)
        noise_model_id = "signal_dead_time_gaussian_approx_poisson"
    elif noise.poisson:
        raw_counts = rng.poisson(actual_expectation.expected_counts).astype(np.int64)
        noise_model_id = "signal_poisson"
    else:
        raw_counts = np.rint(actual_expectation.expected_counts).astype(np.int64)
        noise_model_id = "signal_deterministic"
    raw_signal = raw_counts.astype(float) / instrument.counting_time_s
    covariance_counts = _count_covariance(optics, nominal_expectation, instrument, noise)
    optical_plus, covariance_plus, extraction_flags = _extract_optical_values(
        raw_counts, covariance_counts, optics, instrument
    )
    flags = list(extraction_flags)
    for wavelength_index, wavelength_nm in enumerate(instrument.wavelengths_nm):
        if actual_expectation.expected_counts[2 * wavelength_index + 1] < noise.min_molecular_expected_counts:
            flags.append(f"low_molecular_expected_counts_{wavelength_nm}")
        if raw_counts[2 * wavelength_index + 1] == 0:
            flags.append(f"zero_molecular_counts_{wavelength_nm}")
    return SignalObservation(
        raw_signal=raw_signal,
        raw_counts=raw_counts,
        expected_counts=actual_expectation.expected_counts,
        optical_base=optical_plus[:-1].copy(),
        optical_plus=optical_plus,
        covariance_base=covariance_plus[:-1, :-1].copy(),
        covariance_plus=covariance_plus,
        truth_plus=optical_vector(optics),
        covariance_counts=covariance_counts,
        gain_multiplier=gain_multiplier,
        laser_energy_multiplier=energy_multiplier,
        discriminator_multiplier=discriminator_multiplier,
        cross_talk_multiplier=cross_talk_multiplier,
        seed=seed,
        noise_model_id=noise_model_id + "_gain_laser_discriminator",
        quality_flags=tuple(dict.fromkeys(flags)),
    )
