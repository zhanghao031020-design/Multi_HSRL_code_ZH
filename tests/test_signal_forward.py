from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hsrl_sim.discriminator import DiscriminatorConfig, channel_response_matrix
from hsrl_sim.mie_forward import compute_aerosol_optics
from hsrl_sim.schemas import AerosolState
from hsrl_sim.signal_forward import (
    CHANNEL_NAMES,
    InstrumentConfig,
    SignalNoiseConfig,
    expected_signal_counts,
    simulate_signal_observation,
)


WAVELENGTHS_M = np.array([355e-9, 532e-9, 1064e-9])
RADIUS_GRID_M = np.geomspace(0.01e-6, 15.0e-6, 192)


def make_state() -> AerosolState:
    return AerosolState(
        fine_volume=1.2e-12,
        coarse_volume=3.0e-12,
        fine_rv_m=0.11e-6,
        coarse_rv_m=0.95e-6,
        scenario_id="smoke",
        truth_id="truth-0",
    )


def make_optics():
    return compute_aerosol_optics(make_state(), WAVELENGTHS_M, RADIUS_GRID_M)


def test_discriminator_zero_transmission_removes_the_selected_component():
    discriminator = DiscriminatorConfig(
        molecular_transmission=(0.0, 0.0, 0.0),
        aerosol_transmission=(1.0, 1.0, 1.0),
        cross_talk=np.zeros((3, 2)),
    )

    response = channel_response_matrix(discriminator)

    assert response.shape == (6, 6)
    assert np.all(response[[0, 2, 4], [1, 3, 5]] == 0.0)
    assert np.all(response[[1, 3, 5], [1, 3, 5]] == 0.0)
    assert np.all(response[[0, 2, 4], [0, 2, 4]] == 1.0)


def test_discriminator_rejects_nonfinite_width_and_nonconserving_crosstalk():
    with pytest.raises(ValueError):
        DiscriminatorConfig(frequency_acceptance_width_hz=np.nan)
    with pytest.raises(ValueError):
        DiscriminatorConfig(
            molecular_transmission=(1.0, 1.0, 1.0),
            aerosol_transmission=(1.0, 1.0, 1.0),
            cross_talk=np.ones((3, 2)),
        )


def test_expected_signal_counts_are_finite_nonnegative_and_attenuated():
    optics = make_optics()
    instrument = InstrumentConfig()

    expectation = expected_signal_counts(optics, instrument)

    assert expectation.expected_counts.shape == (6,)
    assert np.all(np.isfinite(expectation.expected_counts))
    assert np.all(expectation.expected_counts >= 0.0)
    assert np.all(expectation.attenuation > 0.0)
    assert np.all(expectation.attenuation <= 1.0)


def test_channel_order_is_explicit_and_1064_molecular_channel_is_weak():
    optics = make_optics()
    expectation = expected_signal_counts(optics, InstrumentConfig())

    assert CHANNEL_NAMES == (
        "mixed_355",
        "molecular_355",
        "mixed_532",
        "molecular_532",
        "mixed_1064",
        "molecular_1064",
    )
    molecular_counts = expectation.expected_counts[[1, 3, 5]]
    assert molecular_counts[2] < 0.1 * molecular_counts[0]
    assert "pure_molecular" not in " ".join(CHANNEL_NAMES)


def test_fixed_seed_reproduces_raw_signal_observation_and_base_plus_pair():
    optics = make_optics()
    noise = SignalNoiseConfig(
        poisson=True,
        gain_relative_error=0.02,
        laser_energy_relative_error=0.01,
        discriminator_relative_error=0.02,
    )
    first = simulate_signal_observation(optics, InstrumentConfig(), seed=123, noise=noise)
    second = simulate_signal_observation(optics, InstrumentConfig(), seed=123, noise=noise)

    assert np.array_equal(first.raw_counts, second.raw_counts)
    assert np.array_equal(first.raw_signal, second.raw_signal)
    assert np.array_equal(first.optical_base, first.optical_plus[:-1])
    assert np.array_equal(first.covariance_base, first.covariance_plus[:-1, :-1])
    assert first.raw_counts.shape == (6,)
    assert first.raw_signal.shape == (6,)


def test_poisson_counts_have_expected_mean_and_variance_without_systematic_errors():
    optics = make_optics()
    instrument = InstrumentConfig()
    noise = SignalNoiseConfig(poisson=True)
    expected = expected_signal_counts(optics, instrument).expected_counts[0]
    samples = np.asarray(
        [
            simulate_signal_observation(optics, instrument, seed=1000 + index, noise=noise)
            .raw_counts[0]
            for index in range(512)
        ],
        dtype=float,
    )

    np.testing.assert_allclose(np.mean(samples), expected, rtol=0.08, atol=8.0)
    np.testing.assert_allclose(np.var(samples, ddof=1), expected, rtol=0.15, atol=12.0)


def test_signal_observation_preserves_raw_and_transformed_records():
    observation = simulate_signal_observation(
        make_optics(),
        InstrumentConfig(),
        seed=9,
        noise=SignalNoiseConfig(poisson=False),
    )

    assert observation.raw_counts.dtype.kind in "iu"
    assert observation.raw_signal.dtype.kind == "f"
    assert observation.optical_base.shape == (5,)
    assert observation.optical_plus.shape == (6,)
    assert observation.covariance_base.shape == (5, 5)
    assert observation.covariance_plus.shape == (6, 6)
    assert observation.noise_model_id.startswith("signal_")
    assert isinstance(observation.quality_flags, tuple)


def test_noiseless_signal_extraction_recovers_the_optical_truth():
    optics = make_optics()
    high_count_instrument = replace(InstrumentConfig(), count_scale=3.0e12)
    observation = simulate_signal_observation(
        optics,
        high_count_instrument,
        seed=9,
        noise=SignalNoiseConfig(poisson=False),
    )

    np.testing.assert_allclose(observation.optical_plus, observation.truth_plus, rtol=2e-4, atol=1e-10)


def test_systematic_error_layer_changes_raw_expectation_but_is_reproducible():
    optics = make_optics()
    noise = SignalNoiseConfig(
        poisson=False,
        gain_relative_error=0.15,
        laser_energy_relative_error=0.10,
        discriminator_relative_error=0.10,
    )
    first = simulate_signal_observation(optics, InstrumentConfig(), seed=42, noise=noise)
    second = simulate_signal_observation(optics, InstrumentConfig(), seed=42, noise=noise)
    nominal = expected_signal_counts(optics, InstrumentConfig()).expected_counts

    assert np.array_equal(first.expected_counts, second.expected_counts)
    assert np.array_equal(first.raw_counts, second.raw_counts)
    assert np.array_equal(first.gain_multiplier, second.gain_multiplier)
    assert np.array_equal(first.laser_energy_multiplier, second.laser_energy_multiplier)
    assert np.array_equal(first.discriminator_multiplier, second.discriminator_multiplier)
    assert np.array_equal(first.cross_talk_multiplier, second.cross_talk_multiplier)
    assert not np.array_equal(first.expected_counts, nominal)


def test_independent_cross_talk_errors_do_not_create_spurious_sibling_covariance():
    optics = make_optics()
    noise = SignalNoiseConfig(
        poisson=False,
        discriminator_cross_talk_relative_error=0.10,
    )
    covariance = simulate_signal_observation(
        optics, InstrumentConfig(), seed=11, noise=noise
    ).covariance_counts

    assert covariance[0, 1] == pytest.approx(0.0, abs=1e-8)
    assert covariance[0, 0] > 0.0
    assert covariance[1, 1] > 0.0


def test_nonzero_dead_time_keeps_counts_and_covariance_finite():
    optics = make_optics()
    instrument = replace(
        InstrumentConfig(),
        dead_time_s=(1.0e-9,) * 6,
        count_scale=3.0e9,
    )
    noise = SignalNoiseConfig(
        poisson=True,
        gain_relative_error=0.10,
        discriminator_cross_talk_relative_error=0.10,
    )
    observation = simulate_signal_observation(optics, instrument, seed=23, noise=noise)

    assert np.all(np.isfinite(observation.raw_counts))
    assert np.all(np.isfinite(observation.covariance_counts))
    assert np.all(np.diag(observation.covariance_counts) >= 0.0)
    assert "dead_time_gaussian_approx" in observation.noise_model_id
