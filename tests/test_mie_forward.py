import numpy as np
import pytest

from hsrl_sim.mie_forward import compute_aerosol_optics, differential_backscatter_from_qback
from hsrl_sim.molecular import molecular_backscatter_m_inv_sr_inv
from hsrl_sim.schemas import AerosolState


WAVELENGTHS_M = np.array([355e-9, 532e-9, 1064e-9])
RADIUS_GRID_M = np.geomspace(0.01e-6, 15.0e-6, 512)


def make_state(fine_volume=1.0e-12, coarse_volume=4.0e-12):
    return AerosolState(
        fine_volume=fine_volume,
        coarse_volume=coarse_volume,
        fine_rv_m=0.10e-6,
        coarse_rv_m=1.0e-6,
    )


def test_qback_conversion_is_per_steradian():
    radius_m = 0.5e-6
    qback = 2.0

    result = differential_backscatter_from_qback(radius_m, qback)

    assert result == pytest.approx(radius_m**2 * qback / 4.0)


def test_mie_optics_are_positive_and_have_expected_shapes():
    optics = compute_aerosol_optics(make_state(), WAVELENGTHS_M, RADIUS_GRID_M)

    assert len(optics.wavelengths_m) == 3
    assert np.all(np.asarray(optics.alpha_aerosol_m_inv) > 0)
    assert np.all(np.asarray(optics.beta_aerosol_m_inv_sr_inv) > 0)
    assert np.all(np.isfinite(optics.alpha_aerosol_m_inv))
    assert np.all(np.isfinite(optics.beta_aerosol_m_inv_sr_inv))


def test_mie_optics_scale_linearly_with_modal_volume():
    optics_a = compute_aerosol_optics(make_state(), WAVELENGTHS_M, RADIUS_GRID_M)
    optics_b = compute_aerosol_optics(make_state(fine_volume=2.0e-12, coarse_volume=8.0e-12), WAVELENGTHS_M, RADIUS_GRID_M)

    assert np.asarray(optics_b.alpha_aerosol_m_inv) == pytest.approx(
        2.0 * np.asarray(optics_a.alpha_aerosol_m_inv), rel=2e-3
    )
    assert np.asarray(optics_b.beta_aerosol_m_inv_sr_inv) == pytest.approx(
        2.0 * np.asarray(optics_a.beta_aerosol_m_inv_sr_inv), rel=2e-3
    )


def test_coarse_refractive_index_changes_coarse_optics():
    state_a = make_state()
    state_b = AerosolState(
        fine_volume=state_a.fine_volume,
        coarse_volume=state_a.coarse_volume,
        fine_rv_m=state_a.fine_rv_m,
        coarse_rv_m=state_a.coarse_rv_m,
        fine_refractive_index=state_a.fine_refractive_index,
        coarse_refractive_index=1.70 + 0.0j,
    )

    optics_a = compute_aerosol_optics(state_a, WAVELENGTHS_M, RADIUS_GRID_M)
    optics_b = compute_aerosol_optics(state_b, WAVELENGTHS_M, RADIUS_GRID_M)

    assert not np.allclose(optics_a.alpha_aerosol_m_inv, optics_b.alpha_aerosol_m_inv)
    assert not np.allclose(optics_a.beta_aerosol_m_inv_sr_inv, optics_b.beta_aerosol_m_inv_sr_inv)


def test_molecular_backscatter_decreases_with_wavelength():
    values = [molecular_backscatter_m_inv_sr_inv(wavelength) for wavelength in WAVELENGTHS_M]

    assert values[0] > values[1] > values[2]
    assert all(value > 0 for value in values)


def test_converged_marine_grid_is_within_two_percent_of_refined_grid():
    marine_state = AerosolState(
        fine_volume=4.0e-13,
        coarse_volume=5.0e-12,
        fine_rv_m=0.12e-6,
        coarse_rv_m=1.40e-6,
    )
    converged = compute_aerosol_optics(
        marine_state,
        WAVELENGTHS_M,
        np.geomspace(0.005e-6, 20.0e-6, 4096),
    )
    reference = compute_aerosol_optics(
        marine_state,
        WAVELENGTHS_M,
        np.geomspace(0.005e-6, 20.0e-6, 8192),
    )
    relative_error = max(
        np.max(
            np.abs(
                np.asarray(converged.alpha_aerosol_m_inv)
                / np.asarray(reference.alpha_aerosol_m_inv)
                - 1.0
            )
        ),
        np.max(
            np.abs(
                np.asarray(converged.beta_aerosol_m_inv_sr_inv)
                / np.asarray(reference.beta_aerosol_m_inv_sr_inv)
                - 1.0
            )
        ),
    )

    assert relative_error < 0.02
