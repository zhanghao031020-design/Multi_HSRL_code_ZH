import numpy as np
import pytest

from hsrl_sim.distributions import (
    aerosol_products,
    bimodal_volume_distribution,
    effective_radius_m,
    modal_volume_distribution,
)
from hsrl_sim.schemas import AerosolState


RADIUS_GRID_M = np.geomspace(0.01e-6, 15.0e-6, 4096)


def test_modal_volume_distribution_conserves_requested_volume():
    distribution = modal_volume_distribution(
        radius_grid_m=RADIUS_GRID_M,
        total_volume=2.0e-12,
        median_radius_m=0.12e-6,
        sigma_g=1.6,
    )

    assert np.trapezoid(distribution, RADIUS_GRID_M) == pytest.approx(2.0e-12, rel=2e-3)


def test_bimodal_distribution_conserves_each_state_volume():
    state = AerosolState(
        fine_volume=1.5e-12,
        coarse_volume=4.0e-12,
        fine_rv_m=0.10e-6,
        coarse_rv_m=1.0e-6,
    )

    distribution = bimodal_volume_distribution(state, RADIUS_GRID_M)

    assert np.trapezoid(distribution, RADIUS_GRID_M) == pytest.approx(
        state.fine_volume + state.coarse_volume, rel=2e-3
    )


def test_effective_radius_is_positive_and_between_modal_scales():
    state = AerosolState(
        fine_volume=1.0e-12,
        coarse_volume=5.0e-12,
        fine_rv_m=0.10e-6,
        coarse_rv_m=1.0e-6,
    )

    reff_m = effective_radius_m(RADIUS_GRID_M, bimodal_volume_distribution(state, RADIUS_GRID_M))

    assert 0.10e-6 < reff_m < 5.0e-6


def test_aerosol_products_include_all_registered_low_dimensional_products():
    state = AerosolState(
        fine_volume=1.0e-12,
        coarse_volume=4.0e-12,
        fine_rv_m=0.10e-6,
        coarse_rv_m=1.0e-6,
    )

    products = aerosol_products(state, RADIUS_GRID_M)

    assert set(products) == {"Vf", "Vc", "Vf_over_Vc", "reff_total", "reff_coarse"}
    assert products["Vf"] == pytest.approx(state.fine_volume)
    assert products["Vc"] == pytest.approx(state.coarse_volume)
    assert products["Vf_over_Vc"] == pytest.approx(products["Vf"] / products["Vc"])
    assert products["reff_total"] > 0
    assert products["reff_coarse"] > products["reff_total"]
