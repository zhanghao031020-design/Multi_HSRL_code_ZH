from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AerosolState:
    """Bimodal aerosol state used by the first optical-level experiment.

    Volumes are aerosol volume per air volume in m3/m3. Median radii are in m.
    The refractive-index and width fields are fixed nuisance values in the
    first retrieval and are varied only by outer sensitivity experiments.
    """

    fine_volume: float
    coarse_volume: float
    fine_rv_m: float
    coarse_rv_m: float
    fine_sigma_g: float = 1.55
    coarse_sigma_g: float = 1.75
    fine_refractive_index: complex = 1.45 + 0.0j
    coarse_refractive_index: complex = 1.50 + 0.0j
    relative_humidity: float = 0.0
    scenario_id: str = "unspecified"
    truth_id: str = "unspecified"

    def __post_init__(self) -> None:
        positive = {
            "fine_volume": self.fine_volume,
            "coarse_volume": self.coarse_volume,
            "fine_rv_m": self.fine_rv_m,
            "coarse_rv_m": self.coarse_rv_m,
            "fine_sigma_g": self.fine_sigma_g,
            "coarse_sigma_g": self.coarse_sigma_g,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.relative_humidity <= 100.0:
            raise ValueError("relative_humidity must be between 0 and 100")


@dataclass(frozen=True)
class OpticalProperties:
    """Three-wavelength aerosol and molecular optical properties.

    Alpha is in m^-1. Beta is a differential volume backscatter coefficient
    in m^-1 sr^-1, i.e. per unit solid angle at 180 degrees.
    """

    wavelengths_m: tuple[float, ...]
    alpha_aerosol_m_inv: tuple[float, ...]
    beta_aerosol_m_inv_sr_inv: tuple[float, ...]
    alpha_molecular_m_inv: tuple[float, ...]
    beta_molecular_m_inv_sr_inv: tuple[float, ...]
    lidar_ratio: tuple[float, ...] = field(default_factory=tuple)
    angstrom_355_532: float | None = None
    angstrom_532_1064: float | None = None
    longwave_curvature: float | None = None

    def __post_init__(self) -> None:
        lengths = {
            "alpha_aerosol_m_inv": len(self.alpha_aerosol_m_inv),
            "beta_aerosol_m_inv_sr_inv": len(self.beta_aerosol_m_inv_sr_inv),
            "alpha_molecular_m_inv": len(self.alpha_molecular_m_inv),
            "beta_molecular_m_inv_sr_inv": len(self.beta_molecular_m_inv_sr_inv),
        }
        if any(length != len(self.wavelengths_m) for length in lengths.values()):
            raise ValueError(f"optical arrays must have equal lengths: {lengths}")

    @property
    def beta_total_m_inv_sr_inv(self) -> tuple[float, ...]:
        return tuple(
            aerosol + molecular
            for aerosol, molecular in zip(
                self.beta_aerosol_m_inv_sr_inv, self.beta_molecular_m_inv_sr_inv
            )
        )

    @property
    def alpha_total_m_inv(self) -> tuple[float, ...]:
        return tuple(
            aerosol + molecular
            for aerosol, molecular in zip(
                self.alpha_aerosol_m_inv, self.alpha_molecular_m_inv
            )
        )
