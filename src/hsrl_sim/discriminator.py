from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DiscriminatorConfig:
    """Parameterized two-output HSRL discriminator for each wavelength.

    ``cross_talk[k, 0]`` is aerosol leakage into the molecular-labelled
    channel and ``cross_talk[k, 1]`` is molecular leakage into the mixed
    channel.  The channel names remain signal labels; neither output is
    treated as a pure physical component.
    """

    molecular_transmission: tuple[float, ...] = (0.12, 0.12, 0.12)
    aerosol_transmission: tuple[float, ...] = (0.85, 0.85, 0.85)
    cross_talk: np.ndarray = field(
        default_factory=lambda: np.full((3, 2), 0.01, dtype=float)
    )
    frequency_offset_hz: tuple[float, ...] = (0.0, 0.0, 0.0)
    frequency_acceptance_width_hz: float = 1.0e8

    def __post_init__(self) -> None:
        molecular = np.asarray(self.molecular_transmission, dtype=float)
        aerosol = np.asarray(self.aerosol_transmission, dtype=float)
        offsets = np.asarray(self.frequency_offset_hz, dtype=float)
        cross_talk = np.asarray(self.cross_talk, dtype=float)
        if molecular.shape != (3,) or aerosol.shape != (3,) or offsets.shape != (3,):
            raise ValueError("discriminator transmissions and offsets must have length three")
        if cross_talk.shape != (3, 2):
            raise ValueError("cross_talk must have shape (3, 2)")
        if np.any(~np.isfinite(molecular)) or np.any(~np.isfinite(aerosol)):
            raise ValueError("discriminator transmissions must be finite")
        if np.any((molecular < 0.0) | (molecular > 1.0)):
            raise ValueError("molecular_transmission must be in [0, 1]")
        if np.any((aerosol < 0.0) | (aerosol > 1.0)):
            raise ValueError("aerosol_transmission must be in [0, 1]")
        if np.any(~np.isfinite(cross_talk)) or np.any((cross_talk < 0.0) | (cross_talk > 1.0)):
            raise ValueError("cross_talk must be finite and in [0, 1]")
        if (
            np.any(~np.isfinite(offsets))
            or not np.isfinite(self.frequency_acceptance_width_hz)
            or self.frequency_acceptance_width_hz <= 0
        ):
            raise ValueError("frequency offsets must be finite and acceptance width positive")
        total_aerosol_response = aerosol * (1.0 + cross_talk[:, 0])
        total_molecular_response = molecular * (1.0 + cross_talk[:, 1])
        if np.any(total_aerosol_response > 1.0 + 1e-12) or np.any(total_molecular_response > 1.0 + 1e-12):
            raise ValueError("transmission plus cross-talk must not exceed unit total response")
        object.__setattr__(self, "molecular_transmission", tuple(float(v) for v in molecular))
        object.__setattr__(self, "aerosol_transmission", tuple(float(v) for v in aerosol))
        object.__setattr__(self, "frequency_offset_hz", tuple(float(v) for v in offsets))
        object.__setattr__(self, "cross_talk", cross_talk.copy())


def _effective_molecular_transmission(config: DiscriminatorConfig) -> np.ndarray:
    molecular = np.asarray(config.molecular_transmission, dtype=float)
    offset = np.asarray(config.frequency_offset_hz, dtype=float)
    width = float(config.frequency_acceptance_width_hz)
    return molecular * np.exp(-0.5 * (offset / width) ** 2)


def channel_response_matrix(
    config: DiscriminatorConfig,
    transmission_scale: np.ndarray | None = None,
    cross_talk_override: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 6x6 response for interleaved aerosol/molecular components.

    The input and output ordering is ``[aerosol_355, molecular_355, ...]``
    and ``[mixed_355, molecular_355, ...]`` respectively.
    """

    aerosol = np.asarray(config.aerosol_transmission, dtype=float)
    molecular = _effective_molecular_transmission(config)
    if transmission_scale is not None:
        scale = np.asarray(transmission_scale, dtype=float)
        if scale.shape != (3,) or np.any(~np.isfinite(scale)) or np.any(scale < 0.0):
            raise ValueError("transmission_scale must be a finite nonnegative length-three vector")
        aerosol = aerosol * scale
        molecular = molecular * scale
    if cross_talk_override is None:
        cross_talk = config.cross_talk
    else:
        cross_talk = np.asarray(cross_talk_override, dtype=float)
        if cross_talk.shape != (3, 2) or np.any(~np.isfinite(cross_talk)) or np.any(cross_talk < 0.0):
            raise ValueError("cross_talk_override must be a finite nonnegative (3, 2) array")
        if np.any(aerosol * (1.0 + cross_talk[:, 0]) > 1.0 + 1e-12) or np.any(
            molecular * (1.0 + cross_talk[:, 1]) > 1.0 + 1e-12
        ):
            raise ValueError("transmission plus cross-talk must not exceed unit total response")
    response = np.zeros((6, 6), dtype=float)
    for wavelength_index in range(3):
        aerosol_index = 2 * wavelength_index
        molecular_index = aerosol_index + 1
        aerosol_to_molecular, molecular_to_mixed = cross_talk[wavelength_index]
        response[aerosol_index, aerosol_index] = aerosol[wavelength_index]
        response[aerosol_index, molecular_index] = molecular_to_mixed * molecular[wavelength_index]
        response[molecular_index, aerosol_index] = aerosol_to_molecular * aerosol[wavelength_index]
        response[molecular_index, molecular_index] = molecular[wavelength_index]
    return response
