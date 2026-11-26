"""Names and schema for local density/DG node features in G7."""

from __future__ import annotations

import numpy as np

from .config import AtomicFeaturesConfig


_CURVATURE_FEATURE_NAMES = (
    "H",
    "K",
    "kappa_min",
    "kappa_max",
    "curvature_valid",
)


def _source_channels(config: AtomicFeaturesConfig) -> tuple[str, ...]:
    """Return configured source-element channels, including ``Other``."""
    return (*config.density_elements, config.other_label)


def _kernel_label(config: AtomicFeaturesConfig, kernel_index: int) -> str:
    """Return the stable name fragment for one configured DG kernel."""
    return f"{config.dg_kernel}_{kernel_index}"


def _load_g6_derivative_primitives():
    """Load the analytical G6 derivative primitives without duplicating them."""
    from src.features.dg_gl_features import (
        density_field_derivatives as g6_density_field_derivatives,
        radial_kernel_derivatives as g6_radial_kernel_derivatives,
    )

    return g6_radial_kernel_derivatives, g6_density_field_derivatives


def radial_kernel_derivatives(
    distances: np.ndarray,
    scales: np.ndarray,
    kernel: str,
    order: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the G6 radial kernel and first-two-derivative implementation."""
    g6_radial_kernel_derivatives, _ = _load_g6_derivative_primitives()
    return g6_radial_kernel_derivatives(
        distances=distances,
        scales=scales,
        kernel=kernel,
        order=order,
    )


def density_field_derivatives(
    evaluation_coords: np.ndarray,
    source_coords: np.ndarray,
    source_weights: np.ndarray,
    pair_scales: np.ndarray,
    allowed_mask: np.ndarray,
    kernel: str,
    order: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the G6 weighted density, gradient, and Hessian implementation."""
    _, g6_density_field_derivatives = _load_g6_derivative_primitives()
    return g6_density_field_derivatives(
        evaluation_coords=evaluation_coords,
        source_coords=source_coords,
        source_weights=source_weights,
        pair_scales=pair_scales,
        allowed_mask=allowed_mask,
        kernel=kernel,
        order=order,
    )


def get_density_feature_names(
    config: AtomicFeaturesConfig,
) -> tuple[str, ...]:
    """Return the ordered local ``x_dg`` feature names for ``config``.

    Features are grouped by density type, source-element channel, and DG
    kernel configuration. The ``Other`` channel is always appended to the
    configured element vocabulary because it is reserved for unlisted source
    elements. Density values use ``rho``; when curvature is enabled, the
    corresponding node-local channels are ``H``, ``K``, ``kappa_min``,
    ``kappa_max``, and ``curvature_valid``.

    This task defines only the schema. It does not evaluate density, derive
    curvature, replace invalid values, or apply numeric transformations.
    """
    if not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance.")

    if not config.density_enabled and not config.curvature_enabled:
        return ()

    names: list[str] = []
    for density_type in config.density_types:
        for source_element in _source_channels(config):
            for kernel_index, _ in enumerate(config.dg_kernel_configs):
                channel = (
                    f"{density_type}_from_{source_element}_"
                    f"kernel_{_kernel_label(config, kernel_index)}"
                )
                if config.density_enabled:
                    names.append(f"rho_{channel}")
                if config.curvature_enabled:
                    names.extend(
                        f"{feature_name}_{channel}"
                        for feature_name in _CURVATURE_FEATURE_NAMES
                    )

    return tuple(names)
