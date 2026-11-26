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


def _number_density_from_geometry(
    coords: np.ndarray,
    mapped_symbols: np.ndarray,
    atomic_radii: np.ndarray,
    config: AtomicFeaturesConfig,
) -> np.ndarray:
    """Build the configured per-atom number-density columns.

    This is deliberately limited to ``rho``. Curvature, validity masks,
    charge weights, and numeric transformations belong to later tasks.
    """
    coords = np.asarray(coords, dtype=float)
    mapped_symbols = np.asarray(mapped_symbols)
    atomic_radii = np.asarray(atomic_radii, dtype=float)

    n_atoms = len(coords)
    columns: list[np.ndarray] = []

    for source_element in _source_channels(config):
        source_indices = np.flatnonzero(mapped_symbols == source_element)

        for order, tau in config.dg_kernel_configs:
            if source_indices.size == 0:
                columns.append(np.zeros(n_atoms, dtype=float))
                continue

            source_coords = coords[source_indices]
            source_radii = atomic_radii[source_indices]
            source_weights = np.ones(source_indices.size, dtype=float)

            differences = (
                coords[:, None, :]
                - source_coords[None, :, :]
            )
            distances = np.linalg.norm(differences, axis=2)
            pair_radius_sums = (
                atomic_radii[:, None]
                + source_radii[None, :]
            )

            # Keep the G6 geometric exclusion: self/covalent-range pairs do
            # not contribute to a local density channel.
            allowed_mask = distances > pair_radius_sums
            pair_scales = tau * pair_radius_sums

            density, _, _ = density_field_derivatives(
                evaluation_coords=coords,
                source_coords=source_coords,
                source_weights=source_weights,
                pair_scales=pair_scales,
                allowed_mask=allowed_mask,
                kernel=config.dg_kernel,
                order=order,
            )
            columns.append(np.asarray(density, dtype=float))

    if not columns:
        return np.empty((n_atoms, 0), dtype=float)

    return np.column_stack(columns)


def build_density_features(
    mol,
    conf_id: int,
    config: AtomicFeaturesConfig,
) -> np.ndarray:
    """Return local per-atom number-density features as ``[N, F_dg]``.

    Coordinates and van der Waals radii follow the existing G6 helpers, and
    density values use the shared analytical G6 derivative implementation.
    The current task intentionally supports only number density; charge and
    curvature are rejected until their dedicated tasks are implemented.
    """
    if not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance.")

    n_atoms = int(mol.GetNumAtoms())
    if not config.density_enabled:
        return np.empty((n_atoms, 0), dtype=np.float32)

    if config.curvature_enabled:
        raise NotImplementedError(
            "curvature features are implemented in T4.5 and later."
        )
    if config.density_types != ("number",):
        raise NotImplementedError(
            "charge density features are implemented in T4.4 and later."
        )

    from src.features.descriptors_3d import get_coordinates
    from src.features.dg_gl_features import get_vdw_radii
    from src.features.density_features import get_mapped_atom_symbols

    coords = get_coordinates(mol=mol, conf_id=conf_id)
    mapped_symbols = get_mapped_atom_symbols(
        mol=mol,
        elements=list(config.density_elements),
        other_label=config.other_label,
    )
    atomic_radii = get_vdw_radii(mol=mol)

    density = _number_density_from_geometry(
        coords=coords,
        mapped_symbols=mapped_symbols,
        atomic_radii=atomic_radii,
        config=config,
    )
    return np.asarray(density, dtype=np.float32)


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
