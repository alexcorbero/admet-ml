"""Column schema for G7 graph-feature normalization.

This module defines which raw feature columns are eligible for scaling. It
does not fit statistics or mutate graphs; those responsibilities belong to
later normalization tasks.
"""

from __future__ import annotations

from typing import Final

from .config import GNNConfig
from .density_features import get_density_feature_names


_SCALABLE_ATOM_SCALARS: Final[tuple[tuple[str, str], ...]] = (
    ("include_formal_charge", "formal_charge"),
    ("include_heavy_atom_degree", "heavy_atom_degree"),
    ("include_attached_hydrogens", "attached_hydrogens"),
    ("include_atomic_mass", "atomic_mass"),
)

_SCALABLE_DG_PREFIXES: Final[tuple[str, ...]] = (
    "rho_",
    "H_",
    "K_",
    "kappa_min_",
    "kappa_max_",
)


def get_scalable_feature_columns(
    config: GNNConfig,
) -> dict[str, tuple[str, ...]]:
    """Return the ordered raw columns eligible for numeric scaling.

    The result is grouped by the tensors that own the columns. ``x`` is not
    returned because it is the concatenation of ``x_chem`` and ``x_dg``;
    later transforms can update those source tensors and rebuild ``x``.

    Included columns are continuous or count-like numeric values:

    * scalar chemical atom features in ``x_chem``;
    * density and curvature values in ``x_dg``;
    * distance and Gaussian RBF columns in ``edge_attr``.

    Atomic numbers, one-hot columns, binary flags, and ``curvature_valid``
    masks are deliberately excluded.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    atomic_config = config.atomic_features
    x_chem = tuple(
        feature_name
        for enabled_field, feature_name in _SCALABLE_ATOM_SCALARS
        if getattr(atomic_config, enabled_field)
    )

    density_names = get_density_feature_names(atomic_config)
    x_dg = tuple(
        feature_name
        for feature_name in density_names
        if feature_name.startswith(_SCALABLE_DG_PREFIXES)
    )

    edge_names: list[str] = []
    if config.graph.include_distance:
        edge_names.append("distance")
    if config.graph.include_rbf:
        edge_names.extend(
            f"gaussian_rbf_center_{index}"
            for index in range(config.graph.rbf_num_centers)
        )

    return {
        "x_chem": x_chem,
        "x_dg": x_dg,
        "edge_attr": tuple(edge_names),
    }


def get_scalable_feature_names(
    config: GNNConfig,
) -> dict[str, tuple[str, ...]]:
    """Compatibility alias for :func:`get_scalable_feature_columns`."""
    return get_scalable_feature_columns(config)
