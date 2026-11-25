"""Stable schema and names for local chemical atom features.

This module defines the column contract for x_chem only. Atomic number
is kept separately in z and is deliberately not included here.
Feature values and RDKit extraction are implemented by later tasks.
"""

from __future__ import annotations

from .config import AtomicFeaturesConfig


# Scalar/binary features are always listed before categorical one-hot groups.
_SCALAR_FEATURES = (
    ("include_formal_charge", "formal_charge"),
    ("include_heavy_atom_degree", "heavy_atom_degree"),
    ("include_attached_hydrogens", "attached_hydrogens"),
    ("include_atomic_mass", "atomic_mass"),
    ("include_aromaticity", "aromaticity"),
    ("include_ring_membership", "ring_membership"),
)


def get_atom_feature_names(
    config: AtomicFeaturesConfig | None = None,
) -> tuple[str, ...]:
    """Return the ordered column names for x_chem.

    The schema is deterministic for a given AtomicFeaturesConfig:
    enabled scalar features come first in the fixed order declared above,
    followed by hybridization, chirality, and CIP one-hot columns in the
    vocabulary order configured for each group.

    z (atomic number) is intentionally excluded because the graph
    contract stores it as a separate tensor and the model embeds it
    independently from x_chem.

    Parameters
    ----------
    config:
        Atom-feature configuration. When omitted, the default
        AtomicFeaturesConfig is used.

    Returns
    -------
    tuple[str, ...]
        Immutable feature names in the exact column order of x_chem.
    """
    if config is None:
        config = AtomicFeaturesConfig()
    elif not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance or None.")

    names: list[str] = [
        feature_name
        for flag_name, feature_name in _SCALAR_FEATURES
        if getattr(config, flag_name)
    ]

    if config.include_hybridization:
        names.extend(
            f"hybridization_{value}" for value in config.hybridization_types
        )

    if config.include_chirality:
        names.extend(
            f"chirality_{value}" for value in config.chirality_types
        )

    if config.include_cip:
        names.extend(f"cip_{value}" for value in config.cip_types)

    return tuple(names)
