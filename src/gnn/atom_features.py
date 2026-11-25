"""Schema and focused encoders for local chemical atom features.

This module defines the column contract for x_chem and implements the
hybridization one-hot encoder. Atomic number is kept separately in z.
Other atom-feature extraction and the complete build_atom_features API
belong to subsequent tasks.
"""

from __future__ import annotations

from rdkit.Chem.rdchem import Atom as RDKitAtom

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

def get_hybridization_one_hot(
    atom: RDKitAtom,
    config: AtomicFeaturesConfig | None = None,
) -> tuple[int, ...]:
    """Encode an RDKit atom's hybridization using the configured vocabulary.

    The output order is exactly config.hybridization_types. A category
    absent from that vocabulary is rejected rather than silently encoded as
    an all-zero vector. When hybridization features are disabled, an empty
    tuple is returned.

    Parameters
    ----------
    atom:
        RDKit atom whose assigned hybridization is encoded.
    config:
        Atom-feature configuration. Defaults to AtomicFeaturesConfig().

    Returns
    -------
    tuple[int, ...]
        Binary one-hot values in configured vocabulary order.
    """
    if not isinstance(atom, RDKitAtom):
        raise TypeError("atom must be an RDKit Atom instance.")

    if config is None:
        config = AtomicFeaturesConfig()
    elif not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance or None.")

    if not config.include_hybridization:
        return ()

    hybridization = atom.GetHybridization()
    hybridization_name = getattr(hybridization, "name", None)
    if not isinstance(hybridization_name, str) or not hybridization_name:
        hybridization_name = str(hybridization).rsplit(".", 1)[-1]
    hybridization_name = hybridization_name.upper()

    if hybridization_name not in config.hybridization_types:
        raise ValueError(
            f"RDKit hybridization {hybridization_name!r} is not present in "
            "config.hybridization_types."
        )

    return tuple(
        1 if category == hybridization_name else 0
        for category in config.hybridization_types
    )

