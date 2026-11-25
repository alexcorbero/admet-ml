"""Schema, builder, and focused encoders for local chemical atom features.

Atomic number is kept separately in z; x_chem follows the ordered schema
returned by get_atom_feature_names().
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdchem import Atom as RDKitAtom

import numpy as np

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


def get_atom_scalar_features(
    atom: RDKitAtom,
    config: AtomicFeaturesConfig | None = None,
) -> tuple[int | float, ...]:
    """Return enabled scalar/binary atom features in schema order.

    Values are emitted in the same fixed order used by
    :func:`get_atom_feature_names`: formal charge, heavy-atom degree,
    attached hydrogens, atomic mass, aromaticity, and ring membership.
    Disabled fields are omitted. Hydrogen counts include both implicit
    hydrogens and explicit hydrogen neighbors.
    """
    if not isinstance(atom, RDKitAtom):
        raise TypeError("atom must be an RDKit Atom instance.")

    if config is None:
        config = AtomicFeaturesConfig()
    elif not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance or None.")

    context_dependent_features = []
    if config.include_attached_hydrogens:
        context_dependent_features.append("attached_hydrogens")
    if config.include_ring_membership:
        context_dependent_features.append("ring_membership")
    if context_dependent_features and not atom.HasOwningMol():
        feature_names = ", ".join(context_dependent_features)
        raise ValueError(
            f"{feature_names} require an atom associated with an RDKit molecule."
        )

    features: list[int | float] = []

    if config.include_formal_charge:
        features.append(atom.GetFormalCharge())

    if config.include_heavy_atom_degree:
        heavy_atom_degree = (
            sum(neighbor.GetAtomicNum() > 1 for neighbor in atom.GetNeighbors())
            if atom.HasOwningMol()
            else 0
        )
        features.append(heavy_atom_degree)

    if config.include_attached_hydrogens:
        features.append(atom.GetTotalNumHs(includeNeighbors=True))

    if config.include_atomic_mass:
        features.append(float(atom.GetMass()))

    if config.include_aromaticity:
        features.append(int(atom.GetIsAromatic()))

    if config.include_ring_membership:
        features.append(int(atom.IsInRing()))

    return tuple(features)


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

def get_chirality_one_hot(
    atom: RDKitAtom,
    config: AtomicFeaturesConfig | None = None,
) -> tuple[int, ...]:
    """Encode the RDKit chiral tag using the configured one-hot vocabulary.

    The vector follows config.chirality_types exactly. Atoms without an
    assigned chiral tag are represented by CHI_UNSPECIFIED. If chirality
    features are disabled, an empty tuple is returned.
    """
    if not isinstance(atom, RDKitAtom):
        raise TypeError("atom must be an RDKit Atom instance.")

    if config is None:
        config = AtomicFeaturesConfig()
    elif not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance or None.")

    if not config.include_chirality:
        return ()

    chiral_tag = atom.GetChiralTag()
    category = getattr(chiral_tag, "name", None)
    if not isinstance(category, str) or not category:
        category = str(chiral_tag).rsplit(".", 1)[-1]
    category = category.upper()

    if category not in config.chirality_types:
        raise ValueError(
            f"RDKit chiral tag {category!r} is not present in "
            "config.chirality_types."
        )

    return tuple(
        1 if value == category else 0
        for value in config.chirality_types
    )


def _get_cip_category(atom: RDKitAtom) -> str:
    """Return the assigned R/S code without mutating the source molecule."""
    if not atom.HasOwningMol():
        # Standalone RDKit atoms have no molecular stereochemical context.
        return "UNSPECIFIED"

    molecule = Chem.Mol(atom.GetOwningMol())

    atom_index = atom.GetIdx()
    if atom_index < 0 or atom_index >= molecule.GetNumAtoms():
        return "UNSPECIFIED"

    # Assign on a copy: callers can safely reuse their original RDKit Mol.
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    stereochemical_atom = molecule.GetAtomWithIdx(atom_index)
    if not stereochemical_atom.HasProp("_CIPCode"):
        return "UNSPECIFIED"

    return stereochemical_atom.GetProp("_CIPCode").upper()


def get_cip_one_hot(
    atom: RDKitAtom,
    config: AtomicFeaturesConfig | None = None,
) -> tuple[int, ...]:
    """Encode an atom's assigned CIP label using the configured vocabulary.

    RDKit assigns CIP labels at molecule level. The helper therefore makes
    a temporary molecule copy, assigns stereochemistry there, and reads the
    corresponding atom's _CIPCode property. Missing labels map to
    UNSPECIFIED. The output follows config.cip_types exactly.
    """
    if not isinstance(atom, RDKitAtom):
        raise TypeError("atom must be an RDKit Atom instance.")

    if config is None:
        config = AtomicFeaturesConfig()
    elif not isinstance(config, AtomicFeaturesConfig):
        raise TypeError("config must be an AtomicFeaturesConfig instance or None.")

    if not config.include_cip:
        return ()

    category = _get_cip_category(atom)
    if category not in config.cip_types:
        raise ValueError(
            f"CIP category {category!r} is not present in config.cip_types."
        )

    return tuple(
        1 if value == category else 0
        for value in config.cip_types
    )



def build_atom_features(
    mol: Chem.Mol,
    config: AtomicFeaturesConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build atomic numbers and the ordered chemical feature matrix.

    Atoms are processed in their existing RDKit index order. The returned
    arrays therefore preserve the correspondence between z[i], x_chem[i],
    and mol.GetAtomWithIdx(i).

    The molecule is consumed as supplied: this function does not add,
    remove, or reorder atoms or hydrogens. Callers must apply the shared
    G7 hydrogen policy before invoking it.

    Parameters
    ----------
    mol:
        RDKit molecule whose atoms are to be encoded.
    config:
        Optional atom-feature configuration. When omitted, the default
        AtomicFeaturesConfig is used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        z with shape [N] and dtype int64, followed by x_chem with shape
        [N, F_chem] and dtype float32. The columns of x_chem follow
        get_atom_feature_names() exactly.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an RDKit Mol instance.")

    # Resolve/validate the configuration and obtain the authoritative width.
    feature_names = get_atom_feature_names(config)
    if config is None:
        config = AtomicFeaturesConfig()

    num_atoms = mol.GetNumAtoms()
    num_features = len(feature_names)
    z = np.empty((num_atoms,), dtype=np.int64)
    x_chem = np.empty((num_atoms, num_features), dtype=np.float32)

    for atom_index in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_index)
        features = (
            get_atom_scalar_features(atom, config)
            + get_hybridization_one_hot(atom, config)
            + get_chirality_one_hot(atom, config)
            + get_cip_one_hot(atom, config)
        )

        if len(features) != num_features:
            raise RuntimeError(
                "Atom feature encoders returned a row whose width does not "
                "match get_atom_feature_names()."
            )

        z[atom_index] = atom.GetAtomicNum()
        if num_features:
            x_chem[atom_index] = features

    if not np.isfinite(x_chem).all():
        raise ValueError("x_chem contains NaN or infinite values.")

    return z, x_chem
