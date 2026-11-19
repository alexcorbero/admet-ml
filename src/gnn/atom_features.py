"""Chemical atom features for the G7 molecular GNN."""

import numpy as np
from rdkit import Chem


HYBRIDIZATION_CATEGORIES = (
    "SP", "SP2", "SP3", "SP3D",
    "SP3D2", "UNSPECIFIED", "OTHER",
)

CHIRALITY_CATEGORIES = (
    "UNASSIGNED", "R", "S", "OTHER",
)

ATOM_FEATURE_NAMES = (
    "formal_charge",
    "heavy_atom_degree",
    "total_attached_hydrogens",
    "atomic_mass",
    "is_aromatic",
    "is_in_ring",
    *(
        f"hybridization_{name.lower()}"
        for name in HYBRIDIZATION_CATEGORIES
    ),
    *(
        f"chirality_{name.lower()}"
        for name in CHIRALITY_CATEGORIES
    ),
)


def get_atom_feature_names() -> list[str]:
    """Return ordered chemical feature names."""
    return list(ATOM_FEATURE_NAMES)


def _one_hot(
    value: str,
    categories: tuple[str, ...],
) -> list[float]:

    return [
        float(value == category)
        for category in categories
    ]


def _get_hybridization(atom: Chem.Atom) -> str:

    value = str(atom.GetHybridization())

    return (
        value
        if value in HYBRIDIZATION_CATEGORIES
        else "OTHER"
    )


def _get_chirality(atom: Chem.Atom) -> str:

    if atom.HasProp("_CIPCode"):

        code = atom.GetProp("_CIPCode")

        return (
            code
            if code in ("R", "S")
            else "OTHER"
        )

    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return "OTHER"

    return "UNASSIGNED"


def _total_attached_hydrogens(atom: Chem.Atom) -> int:

    explicit_h_neighbors = sum(
        neighbor.GetAtomicNum() == 1
        for neighbor in atom.GetNeighbors()
    )

    return (
        int(atom.GetTotalNumHs(includeNeighbors=False))
        + explicit_h_neighbors
    )


def _atom_feature_vector(atom: Chem.Atom) -> list[float]:

    heavy_degree = sum(
        neighbor.GetAtomicNum() > 1
        for neighbor in atom.GetNeighbors()
    )

    features = [
        float(atom.GetFormalCharge()),
        float(heavy_degree),
        float(_total_attached_hydrogens(atom)),
        float(atom.GetMass()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]

    features.extend(
        _one_hot(
            _get_hybridization(atom),
            HYBRIDIZATION_CATEGORIES,
        )
    )

    features.extend(
        _one_hot(
            _get_chirality(atom),
            CHIRALITY_CATEGORIES,
        )
    )

    return features


def build_atom_features(
    mol: Chem.Mol,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build chemical node features.

    Returns
    -------
    z : np.ndarray
        Atomic numbers, shape (n_atoms,).

    x_chem : np.ndarray
        Chemical features, shape (n_atoms, 17).
    """

    working_mol = Chem.Mol(mol)

    Chem.AssignStereochemistry(
        working_mol,
        cleanIt=True,
        force=True,
    )

    z = np.asarray(
        [
            atom.GetAtomicNum()
            for atom in working_mol.GetAtoms()
        ],
        dtype=np.int64,
    )

    x_chem = np.asarray(
        [
            _atom_feature_vector(atom)
            for atom in working_mol.GetAtoms()
        ],
        dtype=np.float32,
    )

    return z, x_chem