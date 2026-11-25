"""RDKit-derived molecular edge topology for G7."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from .config import GraphConfig


def build_covalent_edge_index(mol: Chem.Mol) -> np.ndarray:
    """Return one endpoint pair for each RDKit bond, in RDKit bond order.

    The two rows contain each bond's begin-atom and end-atom indices,
    respectively. Atom indices are kept unchanged, so column j refers
    to mol.GetBondWithIdx(j).

    This is the raw covalent topology only: it does not add reverse edges,
    spatial neighbors, self-loops, or chemical/geometric edge attributes.
    Those are handled by their separately scoped edge-feature tasks.
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an RDKit Mol instance.")

    edge_index = np.empty((2, mol.GetNumBonds()), dtype=np.int64)
    for bond_index, bond in enumerate(mol.GetBonds()):
        edge_index[0, bond_index] = bond.GetBeginAtomIdx()
        edge_index[1, bond_index] = bond.GetEndAtomIdx()

    return edge_index


def build_radius_edge_index(
    mol: Chem.Mol,
    conf_id: int,
    cutoff: float,
) -> np.ndarray:
    """Return spatial neighbor pairs whose 3D distance is <= cutoff.

    The selected RDKit conformer is addressed by its actual conf_id.
    Each unordered atom pair is emitted once, oriented by ascending atom
    index (i < j). Reverse-edge expansion and merging/deduplication with
    covalent edges are deliberately left to later edge tasks.

    Parameters
    ----------
    mol:
        RDKit molecule whose atom indices define graph node indices.
    conf_id:
        Actual RDKit conformer identifier to use for all coordinates.
    cutoff:
        Positive finite distance threshold in Angstrom.

    Returns
    -------
    numpy.ndarray
        Integer edge index with shape [2, E] and dtype int64. If no pair
        satisfies the cutoff, the result has shape [2, 0].
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an RDKit Mol instance.")
    if isinstance(conf_id, (bool, np.bool_)) or not isinstance(
        conf_id, (int, np.integer)
    ):
        raise ValueError("conf_id must be an integer RDKit conformer ID.")
    if isinstance(cutoff, (bool, np.bool_)):
        raise ValueError("cutoff must be a finite positive number.")
    try:
        cutoff_value = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be a finite positive number.") from exc
    if not np.isfinite(cutoff_value) or cutoff_value <= 0.0:
        raise ValueError("cutoff must be a finite positive number.")

    try:
        conformer = mol.GetConformer(int(conf_id))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Molecule does not contain conformer with conf_id={conf_id}."
        ) from exc

    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    expected_shape = (mol.GetNumAtoms(), 3)
    if positions.shape != expected_shape:
        raise ValueError(
            "Conformer coordinates must have shape "
            f"{expected_shape}; got {positions.shape}."
        )
    if not np.isfinite(positions).all():
        raise ValueError("Conformer coordinates contain NaN or Inf.")

    atom_count = mol.GetNumAtoms()
    if atom_count < 2:
        return np.empty((2, 0), dtype=np.int64)

    source, target = np.triu_indices(atom_count, k=1)
    displacement = positions[source] - positions[target]
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(displacement, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise conformer distances contain NaN or Inf.")

    within_cutoff = distances <= cutoff_value
    return np.vstack((source[within_cutoff], target[within_cutoff])).astype(
        np.int64,
        copy=False,
    )


def build_edge_features(
    mol: Chem.Mol,
    conf_id: int | None,
    config: GraphConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the edge topology supported by the current G7 edge tasks.

    Covalent mode reads RDKit bonds; radius mode reads spatial neighbors
    from the selected conformer. Hybrid composition and non-empty chemical
    or geometric edge attributes belong to subsequent tasks.
    """
    if not isinstance(config, GraphConfig):
        raise TypeError("config must be a GraphConfig instance.")

    if config.edge_mode == "covalent":
        edge_index = build_covalent_edge_index(mol)
    elif config.edge_mode == "radius":
        if conf_id is None:
            raise ValueError("conf_id is required for edge_mode='radius'.")
        edge_index = build_radius_edge_index(mol, conf_id, config.cutoff)
    else:
        raise NotImplementedError(
            "edge_mode='hybrid' is not implemented yet; "
            "it belongs to T3.3."
        )

    edge_attr = np.empty((edge_index.shape[1], 0), dtype=np.float32)
    return edge_index, edge_attr
