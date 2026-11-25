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

    Parameters
    ----------
    mol:
        RDKit molecule. Its atom and bond ordering are consumed as supplied.

    Returns
    -------
    numpy.ndarray
        Integer edge index with shape [2, num_bonds] and dtype int64.
        A molecule without bonds returns an array with shape [2, 0].
    """
    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an RDKit Mol instance.")

    edge_index = np.empty((2, mol.GetNumBonds()), dtype=np.int64)
    for bond_index, bond in enumerate(mol.GetBonds()):
        edge_index[0, bond_index] = bond.GetBeginAtomIdx()
        edge_index[1, bond_index] = bond.GetEndAtomIdx()

    return edge_index



def build_edge_features(
    mol: Chem.Mol,
    conf_id: int | None,
    config: GraphConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the currently implemented covalent edge representation.

    T3.1 supports only RDKit bond topology. Spatial radius edges and the
    hybrid combination are intentionally left to their separately scoped
    tasks. A zero-width edge-attribute matrix preserves row alignment until
    chemical bond attributes are implemented.

    conf_id is accepted to keep the public edge-builder interface aligned
    with the architecture contract; covalent connectivity does not depend
    on conformer coordinates.
    """
    if not isinstance(config, GraphConfig):
        raise TypeError("config must be a GraphConfig instance.")
    if config.edge_mode != "covalent":
        raise NotImplementedError(
            "Only edge_mode='covalent' is implemented in T3.1; "
            "radius and hybrid modes belong to subsequent tasks."
        )

    edge_index = build_covalent_edge_index(mol)
    edge_attr = np.empty((edge_index.shape[1], 0), dtype=np.float32)
    return edge_index, edge_attr
