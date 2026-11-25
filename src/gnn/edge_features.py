"""RDKit-derived molecular edge topology for G7."""

from __future__ import annotations

import numpy as np
from rdkit import Chem


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
