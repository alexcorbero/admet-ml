"""Build PyTorch Geometric graphs from RDKit molecular conformers.

Node inputs:
    z       atomic numbers [n_atoms]
    x_chem  chemical atom features [n_atoms, n_chem_features]
    x_dg    G6 density/curvature features [n_atoms, n_dg_features]
    x       concatenation of x_chem and x_dg
    pos     3D coordinates [n_atoms, 3]

Edge inputs:
    edge_index [2, n_edges]
    edge_attr  [n_edges, n_edge_features]

Global molecular descriptors are intentionally not included. They can be
provided to a separate model branch later (e.g. a Transformer).
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem

from .config import GNNConfig
from .atom_features import build_atom_features, get_atom_feature_names
from .edge_features import build_edge_features, get_edge_feature_names
from .density_features import build_density_features, get_density_feature_names


def get_node_feature_names(config: GNNConfig) -> list[str]:
    """Return the ordered names of columns in Data.x.

    Atomic numbers are stored separately in Data.z and are therefore not
    included in this list.
    """
    return (
        get_atom_feature_names()
        + get_density_feature_names(config)
    )


def build_graph(
    mol: Chem.Mol,
    conf_id: int,
    config: GNNConfig,
    y: float | None = None,
    smiles: str | None = None,
    mol_id: str | int | None = None,
) -> Data:
    """Build one PyTorch Geometric graph from an RDKit molecule.

    Parameters
    ----------
    mol:
        RDKit molecule containing the conformer used for this graph.
        Atom order must be identical across all feature-generation modules.
    conf_id:
        ID of the conformer used for coordinates and geometric features.
    config:
        GNN configuration shared by atom, edge and density feature modules.
    y:
        Optional scalar regression target (e.g. experimental logS).
    smiles:
        Optional SMILES string retained as graph metadata.
    mol_id:
        Optional dataset identifier retained as graph metadata.

    Returns
    -------
    torch_geometric.data.Data
        A molecular graph with node features, 3D positions, directed edges
        and optional target/metadata.

    Notes
    -----
    `x` contains chemical and G6-derived node features. `z` is kept
    separately for a learned atomic-number embedding in the GNN model.
    `x_chem` and `x_dg` are also retained separately to make feature
    ablations and debugging easier.

    No feature standardization is fitted here. Any scaler must be fitted
    using training graphs only and then applied to validation/test graphs.
    """
    z, x_chem = build_atom_features(mol)
    x_dg = build_density_features(mol, conf_id, config)
    edge_index, edge_attr = build_edge_features(mol, conf_id, config)

    pos = np.asarray(
        mol.GetConformer(int(conf_id)).GetPositions(),
        dtype=np.float32,
    )

    x = np.concatenate((x_chem, x_dg), axis=1).astype(np.float32)

    data = Data(
        z=torch.from_numpy(np.ascontiguousarray(z)).long(),
        x=torch.from_numpy(np.ascontiguousarray(x)).float(),
        x_chem=torch.from_numpy(np.ascontiguousarray(x_chem)).float(),
        x_dg=torch.from_numpy(np.ascontiguousarray(x_dg)).float(),
        pos=torch.from_numpy(np.ascontiguousarray(pos)).float(),
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)).long(),
        edge_attr=torch.from_numpy(np.ascontiguousarray(edge_attr)).float(),
        num_nodes=mol.GetNumAtoms(),
    )

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)

    if smiles is not None:
        data.smiles = smiles

    if mol_id is not None:
        data.mol_id = str(mol_id)

    data.conf_id = torch.tensor([int(conf_id)], dtype=torch.long)

    return data


def get_graph_feature_names(config: GNNConfig) -> dict[str, list[str]]:
    """Return ordered feature names for nodes and edges."""
    return {
        "x": get_node_feature_names(config),
        "x_chem": get_atom_feature_names(),
        "x_dg": get_density_feature_names(config),
        "edge_attr": get_edge_feature_names(config),
    }
