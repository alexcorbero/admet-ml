"""Feature-schema composition and PyG graph assembly for G7."""

from __future__ import annotations

import numpy as np

from .config import GNNConfig


def get_node_feature_names(config: GNNConfig) -> tuple[str, ...]:
    """Return the ordered column names for the concatenated ``Data.x``.

    ``Data.z`` is intentionally excluded: atomic numbers are stored in their
    own field and are embedded independently by the model. The returned
    order mirrors the graph-builder contract, ``x_chem`` followed by
    ``x_dg``.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    from .atom_features import get_atom_feature_names
    from .density_features import get_density_feature_names

    return (
        *get_atom_feature_names(config.atomic_features),
        *get_density_feature_names(config.atomic_features),
    )


def get_graph_feature_names(
    config: GNNConfig,
) -> dict[str, tuple[str, ...]]:
    """Return ordered schemas for the graph feature tensors.

    The mapping covers the feature-bearing tensors composed by the graph
    builder. Connectivity (``edge_index``), coordinates (``pos``), targets,
    and metadata do not have feature columns and are therefore not included.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    from .atom_features import get_atom_feature_names
    x_chem_names = get_atom_feature_names(config.atomic_features)
    from .density_features import get_density_feature_names
    x_dg_names = get_density_feature_names(config.atomic_features)
    from .edge_features import get_edge_feature_names

    return {
        "x": (*x_chem_names, *x_dg_names),
        "x_chem": x_chem_names,
        "x_dg": x_dg_names,
        "edge_attr": get_edge_feature_names(config.graph),
    }


def build_graph(
    mol,
    conf_id: int,
    config: GNNConfig,
    y: float | None = None,
    smiles: str | None = None,
    mol_id: str | int | None = None,
):
    """Build the raw PyTorch Geometric graph tensors for one molecule.

    This task composes the already-scoped atom, density, and edge builders.
    ``y`` and the optional identifiers are attached without changing any
    node or edge feature tensor. The selected conformer ID is always stored
    so the graph representation remains reproducible.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    import torch
    from torch_geometric.data import Data

    from .atom_features import build_atom_features
    from .density_features import build_density_features
    from .edge_features import build_edge_features
    from src.features.descriptors_3d import get_coordinates

    z, x_chem = build_atom_features(mol, config.atomic_features)
    x_dg = build_density_features(mol, conf_id, config.atomic_features)
    edge_index, edge_attr = build_edge_features(mol, conf_id, config.graph)
    pos = np.asarray(get_coordinates(mol, conf_id), dtype=np.float32)

    x = np.concatenate((x_chem, x_dg), axis=1).astype(
        np.float32,
        copy=False,
    )

    data = Data(
        z=torch.from_numpy(np.ascontiguousarray(z)).long(),
        x=torch.from_numpy(np.ascontiguousarray(x)).float(),
        x_chem=torch.from_numpy(np.ascontiguousarray(x_chem)).float(),
        x_dg=torch.from_numpy(np.ascontiguousarray(x_dg)).float(),
        pos=torch.from_numpy(np.ascontiguousarray(pos)).float(),
        edge_index=torch.from_numpy(
            np.ascontiguousarray(edge_index)
        ).long(),
        edge_attr=torch.from_numpy(
            np.ascontiguousarray(edge_attr)
        ).float(),
        num_nodes=int(mol.GetNumAtoms()),
    )

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    if smiles is not None:
        data.smiles = smiles
    if mol_id is not None:
        data.mol_id = str(mol_id)
    data.conf_id = torch.tensor([int(conf_id)], dtype=torch.long)

    return data
