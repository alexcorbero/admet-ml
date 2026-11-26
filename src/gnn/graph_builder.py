"""Feature-schema composition for molecular graph inputs in G7."""

from __future__ import annotations

from .atom_features import get_atom_feature_names
from .config import GNNConfig
from .density_features import get_density_feature_names
from .edge_features import get_edge_feature_names


def get_node_feature_names(config: GNNConfig) -> tuple[str, ...]:
    """Return the ordered column names for the concatenated ``Data.x``.

    ``Data.z`` is intentionally excluded: atomic numbers are stored in their
    own field and are embedded independently by the model. The returned
    order mirrors the graph-builder contract, ``x_chem`` followed by
    ``x_dg``.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

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

    x_chem_names = get_atom_feature_names(config.atomic_features)
    x_dg_names = get_density_feature_names(config.atomic_features)
    return {
        "x": (*x_chem_names, *x_dg_names),
        "x_chem": x_chem_names,
        "x_dg": x_dg_names,
        "edge_attr": get_edge_feature_names(config.graph),
    }
