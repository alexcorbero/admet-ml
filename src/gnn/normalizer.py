"""Column schema and train-only fitting for G7 graph-feature normalization.

This module defines which raw feature columns are eligible for scaling and
fits their train statistics. It does not transform graphs, handle curvature
masks, or persist statistics; those responsibilities belong to later tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .config import GNNConfig
from .density_features import get_density_feature_names


_SCALABLE_ATOM_SCALARS: Final[tuple[tuple[str, str], ...]] = (
    ("include_formal_charge", "formal_charge"),
    ("include_heavy_atom_degree", "heavy_atom_degree"),
    ("include_attached_hydrogens", "attached_hydrogens"),
    ("include_atomic_mass", "atomic_mass"),
)

_SCALABLE_DG_PREFIXES: Final[tuple[str, ...]] = (
    "rho_",
    "H_",
    "K_",
    "kappa_min_",
    "kappa_max_",
)

_ALL_ATOM_SCALARS: Final[tuple[tuple[str, str], ...]] = (
    *_SCALABLE_ATOM_SCALARS,
    ("include_aromaticity", "aromaticity"),
    ("include_ring_membership", "ring_membership"),
)
_BOND_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "is_bonded",
    "bond_type_UNSPECIFIED",
    "bond_type_SINGLE",
    "bond_type_DOUBLE",
    "bond_type_TRIPLE",
    "bond_type_QUADRUPLE",
    "bond_type_QUINTUPLE",
    "bond_type_HEXTUPLE",
    "bond_type_ONEANDAHALF",
    "bond_type_TWOANDAHALF",
    "bond_type_THREEANDAHALF",
    "bond_type_FOURANDAHALF",
    "bond_type_FIVEANDAHALF",
    "bond_type_AROMATIC",
    "bond_type_IONIC",
    "bond_type_HYDROGEN",
    "bond_type_THREECENTER",
    "bond_type_DATIVEONE",
    "bond_type_DATIVE",
    "bond_type_DATIVEL",
    "bond_type_DATIVER",
    "bond_type_ZERO",
    "bond_is_aromatic",
    "bond_is_conjugated",
    "bond_is_in_ring",
)


@dataclass(frozen=True)
class FeatureStatistics:
    """Statistics fitted for one tensor group and its selected columns."""

    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    count: int


@dataclass
class GraphFeatureNormalizer:
    """Train-fitted statistics for the columns selected by T7.1.

    ``fit`` is the only operation implemented here. It consumes the supplied
    graphs as training data, stores statistics, and returns ``self``. It never
    reads targets or mutates the graph objects.
    """

    config: GNNConfig
    columns: dict[str, tuple[str, ...]] = field(init=False)
    statistics: dict[str, FeatureStatistics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, GNNConfig):
            raise TypeError("config must be a GNNConfig instance.")
        self.columns = get_scalable_feature_columns(self.config)

    @classmethod
    def from_config(cls, config: GNNConfig) -> "GraphFeatureNormalizer":
        """Create an unfitted normalizer for ``config``."""
        return cls(config=config)

    @property
    def fitted(self) -> bool:
        """Whether train statistics have been fitted."""
        return bool(self.statistics)

    def fit(self, train_graphs) -> "GraphFeatureNormalizer":
        """Fit means and population standard deviations on ``train_graphs``.

        Statistics are computed independently for node and edge tensors. A
        configured group must provide at least one finite observation across
        the training graphs. Curvature validity masks are not special-cased
        here because that behavior is assigned to T7.3.
        """
        if isinstance(train_graphs, (str, bytes)):
            raise TypeError("train_graphs must be an iterable of graph objects.")
        try:
            graphs = list(train_graphs)
        except TypeError as exc:
            raise TypeError("train_graphs must be an iterable of graph objects.") from exc
        if not graphs:
            raise ValueError("train_graphs must contain at least one graph.")

        full_columns = _get_full_feature_columns(self.config)
        values_by_group: dict[str, list[np.ndarray]] = {
            group: [] for group, names in self.columns.items() if names
        }

        for graph_index, graph in enumerate(graphs):
            for group, selected_names in self.columns.items():
                if not selected_names:
                    continue
                values = _as_float_matrix(
                    getattr(graph, group, None),
                    group=group,
                    graph_index=graph_index,
                )
                expected_names = full_columns[group]
                if values.shape[1] != len(expected_names):
                    raise ValueError(
                        f"graph {graph_index} field {group!r} has {values.shape[1]} "
                        f"columns; expected {len(expected_names)}."
                    )
                selected_indices = [expected_names.index(name) for name in selected_names]
                selected_values = values[:, selected_indices]
                if selected_values.shape[0]:
                    if not np.isfinite(selected_values).all():
                        raise ValueError(
                            f"graph {graph_index} field {group!r} contains "
                            "NaN or infinite values."
                        )
                    values_by_group[group].append(selected_values)

        fitted: dict[str, FeatureStatistics] = {}
        for group, selected_names in self.columns.items():
            if not selected_names:
                continue
            if not values_by_group[group]:
                raise ValueError(f"no observations available for {group!r}.")
            values = np.concatenate(values_by_group[group], axis=0).astype(
                np.float64,
                copy=False,
            )
            fitted[group] = FeatureStatistics(
                columns=selected_names,
                mean=np.mean(values, axis=0),
                std=np.std(values, axis=0, ddof=0),
                count=int(values.shape[0]),
            )
        self.statistics = fitted
        return self


def get_scalable_feature_columns(
    config: GNNConfig,
) -> dict[str, tuple[str, ...]]:
    """Return the ordered raw columns eligible for numeric scaling.

    The result is grouped by the tensors that own the columns. ``x`` is not
    returned because it is the concatenation of ``x_chem`` and ``x_dg``;
    later transforms can update those source tensors and rebuild ``x``.

    Included columns are continuous or count-like numeric values:

    * scalar chemical atom features in ``x_chem``;
    * density and curvature values in ``x_dg``;
    * distance and Gaussian RBF columns in ``edge_attr``.

    Atomic numbers, one-hot columns, binary flags, and ``curvature_valid``
    masks are deliberately excluded.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    atomic_config = config.atomic_features
    x_chem = tuple(
        feature_name
        for enabled_field, feature_name in _SCALABLE_ATOM_SCALARS
        if getattr(atomic_config, enabled_field)
    )

    density_names = get_density_feature_names(atomic_config)
    x_dg = tuple(
        feature_name
        for feature_name in density_names
        if feature_name.startswith(_SCALABLE_DG_PREFIXES)
    )

    edge_names: list[str] = []
    if config.graph.include_distance:
        edge_names.append("distance")
    if config.graph.include_rbf:
        edge_names.extend(
            f"gaussian_rbf_center_{index}"
            for index in range(config.graph.rbf_num_centers)
        )

    return {
        "x_chem": x_chem,
        "x_dg": x_dg,
        "edge_attr": tuple(edge_names),
    }


def get_scalable_feature_names(
    config: GNNConfig,
) -> dict[str, tuple[str, ...]]:
    """Compatibility alias for :func:`get_scalable_feature_columns`."""
    return get_scalable_feature_columns(config)


def _get_full_feature_columns(config: GNNConfig) -> dict[str, tuple[str, ...]]:
    """Return full tensor schemas without importing RDKit or PyTorch."""
    atomic_config = config.atomic_features
    x_chem: list[str] = [
        feature_name
        for enabled_field, feature_name in _ALL_ATOM_SCALARS
        if getattr(atomic_config, enabled_field)
    ]
    if atomic_config.include_hybridization:
        x_chem.extend(
            f"hybridization_{value}"
            for value in atomic_config.hybridization_types
        )
    if atomic_config.include_chirality:
        x_chem.extend(
            f"chirality_{value}" for value in atomic_config.chirality_types
        )
    if atomic_config.include_cip:
        x_chem.extend(f"cip_{value}" for value in atomic_config.cip_types)

    edge_attr = list(_BOND_FEATURE_NAMES) if config.graph.include_bond_features else []
    if config.graph.include_distance:
        edge_attr.append("distance")
    if config.graph.include_rbf:
        edge_attr.extend(
            f"gaussian_rbf_center_{index}"
            for index in range(config.graph.rbf_num_centers)
        )
    return {
        "x_chem": tuple(x_chem),
        "x_dg": get_density_feature_names(atomic_config),
        "edge_attr": tuple(edge_attr),
    }


def _as_float_matrix(value, *, group: str, graph_index: int) -> np.ndarray:
    """Convert a graph tensor/array to a finite-checkable 2-D float matrix."""
    if value is None:
        raise ValueError(
            f"graph {graph_index} is missing required field {group!r}."
        )
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"graph {graph_index} field {group!r} must be numeric."
        ) from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"graph {graph_index} field {group!r} must be a 2-D matrix; "
            f"got shape {matrix.shape}."
        )
    return matrix
