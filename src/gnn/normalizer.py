"""Column schema and train-only fitting for G7 graph-feature normalization.

This module defines which raw feature columns are eligible for scaling and
fits their train statistics. It does not transform graphs or persist
statistics; those responsibilities belong to later tasks.
"""

from __future__ import annotations

from copy import copy
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
    # A scalar is retained for unmasked groups for compatibility. DG groups
    # can have a different number of valid rows per curvature column.
    count: int | np.ndarray


@dataclass
class GraphFeatureNormalizer:
    """Train-fitted statistics for the columns selected by T7.1.

    ``fit`` consumes the supplied graphs as training data and stores
    statistics. Transform methods return copied graphs and never mutate the
    input objects.
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
        the training graphs. Curvature values only contribute when their
        corresponding ``curvature_valid`` mask is active.
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
        values_by_group: dict[str, list[list[np.ndarray]]] = {
            group: [[] for _ in names]
            for group, names in self.columns.items()
            if names
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
                for column_index, column_name in enumerate(selected_names):
                    column_values = selected_values[:, column_index]
                    if group == "x_dg" and _is_curvature_value(column_name):
                        mask_name = _curvature_mask_name(column_name)
                        mask_index = expected_names.index(mask_name)
                        mask_values = values[:, mask_index]
                        if not np.isfinite(mask_values).all() or not np.isin(
                            mask_values, (0.0, 1.0)
                        ).all():
                            raise ValueError(
                                f"graph {graph_index} field {group!r} has an "
                                f"invalid {mask_name!r} mask; expected finite "
                                "binary values."
                            )
                        column_values = column_values[mask_values.astype(bool)]
                    if not np.isfinite(column_values).all():
                        raise ValueError(
                            f"graph {graph_index} field {group!r} contains "
                            "NaN or infinite values."
                        )
                    if column_values.size:
                        values_by_group[group][column_index].append(column_values)

        fitted: dict[str, FeatureStatistics] = {}
        for group, selected_names in self.columns.items():
            if not selected_names:
                continue
            column_values = []
            for column_name, observations in zip(
                selected_names, values_by_group[group]
            ):
                if not observations:
                    raise ValueError(
                        f"no valid observations available for {group!r} "
                        f"column {column_name!r}."
                    )
                column_values.append(
                    np.concatenate(observations, axis=0).astype(
                        np.float64,
                        copy=False,
                    )
                )
            means = np.array([np.mean(values) for values in column_values])
            stds = np.array([np.std(values, ddof=0) for values in column_values])
            counts = np.array([values.size for values in column_values], dtype=int)
            fitted[group] = FeatureStatistics(
                columns=selected_names,
                mean=means,
                std=stds,
                count=_collapse_counts(counts),
            )
        self.statistics = fitted
        return self

    def transform_graph(self, graph):
        """Return a copy of ``graph`` with fitted numeric columns scaled.

        Only columns recorded by :meth:`fit` are changed. Curvature masks are
        preserved, and invalid curvature values are left at their raw neutral
        value rather than being treated as observations or transformed.
        """
        self._require_fitted()
        transformed = copy(graph)

        for group, statistics in self.statistics.items():
            values = getattr(graph, group, None)
            output = _clone_feature_matrix(
                values,
                group=group,
            )
            full_columns = _get_full_feature_columns(self.config)[group]
            _transform_matrix(
                output,
                full_columns=full_columns,
                statistics=statistics,
                group=group,
            )
            setattr(transformed, group, output)

        if hasattr(transformed, "x_chem") and hasattr(transformed, "x_dg"):
            transformed.x = _concatenate_node_features(
                transformed.x_chem,
                transformed.x_dg,
            )
        return transformed

    def transform_graphs(self, graphs):
        """Return transformed copies for every graph in ``graphs``."""
        self._require_fitted()
        if isinstance(graphs, (str, bytes)):
            raise TypeError("graphs must be an iterable of graph objects.")
        try:
            return [self.transform_graph(graph) for graph in graphs]
        except TypeError as exc:
            raise TypeError("graphs must be an iterable of graph objects.") from exc

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("GraphFeatureNormalizer must be fitted before transforming.")


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


def _is_curvature_value(name: str) -> bool:
    """Return whether a DG column is a masked curvature value."""
    return name.startswith(("H_", "K_", "kappa_min_", "kappa_max_"))


def _curvature_mask_name(name: str) -> str:
    """Return the validity-mask column paired with one curvature value."""
    for prefix in ("H_", "K_", "kappa_min_", "kappa_max_"):
        if name.startswith(prefix):
            return f"curvature_valid_{name[len(prefix):]}"
    raise ValueError(f"not a curvature value column: {name!r}")


def _collapse_counts(counts: np.ndarray) -> int | np.ndarray:
    """Keep a scalar count when every selected column has the same count."""
    if np.all(counts == counts[0]):
        return int(counts[0])
    return counts


def _clone_feature_matrix(value, *, group: str):
    """Clone a graph matrix while retaining its tensor/array implementation."""
    if value is None:
        raise ValueError(f"graph is missing required field {group!r}.")
    if hasattr(value, "clone"):
        return value.clone()
    try:
        return np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"graph field {group!r} must be numeric.") from exc


def _transform_matrix(
    output,
    *,
    full_columns: tuple[str, ...],
    statistics: FeatureStatistics,
    group: str,
) -> None:
    """Apply one group's statistics in place to a cloned matrix."""
    if output.ndim != 2 or output.shape[1] != len(full_columns):
        raise ValueError(
            f"graph field {group!r} has shape {tuple(output.shape)}; expected "
            f"[N, {len(full_columns)}]."
        )
    scales = np.where(statistics.std > 0.0, statistics.std, 1.0)
    for statistic_index, column_name in enumerate(statistics.columns):
        column_index = full_columns.index(column_name)
        valid_rows = None
        if group == "x_dg" and _is_curvature_value(column_name):
            mask_index = full_columns.index(_curvature_mask_name(column_name))
            valid_rows = _column_as_numpy(output[:, mask_index]).astype(bool)
        values = _column_as_numpy(output[:, column_index])
        if valid_rows is None:
            valid_rows = np.ones(values.shape[0], dtype=bool)
        if not np.isfinite(values[valid_rows]).all():
            raise ValueError(
                f"graph field {group!r} contains NaN or infinite values in "
                f"column {column_name!r}."
            )
        normalized = (values - statistics.mean[statistic_index]) / scales[statistic_index]
        _assign_column_values(output, column_index, normalized, valid_rows)


def _column_as_numpy(values) -> np.ndarray:
    """Convert a one-dimensional tensor/array view for validation and masks."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _assign_column_values(output, column_index: int, values: np.ndarray, rows: np.ndarray) -> None:
    """Assign normalized values to either a NumPy or PyTorch matrix."""
    if hasattr(output, "new_tensor"):
        output[rows, column_index] = output.new_tensor(values[rows])
    else:
        output[rows, column_index] = values[rows]


def _concatenate_node_features(x_chem, x_dg):
    """Rebuild the public ``x`` field from its transformed source tensors."""
    if hasattr(x_chem, "new_tensor"):
        import torch

        return torch.cat((x_chem, x_dg), dim=1)
    return np.concatenate((x_chem, x_dg), axis=1)


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
