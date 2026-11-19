"""Training-only normalization for molecular GNN graph features."""

import numpy as np
import torch


CHEMICAL_FEATURES_TO_SCALE = (
    "formal_charge",
    "heavy_atom_degree",
    "total_attached_hydrogens",
    "atomic_mass",
)

CURVATURE_SUFFIXES = (
    "_H",
    "_K",
    "_kappa_min",
    "_kappa_max",
)


def _fit_stats(
    values: np.ndarray,
    valid: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit column-wise population mean/std, optionally using a validity mask."""
    if valid is None:
        valid = np.ones(values.shape, dtype=bool)

    count = valid.sum(axis=0)
    total = np.where(valid, values, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=count > 0,
    )

    squared_deviation = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0)
    variance = np.divide(
        squared_deviation,
        count,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=count > 0,
    )

    scale = np.sqrt(variance)
    scale[scale < epsilon] = 1.0

    return mean, scale


def _get_curvature_mask_pairs(
    feature_names: list[str],
) -> list[tuple[int, int]]:
    """Return (curvature column, matching validity-mask column) pairs."""
    name_to_index = {name: i for i, name in enumerate(feature_names)}
    pairs = []

    for feature_index, name in enumerate(feature_names):
        for suffix in CURVATURE_SUFFIXES:
            if name.endswith(suffix):
                prefix = name[:-len(suffix)]
                mask_index = name_to_index.get(f"{prefix}_valid")
                if mask_index is not None:
                    pairs.append((feature_index, mask_index))
                break

    return pairs


class GraphFeatureNormalizer:
    """Standardize continuous node and edge features using training graphs.

    Scaled:
        - chemical: formal charge, heavy-atom degree, H count, atomic mass
        - G6: density and curvature values
        - edges: raw distance in angstroms

    Unchanged:
        - atomic numbers, binary/one-hot features and validity masks
        - RBFs, radial kernels, coordinates and regression targets

    Curvatures are fitted/transformed only where their matching ``*_valid``
    mask is 1. Invalid curvature values therefore remain exactly zero.
    """

    def __init__(
        self,
        chemical_feature_names: list[str],
        density_feature_names: list[str],
        edge_feature_names: list[str],
        epsilon: float = 1e-8,
    ):
        self.chemical_feature_names = list(chemical_feature_names)
        self.density_feature_names = list(density_feature_names)
        self.edge_feature_names = list(edge_feature_names)
        self.epsilon = epsilon

        self._set_feature_indices()

        self.chem_mean = None
        self.chem_scale = None
        self.dg_mean = None
        self.dg_scale = None
        self.edge_mean = None
        self.edge_scale = None
        self.is_fitted = False

    def _set_feature_indices(self) -> None:
        self.chem_indices = [
            self.chemical_feature_names.index(name)
            for name in CHEMICAL_FEATURES_TO_SCALE
            if name in self.chemical_feature_names
        ]

        self.dg_indices = [
            i for i, name in enumerate(self.density_feature_names)
            if not name.endswith("_valid")
        ]

        self.edge_indices = [
            i for i, name in enumerate(self.edge_feature_names)
            if name == "distance_angstrom"
        ]

        self.curvature_mask_pairs = [
            (feature_index, mask_index)
            for feature_index, mask_index in _get_curvature_mask_pairs(
                self.density_feature_names
            )
            if feature_index in self.dg_indices
        ]

    @classmethod
    def from_config(cls, config, epsilon: float = 1e-8):
        """Build the normalizer using the feature schema from GNNConfig."""
        from .atom_features import get_atom_feature_names
        from .density_features import get_density_feature_names
        from .edge_features import get_edge_feature_names

        return cls(
            chemical_feature_names=get_atom_feature_names(),
            density_feature_names=get_density_feature_names(config),
            edge_feature_names=get_edge_feature_names(config),
            epsilon=epsilon,
        )

    def fit(self, train_graphs: list) -> "GraphFeatureNormalizer":
        """Fit statistics using training graphs only."""
        x_chem = np.concatenate([
            graph.x_chem.detach().cpu().numpy()
            for graph in train_graphs
        ], axis=0)
        x_dg = np.concatenate([
            graph.x_dg.detach().cpu().numpy()
            for graph in train_graphs
        ], axis=0)
        edge_attr = np.concatenate([
            graph.edge_attr.detach().cpu().numpy()
            for graph in train_graphs
        ], axis=0)

        self.chem_mean, self.chem_scale = _fit_stats(
            x_chem[:, self.chem_indices],
            epsilon=self.epsilon,
        )

        dg_values = x_dg[:, self.dg_indices]
        dg_valid = np.ones(dg_values.shape, dtype=bool)
        dg_local_indices = {
            feature_index: local_index
            for local_index, feature_index in enumerate(self.dg_indices)
        }

        for feature_index, mask_index in self.curvature_mask_pairs:
            local_index = dg_local_indices[feature_index]
            dg_valid[:, local_index] = x_dg[:, mask_index] > 0.5

        self.dg_mean, self.dg_scale = _fit_stats(
            dg_values,
            valid=dg_valid,
            epsilon=self.epsilon,
        )

        self.edge_mean, self.edge_scale = _fit_stats(
            edge_attr[:, self.edge_indices],
            epsilon=self.epsilon,
        )

        self.is_fitted = True
        return self

    @staticmethod
    def _transform_columns(
        values: torch.Tensor,
        indices: list[int],
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        if indices:
            mean_tensor = torch.as_tensor(
                mean, dtype=values.dtype, device=values.device
            )
            scale_tensor = torch.as_tensor(
                scale, dtype=values.dtype, device=values.device
            )
            values[:, indices] = (values[:, indices] - mean_tensor) / scale_tensor

    def transform_graph(self, graph):
        """Return a normalized copy of one PyTorch Geometric Data object."""
        normalized = graph.clone()

        x_chem = normalized.x_chem.clone()
        x_dg = normalized.x_dg.clone()
        edge_attr = normalized.edge_attr.clone()

        self._transform_columns(
            x_chem, self.chem_indices, self.chem_mean, self.chem_scale
        )

        if self.dg_indices:
            mean_tensor = torch.as_tensor(
                self.dg_mean, dtype=x_dg.dtype, device=x_dg.device
            )
            scale_tensor = torch.as_tensor(
                self.dg_scale, dtype=x_dg.dtype, device=x_dg.device
            )
            dg_values = (x_dg[:, self.dg_indices] - mean_tensor) / scale_tensor

            dg_local_indices = {
                feature_index: local_index
                for local_index, feature_index in enumerate(self.dg_indices)
            }
            for feature_index, mask_index in self.curvature_mask_pairs:
                local_index = dg_local_indices[feature_index]
                valid = x_dg[:, mask_index] > 0.5
                dg_values[:, local_index] = torch.where(
                    valid,
                    dg_values[:, local_index],
                    torch.zeros_like(dg_values[:, local_index]),
                )

            x_dg[:, self.dg_indices] = dg_values

        self._transform_columns(
            edge_attr, self.edge_indices, self.edge_mean, self.edge_scale
        )

        normalized.x_chem = x_chem
        normalized.x_dg = x_dg
        normalized.edge_attr = edge_attr
        normalized.x = torch.cat((x_chem, x_dg), dim=1)

        return normalized

    def transform(self, graphs: list) -> list:
        """Return normalized copies of a list of PyG graphs."""
        return [self.transform_graph(graph) for graph in graphs]

    def fit_transform(self, train_graphs: list) -> list:
        """Fit on training graphs and return their normalized copies."""
        return self.fit(train_graphs).transform(train_graphs)

    def state_dict(self) -> dict:
        """Return a serializable state to save alongside the model."""
        return {
            "chemical_feature_names": self.chemical_feature_names,
            "density_feature_names": self.density_feature_names,
            "edge_feature_names": self.edge_feature_names,
            "epsilon": self.epsilon,
            "chem_mean": self.chem_mean.tolist(),
            "chem_scale": self.chem_scale.tolist(),
            "dg_mean": self.dg_mean.tolist(),
            "dg_scale": self.dg_scale.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_scale": self.edge_scale.tolist(),
        }

    def load_state_dict(self, state: dict) -> "GraphFeatureNormalizer":
        """Restore a previously fitted normalizer."""
        self.chemical_feature_names = state["chemical_feature_names"]
        self.density_feature_names = state["density_feature_names"]
        self.edge_feature_names = state["edge_feature_names"]
        self.epsilon = state["epsilon"]

        self._set_feature_indices()

        self.chem_mean = np.asarray(state["chem_mean"], dtype=np.float64)
        self.chem_scale = np.asarray(state["chem_scale"], dtype=np.float64)
        self.dg_mean = np.asarray(state["dg_mean"], dtype=np.float64)
        self.dg_scale = np.asarray(state["dg_scale"], dtype=np.float64)
        self.edge_mean = np.asarray(state["edge_mean"], dtype=np.float64)
        self.edge_scale = np.asarray(state["edge_scale"], dtype=np.float64)
        self.is_fitted = True

        return self
