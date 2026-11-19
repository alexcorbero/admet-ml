"""
Chemical and geometric edge features for the G7 molecular GNN.

Returns directed edge_index and edge_attr arrays in RDKit atom order.
Supports covalent, radius and hybrid graphs.
"""

import numpy as np
from rdkit import Chem

from .config import GNNConfig


BOND_TYPES = ("single", "double", "triple", "aromatic", "other")

BOND_FEATURE_NAMES = (
    *(f"bond_type_{name}" for name in BOND_TYPES),
    "bond_is_aromatic",
    "bond_is_conjugated",
    "bond_is_in_ring",
)


def _float_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def get_edge_feature_names(config: GNNConfig) -> list[str]:
    """Return the feature names in the same order as edge_attr columns."""
    graph = config.graph
    geometry = config.atomic_features

    names = ["is_bonded"]

    if graph.include_bond_attributes:
        names.extend(BOND_FEATURE_NAMES)

    if graph.include_distances:
        names.append("distance_angstrom")

    names.extend(
        f"gaussian_rbf_{i:02d}"
        for i in range(graph.rbf_num_centers)
    )
    names.extend(
        f"{geometry.kernel}_order{_float_label(order)}_tau{_float_label(tau)}"
        for order, tau in geometry.kernel_configs
    )

    return names


def _bond_features(bond: Chem.Bond | None) -> list[float]:
    """Bond descriptor, or zeros for spatially connected unbonded atoms."""
    if bond is None:
        return [0.0] * len(BOND_FEATURE_NAMES)

    kind = bond.GetBondType()
    bond_type = {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
        Chem.BondType.AROMATIC: "aromatic",
    }.get(kind, "other")

    return [
        *(float(bond_type == name) for name in BOND_TYPES),
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def _get_edge_pairs(
    mol: Chem.Mol,
    distance_matrix: np.ndarray,
    config: GNNConfig,
) -> set[tuple[int, int]]:
    """Return undirected pairs, keeping all bonds in hybrid mode."""
    graph = config.graph
    pairs = set()

    if graph.edge_mode in ("covalent", "hybrid"):
        pairs.update(
            (min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
             max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for bond in mol.GetBonds()
        )

    if graph.edge_mode in ("radius", "hybrid"):
        cutoff = graph.radius_cutoff_angstrom
        n_atoms = mol.GetNumAtoms()

        if graph.max_neighbors is None:
            i, j = np.triu_indices(n_atoms, k=1)
            selected = distance_matrix[i, j] < cutoff
            pairs.update(zip(i[selected].tolist(), j[selected].tolist()))
        else:
            for i in range(n_atoms):
                neighbors = np.flatnonzero(distance_matrix[i] < cutoff)
                neighbors = neighbors[neighbors != i]
                neighbors = sorted(
                    neighbors,
                    key=lambda j: (distance_matrix[i, j], int(j)),
                )[:graph.max_neighbors]
                pairs.update(
                    (min(i, int(j)), max(i, int(j)))
                    for j in neighbors
                )

    return pairs


def _radial_features(
    distances: np.ndarray,
    radius_sums: np.ndarray,
    config: GNNConfig,
) -> np.ndarray:
    """Gaussian RBFs and exponential/Lorentz kernel values per edge."""
    graph = config.graph
    geometry = config.atomic_features

    centers = np.linspace(
        0.0,
        graph.radius_cutoff_angstrom,
        graph.rbf_num_centers,
    )
    width = centers[1] - centers[0]

    rbf = np.exp(
        -0.5 * ((distances[:, None] - centers[None, :]) / width) ** 2
    )

    kernels = []
    for order, tau in geometry.kernel_configs:
        scaled = distances / (tau * radius_sums)
        if geometry.kernel == "exponential":
            kernel_values = np.exp(-(scaled ** order))
        else:
            kernel_values = 1.0 / (1.0 + scaled ** order)
        kernels.append(kernel_values)

    return np.column_stack((rbf, *kernels))


def build_edge_features(
    mol: Chem.Mol,
    conf_id: int,
    config: GNNConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge_index [2, E] and edge_attr [E, F] as NumPy arrays.

    Edges are bidirectional, sorted, and have no duplicates or self-loops.
    Every covalent bond remains in hybrid mode, regardless of the cutoff.
    """
    graph = config.graph
    coords = np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)

    # Numerical check: malformed conformer coordinates must not enter a GNN.
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite conformer coordinates")

    distances = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :],
        axis=2,
    )
    np.fill_diagonal(distances, np.inf)

    pairs = _get_edge_pairs(mol, distances, config)
    directed = sorted(
        (i, j)
        for a, b in pairs
        for i, j in ((a, b), (b, a))
    )

    n_features = len(get_edge_feature_names(config))
    if not directed:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, n_features), dtype=np.float32),
        )

    edge_index = np.asarray(directed, dtype=np.int64).T
    i, j = edge_index
    edge_distances = distances[i, j]

    bond_features = []
    for source, target in directed:
        bond = mol.GetBondBetweenAtoms(source, target)
        feature = [float(bond is not None)]
        if graph.include_bond_attributes:
            feature.extend(_bond_features(bond))
        bond_features.append(feature)

    attributes = [np.asarray(bond_features, dtype=float)]

    if graph.include_distances:
        attributes.append(edge_distances[:, None])

    periodic_table = Chem.GetPeriodicTable()
    radii = np.asarray(
        [periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    )
    attributes.append(_radial_features(
        distances=edge_distances,
        radius_sums=radii[i] + radii[j],
        config=config,
    ))

    edge_attr = np.concatenate(attributes, axis=1).astype(np.float32)
    return edge_index, edge_attr
