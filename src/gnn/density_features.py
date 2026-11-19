"""Atom-centred G6 density and curvature features for the G7 GNN.

Each atom has one channel per *source* element (C, N, O, ..., Other),
for each density type and kernel configuration. The target element is
already represented by the atom's atomic number in atom_features.py.

The calculations reuse the G6 field derivatives and curvature formulas;
there is no aggregation into molecular statistics.
"""

import numpy as np
from rdkit import Chem

from .config import GNNConfig
from ..features.dg_gl_features import (
    density_field_derivatives,
    get_vdw_radii,
    get_gasteiger_charges,
    compute_principal_curvatures,
)
from ..features.curvature_features import mean_curvature, gaussian_curvature


CURVATURE_NAMES = ("H", "K", "kappa_min", "kappa_max")


def _float_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _source_elements(config: GNNConfig) -> tuple[str, ...]:
    geometry = config.atomic_features
    return (*geometry.elements, geometry.other_label)


def _channel_feature_names(config: GNNConfig) -> tuple[str, ...]:
    geometry = config.atomic_features
    names = []

    if geometry.include_density:
        names.append("density")

    if geometry.include_curvature:
        names.extend(CURVATURE_NAMES)
        if geometry.include_validity_masks:
            names.append("valid")

    return tuple(names)


def get_density_feature_names(config: GNNConfig) -> list[str]:
    """Return the ordered names of G6-derived atomic features."""
    geometry = config.atomic_features
    properties = _channel_feature_names(config)
    names = []

    for density_type in geometry.density_types:
        for order, tau in geometry.kernel_configs:
            for source in _source_elements(config):
                prefix = (
                    f"g6_{density_type}_from_{source}_"
                    f"{geometry.kernel}_order{_float_label(order)}_"
                    f"tau{_float_label(tau)}"
                )
                names.extend(f"{prefix}_{name}" for name in properties)

    return names


def _transform(values: np.ndarray, config: GNNConfig) -> np.ndarray:
    """Compress extreme geometric values without changing their sign."""
    if config.atomic_features.geometry_transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    return values


def build_density_features(
    mol: Chem.Mol,
    conf_id: int,
    config: GNNConfig,
) -> np.ndarray:
    """Calculate G6-derived geometry for every atom of a 3D conformer.

    Returns
    -------
    x_dg : np.ndarray, shape (n_atoms, n_dg_features), dtype float32
        Per-atom density, mean/Gaussian/principal curvatures and validity
        flags, ordered as in get_density_feature_names(config).

    Notes
    -----
    - G6 exclusion: d_ij > r_vdw_i + r_vdw_j + sigma.
    - Pair length: eta_ij = tau * (r_vdw_i + r_vdw_j).
    - Curvature is invalid when |grad(rho)| is below the configured
      absolute/relative threshold; invalid curvatures become zero and
      their validity flag is zero.
    - No normalization is fitted here; do that only on training data.
    """
    geometry = config.atomic_features
    n_atoms = mol.GetNumAtoms()
    property_names = _channel_feature_names(config)

    if not property_names:
        return np.empty((n_atoms, 0), dtype=np.float32)

    coords = np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite conformer coordinates")

    radii = get_vdw_radii(mol)
    symbols = np.asarray([atom.GetSymbol() for atom in mol.GetAtoms()])
    source_elements = _source_elements(config)
    mapped_symbols = np.where(
        np.isin(symbols, geometry.elements),
        symbols,
        geometry.other_label,
    )

    weights_by_type = {"number": np.ones(n_atoms, dtype=float)}
    if "charge" in geometry.density_types:
        weights_by_type["charge"] = get_gasteiger_charges(mol)

    feature_columns = []

    for density_type in geometry.density_types:
        weights = weights_by_type[density_type]

        for order, tau in geometry.kernel_configs:
            for source in source_elements:
                source_idx = np.flatnonzero(mapped_symbols == source)

                if source_idx.size == 0:
                    feature_columns.append(
                        np.zeros((n_atoms, len(property_names)), dtype=float)
                    )
                    continue

                differences = (
                    coords[:, None, :] - coords[source_idx][None, :, :]
                )
                distances = np.linalg.norm(differences, axis=2)
                radius_sums = radii[:, None] + radii[source_idx][None, :]
                allowed = distances > (radius_sums + geometry.sigma_angstrom)
                pair_scales = tau * radius_sums

                density, gradient, hessian = density_field_derivatives(
                    evaluation_coords=coords,
                    source_coords=coords[source_idx],
                    source_weights=weights[source_idx],
                    pair_scales=pair_scales,
                    allowed_mask=allowed,
                    kernel=geometry.kernel,
                    order=order,
                )

                features = {}
                if geometry.include_density:
                    features["density"] = _transform(density, config)

                if geometry.include_curvature:
                    n_eligible = allowed.sum(axis=1)
                    effective_scale = np.divide(
                        (pair_scales * allowed).sum(axis=1),
                        n_eligible,
                        out=np.ones(n_atoms, dtype=float),
                        where=n_eligible > 0,
                    )
                    gradient_norm = np.linalg.norm(gradient, axis=1)
                    threshold = np.maximum(
                        geometry.gradient_epsilon,
                        geometry.relative_gradient_epsilon
                        * np.abs(density) / effective_scale,
                    )

                    H, valid_H = mean_curvature(
                        gradient, hessian,
                        gradient_epsilon=geometry.gradient_epsilon,
                    )
                    K, valid_K = gaussian_curvature(
                        gradient, hessian,
                        gradient_epsilon=geometry.gradient_epsilon,
                    )

                    valid = (
                        (gradient_norm > threshold)
                        & valid_H
                        & valid_K
                        & np.isfinite(H)
                        & np.isfinite(K)
                    )
                    kappa_min, kappa_max, valid = compute_principal_curvatures(
                        H, K, valid,
                    )

                    for name, values in zip(
                        CURVATURE_NAMES, (H, K, kappa_min, kappa_max)
                    ):
                        # Undefined curvature is represented by value=0,
                        # with its corresponding validity flag kept separate.
                        features[name] = _transform(
                            np.where(valid, values, 0.0), config
                        )

                    if geometry.include_validity_masks:
                        features["valid"] = valid.astype(float)

                feature_columns.append(
                    np.column_stack([features[name] for name in property_names])
                )

    return np.concatenate(feature_columns, axis=1).astype(np.float32)
