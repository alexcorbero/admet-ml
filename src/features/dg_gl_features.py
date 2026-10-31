"""
G6: Element-resolved multiscale differential-geometry features.

This module implements a DG-GL-like representation using:
- directed target/source element channels;
- generalized exponential or generalized Lorentz kernels;
- atom-pair van der Waals length scales;
- geometric exclusion of covalent-range interactions;
- mean, Gaussian and principal curvatures;
- robust molecular aggregation statistics.

Curvatures are evaluated at target atom centres. This is an atom-centred
approximation compatible with the existing G5 feature pipeline; it does not
construct or sample a full 3D isosurface/grid over each van der Waals domain.
"""


import numpy as np

from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from .descriptors_3d import get_coordinates
from .density_features import (
    get_mapped_atom_symbols,
    generate_element_channels,
)
from .radial_features import build_element_vocabulary
from .curvature_features import (
    G5_GRADIENT_EPSILON,
    G5_RELATIVE_GRADIENT_EPSILON,
    mean_curvature,
    gaussian_curvature,
)


# ---------------------------------------------------------------------
# Default chemical vocabulary and kernel configurations
# ---------------------------------------------------------------------

DEFAULT_ELEMENTS = (
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
)

DEFAULT_OTHER_LABEL = "Other"

# Initial two-scale configurations reported for the solvation task in
# Nguyen & Wei (2019). They are starting points, not AqSolDB-optimized
# hyperparameters.
DEFAULT_KERNEL_CONFIGS = {
    "exponential": (
        (3.5, 0.3),  # (order, tau)
        (2.5, 1.3),
    ),
    "lorentz": (
        (3.0, 1.3),
        (6.5, 0.3),
    ),
}

G6_CURVATURE_STATISTICS = (
    "mean",
    "std",
    "median",
    "q10",
    "q90",
    "iqr",
    "mean_abs",
    "median_abs",
)

G6_CURVATURE_NAMES = (
    "H",
    "K",
    "kappa_min",
    "kappa_max",
)


# ---------------------------------------------------------------------
# Naming and atomic-property helpers
# ---------------------------------------------------------------------

def _float_label(value: float) -> str:
    """Return a stable feature-name label for a floating-point value."""
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def get_vdw_radii(
    mol: Chem.Mol,
    vdw_radii: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Obtain one van der Waals radius per atom.

    RDKit's periodic table is used by default. `vdw_radii` can override
    individual elements, for example {"Zn": 1.39}.

    The original element symbol is used here, not the mapped channel label
    (e.g. "Other"), so rare elements retain their own atomic radius.
    """
    periodic_table = Chem.GetPeriodicTable()
    overrides = vdw_radii or {}

    radii = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()

        if symbol in overrides:
            radius = float(overrides[symbol])
        else:
            radius = float(
                periodic_table.GetRvdw(atom.GetAtomicNum())
            )

        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"No valid van der Waals radius is available for "
                f"{symbol}. Supply it through vdw_radii."
            )

        radii.append(radius)

    return np.asarray(radii, dtype=float)


def get_gasteiger_charges(
    mol: Chem.Mol,
    n_iter: int = 12,
) -> np.ndarray:
    """
    Compute Gasteiger partial charges without modifying the input molecule.

    This is an optional approximation for the charge-density channels.
    It is not a quantum-mechanical charge calculation.
    """
    charged_mol = Chem.Mol(mol)

    rdPartialCharges.ComputeGasteigerCharges(
        charged_mol,
        nIter=int(n_iter),
        throwOnParamFailure=True,
    )

    charges = []

    for atom in charged_mol.GetAtoms():
        if not atom.HasProp("_GasteigerCharge"):
            raise ValueError(
                "RDKit did not assign a Gasteiger charge to every atom."
            )

        charge = float(atom.GetProp("_GasteigerCharge"))

        if not np.isfinite(charge):
            raise ValueError(
                "A non-finite Gasteiger charge was obtained for "
                f"atom {atom.GetIdx()} ({atom.GetSymbol()})."
            )

        charges.append(charge)

    return np.asarray(charges, dtype=float)


# ---------------------------------------------------------------------
# Generalized radial kernels and analytical derivatives
# ---------------------------------------------------------------------

def radial_kernel_derivatives(
    distances: np.ndarray,
    scales: np.ndarray,
    kernel: str,
    order: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a generalized radial kernel and its first two distance
    derivatives.

    Parameters
    ----------
    distances
        Pairwise distances, in Angstroms.
    scales
        Pairwise characteristic lengths eta, in Angstroms. Must be
        broadcast-compatible with `distances`.
    kernel
        "exponential" or "lorentz".
    order
        Exponential order kappa or Lorentz order nu.

    Returns
    -------
    phi, dphi_dd, d2phi_dd2
        Kernel and first/second derivatives with respect to distance.

    Notes
    -----
    The order is restricted to >= 2 so that the radial kernel is C2 at
    the origin and can be used consistently for Hessian-based curvature.
    """
    distances = np.asarray(distances, dtype=float)
    scales = np.asarray(scales, dtype=float)

    kernel = str(kernel).lower()
    order = float(order)

    if kernel not in {"exponential", "lorentz"}:
        raise ValueError(
            "kernel must be either 'exponential' or 'lorentz'."
        )

    if order < 2.0:
        raise ValueError(
            "order must be >= 2 for a C2 radial kernel."
        )

    if np.any(distances < 0):
        raise ValueError("distances must be non-negative.")

    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("all scales must be finite and positive.")

    x = distances / scales

    with np.errstate(
        over="ignore",
        under="ignore",
        divide="ignore",
        invalid="ignore",
    ):
        if kernel == "exponential":
            # phi = exp(-(d / eta)**order)
            phi = np.exp(-(x ** order))

            dphi = (
                -(order / scales)
                * (x ** (order - 1.0))
                * phi
            )

            d2phi = (
                (
                    order**2 * x ** (2.0 * order - 2.0)
                    - order * (order - 1.0) * x ** (order - 2.0)
                )
                * phi
                / scales**2
            )

        else:
            # phi = 1 / (1 + (d / eta)**order)
            denominator = 1.0 + x**order
            phi = 1.0 / denominator

            dphi = (
                -(order / scales)
                * x ** (order - 1.0)
                / denominator**2
            )

            d2phi = (
                (
                    -order * (order - 1.0)
                    * x ** (order - 2.0)
                    / denominator**2
                    +
                    2.0 * order**2
                    * x ** (2.0 * order - 2.0)
                    / denominator**3
                )
                / scales**2
            )

    return phi, dphi, d2phi


def density_field_derivatives(
    evaluation_coords: np.ndarray,
    source_coords: np.ndarray,
    source_weights: np.ndarray,
    pair_scales: np.ndarray,
    allowed_mask: np.ndarray,
    kernel: str,
    order: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a weighted element-interactive density, gradient and Hessian.

    The density is

        rho(r_i) = sum_j w_j * Phi(||r_i-r_j||; eta_ij)

    where `allowed_mask` excludes self/covalent-range interactions.
    Pair-specific eta values are supported, which is important for the
    "Other" channel because its atoms can have different actual elements
    and therefore different van der Waals radii.
    """
    evaluation_coords = np.asarray(evaluation_coords, dtype=float)
    source_coords = np.asarray(source_coords, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    pair_scales = np.asarray(pair_scales, dtype=float)
    allowed_mask = np.asarray(allowed_mask, dtype=bool)

    n_target = len(evaluation_coords)
    n_source = len(source_coords)

    if n_source == 0:
        return (
            np.zeros(n_target, dtype=float),
            np.zeros((n_target, 3), dtype=float),
            np.zeros((n_target, 3, 3), dtype=float),
        )

    expected_pair_shape = (n_target, n_source)

    if allowed_mask.shape != expected_pair_shape:
        raise ValueError(
            f"allowed_mask must have shape {expected_pair_shape}."
        )

    if pair_scales.shape != expected_pair_shape:
        raise ValueError(
            f"pair_scales must have shape {expected_pair_shape}."
        )

    if source_weights.shape != (n_source,):
        raise ValueError(
            "source_weights must contain one value per source atom."
        )

    differences = (
        evaluation_coords[:, None, :]
        - source_coords[None, :, :]
    )

    distances = np.linalg.norm(differences, axis=2)

    phi, dphi, d2phi = radial_kernel_derivatives(
        distances=distances,
        scales=pair_scales,
        kernel=kernel,
        order=order,
    )

    # For a radial function f(d):
    #
    # grad f = (f'(d)/d) * q
    #
    # Hess f = a I + b q q^T
    #
    # a = f'(d)/d
    # b = (f''(d) - f'(d)/d) / d^2
    #
    # At d=0, a is defined by the radial limit f''(0), while q q^T = 0.
    near_zero = distances <= 1e-10

    a = np.empty_like(distances)
    nonzero = ~near_zero

    a[nonzero] = dphi[nonzero] / distances[nonzero]
    a[near_zero] = d2phi[near_zero]

    b = np.zeros_like(distances)
    b[nonzero] = (
        (d2phi[nonzero] - a[nonzero])
        / distances[nonzero]**2
    )

    weighted_mask = (
        allowed_mask.astype(float)
        * source_weights[None, :]
    )

    weighted_phi = phi * weighted_mask
    weighted_a = a * weighted_mask
    weighted_b = b * weighted_mask

    density = np.sum(weighted_phi, axis=1)

    gradient = np.sum(
        weighted_a[:, :, None] * differences,
        axis=1,
    )

    outer_products = (
        differences[:, :, :, None]
        * differences[:, :, None, :]
    )

    identity = np.eye(3, dtype=float)

    hessian_contributions = (
        weighted_a[:, :, None, None]
        * identity[None, None, :, :]
        +
        weighted_b[:, :, None, None]
        * outer_products
    )

    hessian = np.sum(hessian_contributions, axis=1)

    return density, gradient, hessian


# ---------------------------------------------------------------------
# Curvature and robust aggregation
# ---------------------------------------------------------------------

def _regular_density_mask(
    density: np.ndarray,
    gradient: np.ndarray,
    effective_scales: np.ndarray,
    absolute_epsilon: float,
    relative_epsilon: float,
) -> np.ndarray:
    """Scale-aware regularity criterion for atom-specific pair scales."""
    gradient_norm = np.linalg.norm(gradient, axis=1)

    relative_threshold = (
        relative_epsilon
        * np.abs(density)
        / np.maximum(effective_scales, 1e-12)
    )

    threshold = np.maximum(
        absolute_epsilon,
        relative_threshold,
    )

    return (
        np.isfinite(density)
        & np.isfinite(gradient_norm)
        & np.isfinite(effective_scales)
        & (gradient_norm > threshold)
    )


def compute_principal_curvatures(
    mean_curvatures: np.ndarray,
    gaussian_curvatures: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the two principal curvatures:

        kappa_min = H - sqrt(H**2 - K)
        kappa_max = H + sqrt(H**2 - K)

    Small negative radicands caused by floating-point round-off are clipped
    to zero. Materially negative radicands are marked invalid.
    """
    H = np.asarray(mean_curvatures, dtype=float)
    K = np.asarray(gaussian_curvatures, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool).copy()

    radicand = H**2 - K

    tolerance = (
        1e-10
        * np.maximum(
            1.0,
            np.maximum(H**2, np.abs(K)),
        )
    )

    valid &= np.isfinite(radicand)
    valid &= radicand >= -tolerance

    root = np.full(H.shape, np.nan, dtype=float)
    root[valid] = np.sqrt(
        np.maximum(radicand[valid], 0.0)
    )

    kappa_min = np.full(H.shape, np.nan, dtype=float)
    kappa_max = np.full(H.shape, np.nan, dtype=float)

    kappa_min[valid] = H[valid] - root[valid]
    kappa_max[valid] = H[valid] + root[valid]

    return kappa_min, kappa_max, valid


def aggregate_curvature_statistics(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """
    Aggregate valid atom-level curvature values into robust molecular
    statistics.

    Empty/invalid channels return zeros, while their valid_fraction is
    reported separately by the calling function.
    """
    values = np.asarray(values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    selected = values[
        valid_mask & np.isfinite(values)
    ]

    if selected.size == 0:
        return {
            name: 0.0
            for name in G6_CURVATURE_STATISTICS
        }

    absolute = np.abs(selected)
    q25, q75 = np.percentile(selected, [25, 75])

    return {
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "median": float(np.median(selected)),
        "q10": float(np.percentile(selected, 10)),
        "q90": float(np.percentile(selected, 90)),
        "iqr": float(q75 - q25),
        "mean_abs": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
    }


# ---------------------------------------------------------------------
# Element-interactive curvature for one directed channel and one scale
# ---------------------------------------------------------------------

def element_interactive_curvature_at_atoms(
    coords: np.ndarray,
    mapped_symbols: np.ndarray,
    atomic_radii: np.ndarray,
    atom_weights: np.ndarray,
    target_element: str,
    source_element: str,
    kernel: str,
    order: float,
    tau: float,
    sigma: float = 0.0,
    gradient_epsilon: float = G5_GRADIENT_EPSILON,
    relative_gradient_epsilon: float = G5_RELATIVE_GRADIENT_EPSILON,
) -> dict[str, np.ndarray | float]:
    """
    Compute G6 curvature values for one directed element channel.

    Target atoms define the evaluation points. Source atoms generate the
    density field.

    Pair eligibility follows the DG-GL geometric exclusion:

        d_ij > r_i(vdW) + r_j(vdW) + sigma

    The characteristic length is:

        eta_ij = tau * (r_i(vdW) + r_j(vdW))

    For ordinary element channels, the radii are constant within each
    element and this reduces to the usual element-pair scale. For "Other",
    actual atom-specific radii are retained.
    """
    coords = np.asarray(coords, dtype=float)
    mapped_symbols = np.asarray(mapped_symbols)
    atomic_radii = np.asarray(atomic_radii, dtype=float)
    atom_weights = np.asarray(atom_weights, dtype=float)

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if tau <= 0:
        raise ValueError("tau must be greater than zero.")

    target_indices = np.flatnonzero(
        mapped_symbols == target_element
    )
    source_indices = np.flatnonzero(
        mapped_symbols == source_element
    )

    n_target = target_indices.size

    empty = np.full(n_target, np.nan, dtype=float)
    no_valid = np.zeros(n_target, dtype=bool)

    if n_target == 0:
        return {
            "H": empty.copy(),
            "K": empty.copy(),
            "kappa_min": empty.copy(),
            "kappa_max": empty.copy(),
            "valid": no_valid,
            "interaction_fraction": 0.0,
        }

    target_coords = coords[target_indices]
    source_coords = coords[source_indices]

    target_radii = atomic_radii[target_indices]
    source_radii = atomic_radii[source_indices]
    source_weights = atom_weights[source_indices]

    if source_indices.size == 0:
        return {
            "H": empty.copy(),
            "K": empty.copy(),
            "kappa_min": empty.copy(),
            "kappa_max": empty.copy(),
            "valid": no_valid,
            "interaction_fraction": 0.0,
        }

    differences = (
        target_coords[:, None, :]
        - source_coords[None, :, :]
    )
    distances = np.linalg.norm(differences, axis=2)

    pair_radius_sums = (
        target_radii[:, None]
        + source_radii[None, :]
    )

    allowed_mask = (
        distances > pair_radius_sums + sigma
    )

    pair_scales = tau * pair_radius_sums

    density, gradient, hessian = density_field_derivatives(
        evaluation_coords=target_coords,
        source_coords=source_coords,
        source_weights=source_weights,
        pair_scales=pair_scales,
        allowed_mask=allowed_mask,
        kernel=kernel,
        order=order,
    )

    n_eligible = np.sum(allowed_mask, axis=1)

    effective_scales = np.divide(
        np.sum(pair_scales * allowed_mask, axis=1),
        n_eligible,
        out=np.ones(n_target, dtype=float),
        where=n_eligible > 0,
    )

    regular_mask = _regular_density_mask(
        density=density,
        gradient=gradient,
        effective_scales=effective_scales,
        absolute_epsilon=gradient_epsilon,
        relative_epsilon=relative_gradient_epsilon,
    )

    H, valid_H = mean_curvature(
        gradient=gradient,
        hessian=hessian,
        gradient_epsilon=gradient_epsilon,
    )

    K, valid_K = gaussian_curvature(
        gradient=gradient,
        hessian=hessian,
        gradient_epsilon=gradient_epsilon,
    )

    valid = (
        regular_mask
        & valid_H
        & valid_K
        & np.isfinite(H)
        & np.isfinite(K)
    )

    kappa_min, kappa_max, valid = compute_principal_curvatures(
        mean_curvatures=H,
        gaussian_curvatures=K,
        valid_mask=valid,
    )

    for values in (H, K):
        values[~valid] = np.nan

    interaction_fraction = float(
        np.mean(n_eligible > 0)
    )

    return {
        "H": H,
        "K": K,
        "kappa_min": kappa_min,
        "kappa_max": kappa_max,
        "valid": valid,
        "interaction_fraction": interaction_fraction,
    }


# ---------------------------------------------------------------------
# Complete G6 representation
# ---------------------------------------------------------------------

def calculate_dg_gl_features(
    mol: Chem.Mol,
    conf_id: int,
    elements: list[str] | tuple[str, ...] = DEFAULT_ELEMENTS,
    kernel: str = "exponential",
    kernel_configs: list[tuple[float, float]]
    | tuple[tuple[float, float], ...]
    | None = None,
    density_types: tuple[str, ...] | list[str] = ("number",),
    other_label: str = DEFAULT_OTHER_LABEL,
    sigma: float = 0.0,
    vdw_radii: dict[str, float] | None = None,
    gradient_epsilon: float = G5_GRADIENT_EPSILON,
    relative_gradient_epsilon: float = G5_RELATIVE_GRADIENT_EPSILON,
    gasteiger_iterations: int = 12,
) -> dict[str, float]:
    """
    Calculate the complete G6 DG-GL-like representation for one conformer.

    Parameters
    ----------
    mol
        RDKit molecule containing the requested 3D conformer.
    conf_id
        RDKit conformer identifier.
    elements
        Explicit element vocabulary. All other elements are mapped to
        `other_label` for feature-channel construction.
    kernel
        "exponential" or "lorentz".
    kernel_configs
        Multiscale configurations as (order, tau) pairs.
        `order` is kappa for exponential and nu for Lorentz.
        `tau` controls eta_ij = tau * (r_i + r_j).
        If omitted, initial two-scale configurations from the DG-GL
        solvation experiments are used.
    density_types
        Any combination of "number" and "charge".
        "number" uses source weights w_j = 1.
        "charge" uses RDKit Gasteiger partial charges w_j = q_j.
    other_label
        Label used for atoms outside `elements`.
    sigma
        Dataset-level correction in Angstroms for the geometric exclusion
        d_ij > r_i + r_j + sigma. Defaults to 0.0; tune/estimate it using
        training data in a later experimental stage.
    vdw_radii
        Optional per-element overrides for RDKit van der Waals radii.
    gradient_epsilon
        Absolute regularity threshold for density gradients.
    relative_gradient_epsilon
        Scale-aware regularity threshold.
    gasteiger_iterations
        Number of Gasteiger charge iterations if charge density is enabled.

    Returns
    -------
    dict[str, float]
        Fixed-size molecular G6 feature dictionary.

    Notes
    -----
    This implementation evaluates EICs at target atom centres, following
    the atom-centred G5 pipeline. A full DG-GL surface/grid implementation
    would additionally sample the complete van der Waals domains.
    """
    kernel = str(kernel).lower()

    if kernel not in DEFAULT_KERNEL_CONFIGS:
        raise ValueError(
            "kernel must be either 'exponential' or 'lorentz'."
        )

    if kernel_configs is None:
        configs = DEFAULT_KERNEL_CONFIGS[kernel]
    else:
        configs = tuple(
            (float(order), float(tau))
            for order, tau in kernel_configs
        )

    if not configs:
        raise ValueError("kernel_configs must contain at least one scale.")

    for order, tau in configs:
        if order < 2.0:
            raise ValueError(
                "All kernel orders must be >= 2 for C2 curvature."
            )
        if tau <= 0:
            raise ValueError("All tau values must be positive.")

    density_types = tuple(
        str(value).lower()
        for value in density_types
    )

    if not density_types:
        raise ValueError("density_types cannot be empty.")

    unknown_density_types = (
        set(density_types) - {"number", "charge"}
    )
    if unknown_density_types:
        raise ValueError(
            f"Unknown density types: {unknown_density_types}."
        )

    if len(set(density_types)) != len(density_types):
        raise ValueError("density_types cannot contain duplicates.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    coords = get_coordinates(
        mol=mol,
        conf_id=conf_id,
    )

    if (
        coords.ndim != 2
        or coords.shape != (mol.GetNumAtoms(), 3)
        or not np.all(np.isfinite(coords))
    ):
        raise ValueError(
            "The conformer coordinates must be finite and have "
            "shape (n_atoms, 3)."
        )

    # Keep the original atom symbols and radii for physical calculations.
    atomic_radii = get_vdw_radii(
        mol=mol,
        vdw_radii=vdw_radii,
    )

    mapped_symbols = get_mapped_atom_symbols(
        mol=mol,
        elements=list(elements),
        other_label=other_label,
    )

    vocabulary = build_element_vocabulary(
        elements=list(elements),
        other_label=other_label,
    )

    channels = generate_element_channels(vocabulary)

    weights_by_type: dict[str, np.ndarray] = {
        "number": np.ones(mol.GetNumAtoms(), dtype=float),
    }

    if "charge" in density_types:
        weights_by_type["charge"] = get_gasteiger_charges(
            mol=mol,
            n_iter=gasteiger_iterations,
        )

    features: dict[str, float] = {}

    for density_type in density_types:
        atom_weights = weights_by_type[density_type]

        for target_element, source_element in channels:
            for order, tau in configs:
                curvature_data = element_interactive_curvature_at_atoms(
                    coords=coords,
                    mapped_symbols=mapped_symbols,
                    atomic_radii=atomic_radii,
                    atom_weights=atom_weights,
                    target_element=target_element,
                    source_element=source_element,
                    kernel=kernel,
                    order=order,
                    tau=tau,
                    sigma=sigma,
                    gradient_epsilon=gradient_epsilon,
                    relative_gradient_epsilon=relative_gradient_epsilon,
                )

                valid_mask = curvature_data["valid"]
                interaction_fraction = float(
                    curvature_data["interaction_fraction"]
                )

                target_count = int(
                    np.count_nonzero(mapped_symbols == target_element)
                )

                valid_fraction = (
                    float(np.count_nonzero(valid_mask) / target_count)
                    if target_count > 0
                    else 0.0
                )

                order_label = _float_label(order)
                tau_label = _float_label(tau)

                prefix = (
                    f"g6_{density_type}_"
                    f"{target_element}_from_{source_element}_"
                    f"{kernel}_order{order_label}_tau{tau_label}"
                )

                for curvature_name in G6_CURVATURE_NAMES:
                    statistics = aggregate_curvature_statistics(
                        values=curvature_data[curvature_name],
                        valid_mask=valid_mask,
                    )

                    for statistic_name, value in statistics.items():
                        feature_name = (
                            f"{prefix}_{curvature_name}_{statistic_name}"
                        )
                        features[feature_name] = float(value)

                features[f"{prefix}_valid_fraction"] = valid_fraction
                features[
                    f"{prefix}_interaction_fraction"
                ] = interaction_fraction

    return features


def build_g6_feature_names(
    elements=DEFAULT_ELEMENTS,
    kernel="exponential",
    kernel_configs=None,
    density_types=("number",),
    other_label=DEFAULT_OTHER_LABEL,
):
    """
    Generate the complete and ordered G6 feature schema.
    """

    kernel = kernel.lower()

    if kernel not in DEFAULT_KERNEL_CONFIGS:
        raise ValueError(
            "kernel must be 'exponential' or 'lorentz'."
        )

    if kernel_configs is None:
        kernel_configs = DEFAULT_KERNEL_CONFIGS[kernel]

    vocabulary = build_element_vocabulary(
        elements=list(elements),
        other_label=other_label,
    )

    channels = generate_element_channels(vocabulary)

    feature_names = []

    for density_type in density_types:

        for target_element, source_element in channels:

            for order, tau in kernel_configs:

                order_label = _float_label(order)
                tau_label = _float_label(tau)

                prefix = (
                    f"g6_{density_type}_"
                    f"{target_element}_from_{source_element}_"
                    f"{kernel}_order{order_label}_tau{tau_label}"
                )

                for curvature in G6_CURVATURE_NAMES:

                    for statistic in G6_CURVATURE_STATISTICS:

                        feature_names.append(
                            f"{prefix}_{curvature}_{statistic}"
                        )

                feature_names.extend([
                    f"{prefix}_valid_fraction",
                    f"{prefix}_interaction_fraction",
                ])

    return feature_names