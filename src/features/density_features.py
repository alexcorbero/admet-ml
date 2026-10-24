import numpy as np

from rdkit import Chem
from itertools import product

from .descriptors_3d import get_coordinates
from .radial_features import (
    map_element_symbol,
    build_element_vocabulary,
)


DENSITY_STATISTICS = (
    "sum",
    "mean",
    "std",
    "min",
    "max",
)


def density_kernel(
    distances: np.ndarray,
    scale: float,
    power: float = 2.0,
) -> np.ndarray:
    """
    Evaluate a generalized exponential density kernel.

    phi(d; eta) = exp(-(d / eta)**power)

    Parameters
    ----------
    distances : np.ndarray
        Distances in Angstroms.

    scale : float
        Spatial scale eta in Angstroms.

    power : float, default=2.0
        Kernel exponent.
        power=2 corresponds to a Gaussian kernel.

    Returns
    -------
    np.ndarray
        Kernel values with the same shape as `distances`.
    """

    if scale <= 0:
        raise ValueError("scale must be greater than 0.")

    if power <= 0:
        raise ValueError("power must be greater than 0.")

    distances = np.asarray(
        distances,
        dtype=float,
    )

    if np.any(distances < 0):
        raise ValueError(
            "distances must be non-negative."
        )

    return np.exp(
        -((distances / scale) ** power)
    )


def get_mapped_atom_symbols(
    mol: Chem.Mol,
    elements: list[str],
    other_label: str = "Other",
) -> np.ndarray:
    """
    Return G4 element labels for every atom.

    Elements present in `elements` retain their original symbol.
    Every other element is mapped to `other_label`.
    """

    symbols = [
        map_element_symbol(
            atom.GetSymbol(),
            elements=elements,
            other_label=other_label,
        )
        for atom in mol.GetAtoms()
    ]

    return np.asarray(
        symbols,
        dtype=str,
    )


def element_density_at_atoms(
    coords: np.ndarray,
    symbols: np.ndarray,
    target_element: str,
    source_element: str,
    scale: float,
    power: float = 2.0,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Evaluate source-element density at all target-element atoms.

    For every target atom i:

        rho_i =
            sum_j phi(||r_i - r_j||; scale)

    where j runs over atoms belonging to `source_element`.

    Returns one density value for every target atom.
    """

    coords = np.asarray(
        coords,
        dtype=float,
    )

    symbols = np.asarray(symbols)

    if (
        coords.ndim != 2
        or coords.shape[1] != 3
    ):
        raise ValueError(
            "coords must have shape (n_atoms, 3)."
        )

    if len(symbols) != len(coords):
        raise ValueError(
            "symbols and coords must contain "
            "the same number of atoms."
        )

    target_indices = np.flatnonzero(
        symbols == target_element
    )

    source_indices = np.flatnonzero(
        symbols == source_element
    )

    # No target atoms -> no density values to aggregate.
    if target_indices.size == 0:
        return np.array(
            [],
            dtype=float,
        )

    # Target atoms exist but there are no source atoms.
    # Each target therefore feels zero source density.
    if source_indices.size == 0:
        return np.zeros(
            target_indices.size,
            dtype=float,
        )

    target_coords = coords[
        target_indices
    ]

    source_coords = coords[
        source_indices
    ]

    # Shape:
    # (n_target, 1, 3)
    # -
    # (1, n_source, 3)
    #
    # ->
    # (n_target, n_source, 3)
    differences = (
        target_coords[:, None, :]
        - source_coords[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    # Shape:
    # (n_target, n_source)
    kernel_values = density_kernel(
        distances=distances,
        scale=scale,
        power=power,
    )

    if exclude_self:
        self_mask = (
            target_indices[:, None]
            == source_indices[None, :]
        )

        kernel_values[
            self_mask
        ] = 0.0

    # Sum source contributions for each target.
    densities = np.sum(
        kernel_values,
        axis=1,
    )

    return densities


def aggregate_density_values(
    density_values: np.ndarray,
) -> dict[str, float]:
    """
    Convert a variable number of atom-level density values
    into fixed-size summary statistics.
    """

    density_values = np.asarray(
        density_values,
        dtype=float,
    ).ravel()

    if density_values.size == 0:
        return {
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "sum": float(
            np.sum(density_values)
        ),
        "mean": float(
            np.mean(density_values)
        ),
        "std": float(
            np.std(density_values)
        ),
        "min": float(
            np.min(density_values)
        ),
        "max": float(
            np.max(density_values)
        ),
    }


def generate_element_channels(
    vocabulary: list[str],
) -> list[tuple[str, str]]:
    """
    Generate all directed target-source element channels.

    For vocabulary = ["C", "O"]:

        C <- C
        C <- O
        O <- C
        O <- O
    """

    return list(
        product(
            vocabulary,
            repeat=2,
        )
    )


def density_feature_names(
    channels: list[tuple[str, str]],
    scales: np.ndarray,
    statistics: tuple[str, ...],
) -> list[str]:
    """
    Generate deterministic names for all G4 density features.
    """

    valid_statistics = set(
        DENSITY_STATISTICS
    )

    unknown_statistics = (
        set(statistics)
        - valid_statistics
    )

    if unknown_statistics:
        raise ValueError(
            "Unknown density statistics: "
            f"{unknown_statistics}"
        )

    names = []

    for target_element, source_element in channels:

        for scale in scales:

            scale_label = (
                f"{float(scale):g}"
                .replace(".", "p")
            )

            for statistic in statistics:

                name = (
                    f"density_"
                    f"{target_element}_"
                    f"from_{source_element}_"
                    f"eta{scale_label}_"
                    f"{statistic}"
                )

                names.append(name)

    return names


def element_channel_density_features(
    coords: np.ndarray,
    symbols: np.ndarray,
    channels: list[tuple[str, str]],
    scales: np.ndarray,
    power: float = 2.0,
) -> dict[str, float]:
    """
    Calculate multiscale density features for every
    directed element channel.
    """

    scales = np.asarray(
        scales,
        dtype=float,
    )

    features = {}

    for target_element, source_element in channels:

        for scale in scales:

            density_values = (
                element_density_at_atoms(
                    coords=coords,
                    symbols=symbols,
                    target_element=target_element,
                    source_element=source_element,
                    scale=float(scale),
                    power=power,
                    exclude_self=True,
                )
            )

            statistics = (
                aggregate_density_values(
                    density_values
                )
            )

            scale_label = (
                f"{float(scale):g}"
                .replace(".", "p")
            )

            for statistic, value in statistics.items():

                feature_name = (
                    f"density_"
                    f"{target_element}_"
                    f"from_{source_element}_"
                    f"eta{scale_label}_"
                    f"{statistic}"
                )

                features[
                    feature_name
                ] = value

    return features


def calculate_density_features(
    mol: Chem.Mol,
    conf_id: int,
    elements: list[str],
    scales: list[float],
    power: float = 2.0,
    other_label: str = "Other",
) -> dict[str, float]:
    """
    Calculate the complete G4 multiscale density representation
    for one molecular conformer.
    """

    coords = get_coordinates(
        mol=mol,
        conf_id=conf_id,
    )

    symbols = get_mapped_atom_symbols(
        mol=mol,
        elements=elements,
        other_label=other_label,
    )

    vocabulary = build_element_vocabulary(
        elements=elements,
        other_label=other_label,
    )

    channels = generate_element_channels(
        vocabulary
    )

    features = (
        element_channel_density_features(
            coords=coords,
            symbols=symbols,
            channels=channels,
            scales=np.asarray(
                scales,
                dtype=float,
            ),
            power=power,
        )
    )

    return features