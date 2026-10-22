import numpy as np
from .descriptors_3d import get_coordinates

from rdkit import Chem

from collections import defaultdict
from itertools import combinations_with_replacement

def get_atom_symbols(
        mol: Chem.Mol) -> np.ndarray:
    """
    Return the element symbol associated with each atom.

    Example
    -------
    ["C", "C", "O", "N"]
    """
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    return np.array(symbols)

def canonical_element_pair(
    element_a: str,
    element_b: str) -> tuple[str, str]:
    """
    Return an order-independent representation of an element pair.

    Examples
    --------
    ("O", "C") -> ("C", "O")
    ("N", "N") -> ("N", "N")

    Ensures that C-O and O-C correspond to the same feature channel.
    """
    return tuple(sorted((element_a, element_b)))

def get_element_pair_distances(
    mol: Chem.Mol,
    coords: np.ndarray,
    elements: list[str],
    other_label: str = "Other",
) -> dict[tuple[str, str], np.ndarray]:
    """
    Compute pairwise atomic distances grouped by mapped element pair.

    Elements not present in `elements` are mapped to `other_label`.

    Examples
    --------
    If elements = ["C", "N", "O"]:

        C-Zn  -> C-Other
        Fe-O  -> O-Other
        Zn-Fe -> Other-Other

    Returns
    -------
    dict[tuple[str, str], np.ndarray]
        Mapping from canonical element pair to all corresponding
        interatomic distances.
    """

    symbols = get_atom_symbols(mol)

    pair_distances = defaultdict(list)

    for i in range(mol.GetNumAtoms()):

        for j in range(i + 1, mol.GetNumAtoms()):

            diff = coords[i] - coords[j]

            distance = np.linalg.norm(diff)

            element_i = map_element_symbol(
                symbols[i],
                elements,
                other_label=other_label,
            )

            element_j = map_element_symbol(
                symbols[j],
                elements,
                other_label=other_label,
            )

            pair = canonical_element_pair(
                element_i,
                element_j,
            )

            pair_distances[pair].append(
                distance
            )

    return {
        pair: np.asarray(
            distances,
            dtype=float,
        )
        for pair, distances
        in pair_distances.items()
    }

def radial_centers(
    min_distance: float = 0.0,
    max_distance: float = 10.0,
    n_centers: int = 20) -> np.ndarray:
    """
    Generate evenly spaced centers for the radial basis functions.

    Returns
    -------
    Array of radial centers μ_k.
    """
    centers = np.linspace(min_distance, max_distance, n_centers)
    
    return centers

def gaussian_rbf(
    distances: np.ndarray,
    centers: np.ndarray,
    gamma: float) -> np.ndarray:
    """
    Expand distances using Gaussian radial basis functions.

    Computes

        exp(-gamma * (d - mu_k)**2)

    for every distance d and radial center mu_k.

    Returns
    -------
    shape = (n_distances, n_centers)
    """

    diff = distances[:, None] - centers[None, :]

    rbf = np.exp(-gamma * diff**2)

    return rbf

def aggregate_rbf(
    rbf_values: np.ndarray,
    normalize: bool = False) -> np.ndarray:

    if rbf_values.shape[0] == 0:
        return np.zeros(
            rbf_values.shape[1],
            dtype=float,
        )

    if normalize:
        return np.mean(
            rbf_values,
            axis=0,
        )

    return np.sum(
        rbf_values,
        axis=0,
    )

def generate_element_pairs(
    elements: list[str]) -> list[tuple[str, str]]:

    pairs = list(
        combinations_with_replacement(
            elements,
            2,
        )
    )

    return pairs

def radial_feature_names(
    pairs: list[tuple[str, str]],
    centers: np.ndarray) -> list[str]:
    """
    Return names of all RBF and pair-count features.
    """

    names = []

    for element_a, element_b in pairs:

        pair = canonical_element_pair(
            element_a,
            element_b,
        )

        names.append(
            f"n_pair_{pair[0]}_{pair[1]}"
        )

        for k in range(len(centers)):

            names.append(
                f"rbf_{pair[0]}_"
                f"{pair[1]}_"
                f"{k:02d}"
            )

    return names

def build_element_vocabulary(
    elements: list[str],
    other_label: str = "Other") -> list[str]:
    """
    Build the complete G3 element vocabulary.

    Adds the Other category if it is not already present.
    """

    vocabulary = list(elements)

    if other_label not in vocabulary:
        vocabulary.append(other_label)

    return vocabulary

def element_pair_rbf(
    mol: Chem.Mol,
    coords: np.ndarray,
    centers: np.ndarray,
    gamma: float,
    element_pairs: list[tuple[str, str]],
    elements: list[str],
    normalize: bool = True,
    other_label: str = "Other") -> dict[str, float]:
    """
    Compute element-pair Gaussian radial basis features.

    For every element-pair channel:

    1. Collect all corresponding pairwise distances.
    2. Expand distances through Gaussian RBFs.
    3. Aggregate the radial distribution.
    4. Add the number of contributing atom pairs explicitly.

    Rare/unselected elements are mapped to `other_label`.

    Returns
    -------
    dict[str, float]

    Examples of generated features
    ------------------------------
    rbf_C_O_00
    rbf_C_O_01
    ...
    n_pair_C_O

    rbf_C_Other_00
    ...
    n_pair_C_Other
    """

    pair_distances = get_element_pair_distances(
        mol=mol,
        coords=coords,
        elements=elements,
        other_label=other_label,
    )

    features = {}

    n_centers = len(centers)

    for element_a, element_b in element_pairs:

        pair = canonical_element_pair(
            element_a,
            element_b,
        )

        distances = pair_distances.get(
            pair,
            np.array([], dtype=float),
        )

        # --------------------------------
        # Explicit pair-count information
        # --------------------------------

        n_pairs = distances.size

        pair_count_name = (
            f"n_pair_{pair[0]}_{pair[1]}"
        )

        features[pair_count_name] = float(
            n_pairs
        )

        # --------------------------------
        # Radial distribution information
        # --------------------------------

        if n_pairs == 0:

            aggregated = np.zeros(
                n_centers,
                dtype=float,
            )

        else:

            rbf_values = gaussian_rbf(
                distances=distances,
                centers=centers,
                gamma=gamma,
            )

            aggregated = aggregate_rbf(
                rbf_values=rbf_values,
                normalize=normalize,
            )

        for k, value in enumerate(
            aggregated
        ):

            feature_name = (
                f"rbf_{pair[0]}_"
                f"{pair[1]}_"
                f"{k:02d}"
            )

            features[feature_name] = float(
                value
            )

    return features

def map_element_symbol(
    symbol: str,
    elements: list[str],
    other_label: str = "Other") -> str:
    """
    Map an atomic symbol into the G3 element vocabulary.

    Elements explicitly included in `elements` keep their symbol.
    Every other element is mapped to `other_label`.

    Examples
    --------
    elements = ["C", "N", "O"]

    "C"  -> "C"
    "O"  -> "O"
    "Zn" -> "Other"
    "Fe" -> "Other"
    """

    if symbol in elements:
        return symbol

    return other_label

def calculate_radial_features(
    mol: Chem.Mol,
    conf_id: int,
    elements: list[str],
    min_distance: float = 1.0,
    max_distance: float = 10.0,
    n_centers: int = 20,
    gamma: float = 4.0,
    normalize: bool = True,
    other_label: str = "Other",
) -> dict[str, float]:
    """
    Calculate G3 element-pair radial features for one conformer.

    The representation contains:

    - normalized/summed Gaussian radial basis features;
    - explicit element-pair counts;
    - an Other channel for elements outside the selected vocabulary.
    """

    coords = get_coordinates(
        mol,
        conf_id,
    )

    centers = radial_centers(
        min_distance=min_distance,
        max_distance=max_distance,
        n_centers=n_centers,
    )

    vocabulary = build_element_vocabulary(
        elements,
        other_label=other_label,
    )

    element_pairs = generate_element_pairs(
        vocabulary
    )

    features = element_pair_rbf(
        mol=mol,
        coords=coords,
        centers=centers,
        gamma=gamma,
        element_pairs=element_pairs,
        elements=elements,
        normalize=normalize,
        other_label=other_label,
    )

    return features