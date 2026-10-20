from rdkit import Chem
import numpy as np

def get_coordinates(
        mol: Chem.Mol,
        conf_id: int) -> np.ndarray: 

    # obtener conformero con conf_id
    conf = mol.GetConformer(conf_id)

    coords = []

    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)

        coords.append([
            pos.x,
            pos.y,
            pos.z,
        ])

    coords = np.array(coords)

    return coords

def distance_matrix(
        coords: np.ndarray) -> np.ndarray:
    n_atoms = len(coords)

    distance_matrix = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(coords[i] - coords[j])

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def geometric_center(
        coords: np.ndarray) -> np.ndarray:

    return np.mean(coords, axis=0)

def _get_atomic_masses(
    mol: Chem.Mol) -> np.ndarray:
    masses = np.array([
        mol.GetAtomWithIdx(i).GetMass()
        for i in range(mol.GetNumAtoms())
    ])

    return masses

def center_of_mass(
        mol: Chem.Mol,
        coords: np.ndarray) -> np.ndarray:

    masses = _get_atomic_masses(mol)

    weighted_coords = masses[:, None] * coords

    weighted_sum = np.sum(weighted_coords, axis = 0)
    total_mass = np.sum(masses)
    
    return weighted_sum / total_mass

def radius_of_gyration(
        mol: Chem.Mol,
        coords: np.ndarray) -> float:

    center = center_of_mass(mol, coords)

    centered_coords = coords - center

    masses = _get_atomic_masses(mol)

    squared_distances = np.sum(
        centered_coords**2,
        axis=1,
    )

    weighted_sum = np.sum(
        masses * squared_distances
    )

    return np.sqrt(
        weighted_sum / np.sum(masses)
    )

def inertia_tensor(
        mol: Chem.Mol,
        coords: np.ndarray) -> np.ndarray:
    
    center = center_of_mass(mol, coords)
    centered_coords = coords - center

    masses = _get_atomic_masses(mol)

    x = centered_coords[:, 0]
    y = centered_coords[:, 1]
    z = centered_coords[:, 2]

    I_xx = np.sum(masses * (y**2 + z**2))
    I_yy = np.sum(masses * (x**2 + z**2))
    I_zz = np.sum(masses * (x**2 + y**2))

    I_xy = -np.sum(masses * x * y)
    I_xz = -np.sum(masses * x * z)
    I_yz = -np.sum(masses * y * z)

    tensor = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ])
    
    return tensor

def principal_moments(
        mol: Chem.Mol,
        coords: np.ndarray) -> np.ndarray:
    tensor = inertia_tensor(mol, coords)
    
    return np.linalg.eigvalsh(tensor)

def gyration_tensor(
    mol: Chem.Mol,
    coords: np.ndarray,
    use_atomic_masses: bool = True) -> np.ndarray:

    if use_atomic_masses:
        weights = _get_atomic_masses(mol)
        center = center_of_mass(
            mol,
            coords,
        )
    else:
        weights = np.ones(len(coords))
        center = geometric_center(coords)

    centered = coords - center

    G = (
        centered.T
        @ (weights[:, None] * centered)
    )

    G = G / np.sum(weights)

    return G

def gyration_moments(
    mol: Chem.Mol,
    coords: np.ndarray,
    use_atomic_masses: bool = True) -> np.ndarray:

    G = gyration_tensor(
        mol,
        coords,
        use_atomic_masses=use_atomic_masses,
    )

    return np.linalg.eigvalsh(G)

def npr1(
    moments: np.ndarray) -> float:

    I1, _, I3 = moments

    if np.isclose(I3, 0.0):
        return 0.0

    return I1 / I3


def npr2(
    moments: np.ndarray) -> float:

    _, I2, I3 = moments

    if np.isclose(I3, 0.0):
        return 0.0

    return I2 / I3


def asphericity(
    moments: np.ndarray) -> float:

    t1, t2, t3 = moments

    numerator = 0.5 * (
        (t1 - t2)**2
        + (t1 - t3)**2
        + (t2 - t3)**2
    )

    denominator = (
        t1 + t2 + t3
    )**2

    if np.isclose(
        denominator,
        0.0,
    ):
        return 0.0

    return numerator / denominator


def eccentricity(
    moments: np.ndarray) -> float:

    I1, _, I3 = moments

    if np.isclose(I3, 0.0):
        return 0.0

    value = I3**2 - I1**2

    value = max(value, 0.0)

    return np.sqrt(value) / I3


def inertial_shape_factor(
    moments: np.ndarray) -> float:

    I1, I2, I3 = moments

    denominator = I1 * I3

    if np.isclose(
        denominator,
        0.0,
    ):
        return 0.0

    return I2 / denominator


def spherocity_index(
    moments: np.ndarray) -> float:

    t1, t2, t3 = moments

    denominator = (
        t1 + t2 + t3
    )

    if np.isclose(
        denominator,
        0.0,
    ):
        return 0.0

    return (
        3 * t1 / denominator
    )


def plane_of_best_fit(
    coords: np.ndarray) -> float:

    center = geometric_center(coords)

    centered = coords - center

    G = (
        centered.T @ centered
    )

    G = G / len(coords)

    _, eigenvectors = np.linalg.eigh(G)

    normal = eigenvectors[:, 0]

    distances = (
        centered @ normal
    )

    distances = np.abs(distances)

    return np.mean(distances)


def calculate_3d_descriptors(
    mol: Chem.Mol,
    conf_id: int) -> dict[str, float]:

    coords = get_coordinates(mol, conf_id)

    inertia_moments = principal_moments(mol, coords)

    gyration_moments_mass = gyration_moments(mol, coords, use_atomic_masses=True)

    gyration_moments_no_mass = gyration_moments(mol, coords, use_atomic_masses=False)

    descriptors = {
        "radius_of_gyration": radius_of_gyration(mol, coords),

        "pmi1": inertia_moments[0],

        "pmi2": inertia_moments[1],

        "pmi3": inertia_moments[2],

        "npr1": npr1(inertia_moments),

        "npr2": npr2(inertia_moments),

        "asphericity": asphericity(gyration_moments_mass),

        "eccentricity":eccentricity(inertia_moments),

        "inertial_shape_factor": inertial_shape_factor(inertia_moments),

        "spherocity_index": spherocity_index(gyration_moments_no_mass),

        "plane_of_best_fit": plane_of_best_fit(coords),
    }

    return descriptors