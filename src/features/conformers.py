import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(
    mol: Chem.Mol,
    n_conformers: int = 5,
    random_seed: int = 42,
    prune_rms_thresh: float = 0.5,
    num_threads: int = 1,
) -> tuple[Chem.Mol, list[int]]:
    """Add explicit hydrogens and generate reproducible ETKDGv3 conformers."""
    if mol is None:
        raise ValueError("mol must not be None.")
    if n_conformers < 1:
        raise ValueError("n_conformers must be at least 1.")
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1.")

    mol = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    params.numThreads = int(num_threads)

    conf_ids = list(AllChem.EmbedMultipleConfs(
        mol, numConfs=int(n_conformers), params=params
    ))
    if not conf_ids:
        raise ValueError("No conformers could be generated.")
    return mol, [int(conf_id) for conf_id in conf_ids]


def optimize_conformers(
    mol: Chem.Mol,
    force_field: str = "MMFF",
    max_iters: int = 200,
    num_threads: int = 1,
) -> list[tuple[int, float]]:
    """Optimize all conformers; status 0 denotes successful convergence."""
    if mol is None:
        raise ValueError("mol must not be None.")
    if max_iters < 1 or num_threads < 1:
        raise ValueError("max_iters and num_threads must be positive.")

    force_field = force_field.upper()
    if force_field == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise ValueError("MMFF parameters are not available for this molecule.")
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=int(num_threads), maxIters=int(max_iters)
        )
    elif force_field == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise ValueError("UFF parameters are not available for this molecule.")
        results = AllChem.UFFOptimizeMoleculeConfs(
            mol, numThreads=int(num_threads), maxIters=int(max_iters)
        )
    else:
        raise ValueError("force_field must be either 'MMFF' or 'UFF'.")
    return [(int(status), float(energy)) for status, energy in results]


def get_conformer_energies(
    optimization_results: list[tuple[int, float]],
) -> list[float]:
    return [float(energy) for _, energy in optimization_results]


def get_lowest_energy_conformer(
    conf_ids: list[int],
    optimization_results: list[tuple[int, float]],
) -> int:
    """Return the RDKit conf_id of the lowest-energy converged finite result."""
    if len(conf_ids) != len(optimization_results):
        raise ValueError("conf_ids and optimization_results must have the same length.")
    if not conf_ids:
        raise ValueError("At least one conformer is required.")

    candidates = []
    for position, result in enumerate(optimization_results):
        try:
            status, energy = result
            energy = float(energy)
        except (TypeError, ValueError, OverflowError):
            continue
        if status == 0 and math.isfinite(energy):
            candidates.append((position, energy))
    if not candidates:
        raise ValueError("No converged conformer with a finite energy is available.")
    best_position, _ = min(candidates, key=lambda item: item[1])
    return int(conf_ids[best_position])


def keep_only_conformer(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
    """Return a copy containing exactly the selected RDKit conformer ID."""
    if mol is None:
        raise ValueError("mol must not be None.")
    available_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
    if int(conf_id) not in available_ids:
        raise ValueError(f"Conformer {conf_id} is not present in the molecule.")
    selected = Chem.Mol(mol)
    for conformer in list(selected.GetConformers()):
        if int(conformer.GetId()) != int(conf_id):
            selected.RemoveConformer(int(conformer.GetId()))
    if [int(conf.GetId()) for conf in selected.GetConformers()] != [int(conf_id)]:
        raise RuntimeError("Failed to retain exactly the selected conformer.")
    return selected


def validate_conformer_geometry(
    mol: Chem.Mol,
    conf_id: int,
    min_pairwise_distance: float = 0.05,
) -> np.ndarray:
    """Return finite coordinates, rejecting malformed or overlapping geometries."""
    if mol is None:
        raise ValueError("mol must not be None.")
    if not math.isfinite(float(min_pairwise_distance)) or min_pairwise_distance < 0:
        raise ValueError("min_pairwise_distance must be finite and non-negative.")
    try:
        positions = np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), dtype=float)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Conformer {conf_id} is not present in the molecule.") from exc

    expected = (mol.GetNumAtoms(), 3)
    if positions.shape != expected:
        raise ValueError(f"Coordinates must have shape {expected}; got {positions.shape}.")
    if not np.isfinite(positions).all():
        raise ValueError("Conformer contains non-finite coordinates.")
    if positions.shape[0] < 2:
        raise ValueError("A valid molecular geometry requires at least two atoms.")

    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    pair_distances = distances[np.triu_indices(positions.shape[0], k=1)]
    if not np.isfinite(pair_distances).all():
        raise ValueError("Pairwise distances contain NaN or Inf.")
    if np.any(pair_distances < float(min_pairwise_distance)):
        raise ValueError(
            "Degenerate conformer: two atoms are closer than "
            f"{min_pairwise_distance} Angstrom."
        )
    return positions


def remove_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Remove hydrogens; do not use for G7 conformer preparation."""
    return Chem.RemoveHs(mol)
