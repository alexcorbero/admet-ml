from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(
    mol: Chem.Mol,
    n_conformers: int = 10,
    random_seed: int = 42,
    prune_rms_thresh: float = 0.5,) -> Chem.Mol:
    
    """
    Generate 3D conformers for an RDKit molecule using ETKDG.

    Parameters
    ----------
    mol : Chem.Mol
    Input RDKit molecule.

    n_conformers : int, default=10
        Maximum number of conformers to generate.

    random_seed : int, default=42
        Random seed for reproducibility.

    prune_rms_thresh : float, default=0.5
        RMSD threshold used to discard similar conformers.

    Returns
    -------
    Chem.Mol
        Copy of the molecule containing explicit hydrogens
        and generated 3D conformers.
    """
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_rms_thresh

    conformer_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=n_conformers,
        params=params,
    )

    if len(conformer_ids) == 0:
        raise ValueError("No conformers could be generated.")

    return mol


def optimize_conformers(
    mol: Chem.Mol,
    force_field: str = "MMFF",
    max_iters: int = 500,) -> list[tuple[int, float]]:
    """
    Optimize all conformers and calculate their energies.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule containing 3D conformers.

    force_field : {"MMFF", "UFF"}, default="MMFF"
        Force field used for geometry optimization.

    max_iters : int, default=500
        Maximum number of optimization iterations.

    Returns
    -------
    list of tuple
        Tuples containing:

        (convergence_status, energy)

    for each conformer.
    """

    force_field = force_field.upper()

    if force_field == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise ValueError(
                "MMFF parameters are not available for this molecule."
            )

        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            maxIters=max_iters,
        )

    elif force_field == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise ValueError(
                "UFF parameters are not available for this molecule."
        )

        results = AllChem.UFFOptimizeMoleculeConfs(
            mol,
            maxIters=max_iters,
        )

    else:
        raise ValueError(
            "force_field must be either 'MMFF' or 'UFF'."
        )

    return results


def get_conformer_energies(
    optimization_results: list[tuple[int, float]],) -> list[float]:
    """
    Extract conformer energies from RDKit optimization results.
    """

    return [
        energy
        for _, energy in optimization_results
    ]


def get_lowest_energy_conformer(
    optimization_results: list[tuple[int, float]],) -> int:
    """
    Return the index of the lowest-energy conformer.
    """

    energies = get_conformer_energies(
        optimization_results
    )

    return min(
        range(len(energies)),
        key=energies.__getitem__,
    )

def remove_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
        Removes Hs added in generate_conformers().
        This will reduce nxn representation when doing geometry analysis.
    """
    return Chem.RemoveHs(mol)