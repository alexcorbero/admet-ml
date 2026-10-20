from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(
    mol: Chem.Mol,
    n_conformers: int = 5,
    random_seed: int = 42,
    prune_rms_thresh: float = 0.5,
) -> tuple[Chem.Mol, list[int]]:

    mol = Chem.AddHs(Chem.Mol(mol))

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_rms_thresh
    params.numThreads = 0

    conf_ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_conformers,
            params=params,
        )
    )

    if not conf_ids:
        raise ValueError("No conformers could be generated.")

    return mol, conf_ids


def optimize_conformers(
    mol: Chem.Mol,
    force_field: str = "MMFF",
    max_iters: int = 200,
    num_threads: int = 0) -> list[tuple[int, float]]:
    """
    Optimize all conformers and calculate their energies.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule containing 3D conformers.

    force_field : {"MMFF", "UFF"}, default="MMFF"
        Force field used for geometry optimization.

    max_iters : int, default=200
        Maximum number of optimization iterations.

    num_threads : int, default=0
        Number of threads used by RDKit.
        0 means use all available threads.

    Returns
    -------
    list of tuple
        (convergence_status, energy) for each conformer.
    """

    force_field = force_field.upper()

    if force_field == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise ValueError(
                "MMFF parameters are not available for this molecule."
            )

        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=num_threads,
            maxIters=max_iters,
        )

    elif force_field == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise ValueError(
                "UFF parameters are not available for this molecule."
            )

        results = AllChem.UFFOptimizeMoleculeConfs(
            mol,
            numThreads=num_threads,
            maxIters=max_iters,
        )

    else:
        raise ValueError(
            "force_field must be either 'MMFF' or 'UFF'."
        )

    return list(results)

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