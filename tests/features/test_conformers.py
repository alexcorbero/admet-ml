import ast
import json
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from src.features.conformers import (
    generate_conformers,
    get_lowest_energy_conformer,
    keep_only_conformer,
    optimize_conformers,
    validate_conformer_geometry,
)


class ConformerUtilityTests(unittest.TestCase):
    def make_molecule(self, smiles="CCO", count=3, seed=42):
        return generate_conformers(
            Chem.MolFromSmiles(smiles), n_conformers=count,
            random_seed=seed, num_threads=1,
        )

    def test_embedding_adds_hydrogens_and_returns_real_ids(self):
        mol, ids = self.make_molecule(count=2)
        self.assertGreater(mol.GetNumAtoms(), 3)
        self.assertEqual(ids, [conf.GetId() for conf in mol.GetConformers()])
        self.assertEqual(len(ids), 2)

    def test_embedding_is_reproducible(self):
        mol_a, ids_a = self.make_molecule(seed=1234)
        mol_b, ids_b = self.make_molecule(seed=1234)
        self.assertEqual(ids_a, ids_b)
        for conf_id in ids_a:
            np.testing.assert_allclose(
                mol_a.GetConformer(conf_id).GetPositions(),
                mol_b.GetConformer(conf_id).GetPositions(),
                rtol=0.0, atol=1e-6,
            )

    def test_optimization_and_selection(self):
        mol, ids = self.make_molecule()
        results = optimize_conformers(mol, force_field="MMFF", num_threads=1)
        self.assertEqual(len(results), len(ids))
        best_id = get_lowest_energy_conformer(ids, results)
        self.assertIn(best_id, ids)
        self.assertTrue(np.isfinite(mol.GetConformer(best_id).GetPositions()).all())

    def test_selection_ignores_nonconverged_and_nonfinite_results(self):
        self.assertEqual(
            get_lowest_energy_conformer(
                [7, 11, 15, 20],
                [(0, 3.0), (1, -100.0), (0, -2.0), (0, float("nan"))],
            ),
            15,
        )

    def test_selection_rejects_misaligned_or_empty_results(self):
        with self.assertRaises(ValueError):
            get_lowest_energy_conformer([0, 1], [(0, 1.0)])
        with self.assertRaises(ValueError):
            get_lowest_energy_conformer([], [])

    def test_selection_rejects_no_converged_finite_candidate(self):
        with self.assertRaisesRegex(ValueError, "No converged conformer"):
            get_lowest_energy_conformer([0, 1], [(1, -5.0), (0, float("inf"))])

    def test_geometry_validation_accepts_valid_geometry(self):
        mol, ids = self.make_molecule(count=1)
        coords = validate_conformer_geometry(mol, ids[0])
        self.assertEqual(coords.shape, (mol.GetNumAtoms(), 3))
        self.assertTrue(np.isfinite(coords).all())

    def test_geometry_validation_rejects_nonfinite_coordinates(self):
        mol, ids = self.make_molecule(count=1)
        mol.GetConformer(ids[0]).SetAtomPosition(0, Point3D(float("nan"), 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_conformer_geometry(mol, ids[0])

    def test_geometry_validation_rejects_overlapping_atoms(self):
        mol, ids = self.make_molecule(smiles="CC", count=1)
        conf = mol.GetConformer(ids[0])
        conf.SetAtomPosition(1, conf.GetAtomPosition(0))
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            validate_conformer_geometry(mol, ids[0])

    def test_keep_only_conformer_preserves_rdkit_id_and_hydrogens(self):
        mol, ids = self.make_molecule(count=3)
        selected_id = ids[-1]
        selected = keep_only_conformer(mol, selected_id)
        self.assertEqual(selected.GetNumConformers(), 1)
        self.assertEqual(selected.GetConformer().GetId(), selected_id)
        self.assertEqual(selected.GetNumAtoms(), mol.GetNumAtoms())
        self.assertTrue(any(atom.GetAtomicNum() == 1 for atom in selected.GetAtoms()))


class ConformerNotebookContractTests(unittest.TestCase):
    def test_notebook_is_valid_and_does_not_implement_later_tasks(self):
        root = Path(__file__).resolve().parents[2]
        notebook = json.loads(
            (root / "notebooks" / "prepare_aqsoldb_conformers.ipynb").read_text(encoding="utf-8")
        )
        self.assertEqual(notebook["nbformat"], 4)
        code = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for source in code:
            ast.parse(source)
        source = "\n".join(code)
        for required in (
            "generate_conformers", "optimize_conformers",
            "get_lowest_energy_conformer", "validate_conformer_geometry",
            "keep_only_conformer", "conformers.pkl", "manifest.json", "excluded.csv",
        ):
            self.assertIn(required, source)
        for forbidden in ("build_graph", "GNNConfig", "DataLoader", "optimizer.step(", "backward("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
