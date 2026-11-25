"""Unit tests for the T2.5 atom-feature matrix builder."""

import unittest

import numpy as np
from rdkit import Chem

from src.gnn.atom_features import (
    build_atom_features,
    get_atom_feature_names,
    get_atom_scalar_features,
    get_chirality_one_hot,
    get_cip_one_hot,
    get_hybridization_one_hot,
)
from src.gnn.config import AtomicFeaturesConfig


class BuildAtomFeaturesTests(unittest.TestCase):
    def test_default_outputs_have_expected_shapes_and_dtypes(self):
        molecule = Chem.MolFromSmiles("CO")
        z, x_chem = build_atom_features(molecule)

        self.assertIsInstance(z, np.ndarray)
        self.assertIsInstance(x_chem, np.ndarray)
        self.assertEqual(z.shape, (molecule.GetNumAtoms(),))
        self.assertEqual(x_chem.shape, (molecule.GetNumAtoms(), len(get_atom_feature_names())))
        self.assertEqual(z.dtype, np.dtype(np.int64))
        self.assertEqual(x_chem.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(x_chem).all())

    def test_atomic_numbers_follow_rdkit_atom_order(self):
        molecule = Chem.MolFromSmiles("CON")
        z, _ = build_atom_features(molecule)
        self.assertEqual(
            z.tolist(),
            [molecule.GetAtomWithIdx(i).GetAtomicNum() for i in range(molecule.GetNumAtoms())],
        )

    def test_rows_compose_existing_encoders_in_schema_order(self):
        molecule = Chem.MolFromSmiles("CO")
        _, x_chem = build_atom_features(molecule)

        for atom_index in range(molecule.GetNumAtoms()):
            atom = molecule.GetAtomWithIdx(atom_index)
            expected = (
                get_atom_scalar_features(atom)
                + get_hybridization_one_hot(atom)
                + get_chirality_one_hot(atom)
                + get_cip_one_hot(atom)
            )
            np.testing.assert_array_equal(
                x_chem[atom_index],
                np.asarray(expected, dtype=np.float32),
            )

    def test_custom_config_controls_feature_width_and_order(self):
        molecule = Chem.MolFromSmiles("C=C")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            hybridization_types=("SP3", "SP2"),
            include_chirality=False,
            include_cip=False,
        )
        z, x_chem = build_atom_features(molecule, config)

        self.assertEqual(z.tolist(), [6, 6])
        self.assertEqual(x_chem.shape, (2, 2))
        np.testing.assert_array_equal(x_chem, np.asarray([[0, 1], [0, 1]], dtype=np.float32))

    def test_all_features_can_be_disabled(self):
        molecule = Chem.MolFromSmiles("CO")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        z, x_chem = build_atom_features(molecule, config)
        self.assertEqual(z.shape, (2,))
        self.assertEqual(x_chem.shape, (2, 0))

    def test_empty_molecule_returns_two_dimensional_empty_feature_matrix(self):
        z, x_chem = build_atom_features(Chem.Mol())
        self.assertEqual(z.shape, (0,))
        self.assertEqual(x_chem.shape, (0, len(get_atom_feature_names())))

    def test_builder_does_not_add_or_reorder_atoms(self):
        molecule = Chem.MolFromSmiles("C")
        original_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
        original_count = molecule.GetNumAtoms()

        z, _ = build_atom_features(molecule)

        self.assertEqual(molecule.GetNumAtoms(), original_count)
        self.assertEqual(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            original_numbers,
        )
        self.assertEqual(z.tolist(), original_numbers)

    def test_invalid_molecule_and_config_are_rejected(self):
        with self.assertRaises(TypeError):
            build_atom_features(None)
        with self.assertRaises(TypeError):
            build_atom_features(Chem.MolFromSmiles("C"), {})


if __name__ == "__main__":
    unittest.main()
