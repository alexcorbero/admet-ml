"""Focused T2.6 tests for atom-feature edge cases."""

import unittest

import numpy as np
from rdkit import Chem

from src.gnn.atom_features import build_atom_features, get_atom_feature_names


class AtomFeatureEdgeCaseTests(unittest.TestCase):
    def test_explicit_hydrogens_are_kept_as_atoms_and_counted(self):
        molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
        original_atom_count = molecule.GetNumAtoms()

        z, x_chem = build_atom_features(molecule)
        names = get_atom_feature_names()
        hydrogen_count_column = names.index("attached_hydrogens")
        heavy_degree_column = names.index("heavy_atom_degree")

        self.assertEqual(molecule.GetNumAtoms(), original_atom_count)
        self.assertEqual(z.shape, (5,))
        self.assertEqual(int(z[0]), 6)
        hydrogen_indices = np.flatnonzero(z == 1)
        self.assertEqual(len(hydrogen_indices), 4)
        self.assertEqual(float(x_chem[0, hydrogen_count_column]), 4.0)

        for hydrogen_index in hydrogen_indices:
            self.assertEqual(float(x_chem[hydrogen_index, hydrogen_count_column]), 0.0)
            self.assertEqual(float(x_chem[hydrogen_index, heavy_degree_column]), 1.0)

    def test_aromaticity_is_distinct_from_ring_membership(self):
        names = get_atom_feature_names()
        aromaticity_column = names.index("aromaticity")
        ring_column = names.index("ring_membership")

        benzene = Chem.MolFromSmiles("c1ccccc1")
        cyclohexane = Chem.MolFromSmiles("C1CCCCC1")
        _, aromatic_features = build_atom_features(benzene)
        _, aliphatic_features = build_atom_features(cyclohexane)

        np.testing.assert_array_equal(
            aromatic_features[:, [aromaticity_column, ring_column]],
            np.ones((6, 2), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            aliphatic_features[:, [aromaticity_column, ring_column]],
            np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (6, 1)),
        )

    def test_enantiomers_have_consistent_chirality_and_cip_one_hot(self):
        names = get_atom_feature_names()
        chiral_columns = [names.index(name) for name in (
            "chirality_CHI_UNSPECIFIED",
            "chirality_CHI_TETRAHEDRAL_CW",
            "chirality_CHI_TETRAHEDRAL_CCW",
            "chirality_CHI_OTHER",
        )]
        cip_columns = [names.index(name) for name in ("cip_R", "cip_S", "cip_UNSPECIFIED")]

        expected = (
            ("N[C@@H](C)C(=O)O", "chirality_CHI_TETRAHEDRAL_CW", "cip_S"),
            ("N[C@H](C)C(=O)O", "chirality_CHI_TETRAHEDRAL_CCW", "cip_R"),
        )
        for smiles, chirality_name, cip_name in expected:
            molecule = Chem.MolFromSmiles(smiles)
            _, x_chem = build_atom_features(molecule)
            chiral_atom_features = x_chem[1]

            self.assertEqual(float(chiral_atom_features[names.index(chirality_name)]), 1.0)
            self.assertEqual(float(chiral_atom_features[names.index(cip_name)]), 1.0)
            self.assertEqual(float(chiral_atom_features[chiral_columns].sum()), 1.0)
            self.assertEqual(float(chiral_atom_features[cip_columns].sum()), 1.0)

    def test_uncommon_elements_keep_their_atomic_number_and_finite_features(self):
        examples = (
            ("[BH3]", 5),
            ("[SiH4]", 14),
            ("[SeH2]", 34),
            ("[Xe]", 54),
        )
        names = get_atom_feature_names()
        mass_column = names.index("atomic_mass")

        for smiles, expected_atomic_number in examples:
            with self.subTest(smiles=smiles):
                molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
                z, x_chem = build_atom_features(molecule)
                atom = molecule.GetAtomWithIdx(0)

                self.assertEqual(int(z[0]), expected_atomic_number)
                self.assertAlmostEqual(
                    float(x_chem[0, mass_column]),
                    float(atom.GetMass()),
                    places=3,
                )
                self.assertTrue(np.isfinite(x_chem).all())


if __name__ == "__main__":
    unittest.main()
