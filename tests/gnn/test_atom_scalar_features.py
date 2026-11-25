"""Unit tests for scalar atom features (T2.4)."""

import unittest

from rdkit import Chem

from src.gnn.atom_features import get_atom_scalar_features
from src.gnn.config import AtomicFeaturesConfig


class AtomScalarFeatureTests(unittest.TestCase):
    @staticmethod
    def atom_from_smiles(smiles: str, atom_index: int = 0):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise AssertionError(f"RDKit could not parse test SMILES: {smiles}")
        return molecule.GetAtomWithIdx(atom_index)

    def test_default_features_follow_schema_order(self):
        atom = self.atom_from_smiles("[NH4+]")
        features = get_atom_scalar_features(atom)
        self.assertEqual(features[:3], (1, 0, 4))
        self.assertAlmostEqual(features[3], 14.007, places=3)
        self.assertEqual(features[4:], (0, 0))

    def test_heavy_atom_degree_excludes_hydrogen_and_counts_heavy_neighbors(self):
        atom = self.atom_from_smiles("CCO", atom_index=1)
        features = get_atom_scalar_features(
            atom,
            AtomicFeaturesConfig(
                include_formal_charge=False,
                include_attached_hydrogens=False,
                include_atomic_mass=False,
                include_aromaticity=False,
                include_ring_membership=False,
                include_hybridization=False,
                include_chirality=False,
                include_cip=False,
            ),
        )
        self.assertEqual(features, (2,))

    def test_attached_hydrogens_include_implicit_hydrogens(self):
        atom = self.atom_from_smiles("C")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        self.assertEqual(get_atom_scalar_features(atom, config), (4,))

    def test_atomic_mass_is_returned_as_numeric_value(self):
        atom = self.atom_from_smiles("O")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        features = get_atom_scalar_features(atom, config)
        self.assertEqual(len(features), 1)
        self.assertAlmostEqual(features[0], 15.999, places=3)

    def test_aromaticity_and_ring_membership_are_distinct(self):
        aromatic_atom = self.atom_from_smiles("c1ccccc1")
        aliphatic_ring_atom = self.atom_from_smiles("C1CCCCC1")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        self.assertEqual(get_atom_scalar_features(aromatic_atom, config), (1, 1))
        self.assertEqual(get_atom_scalar_features(aliphatic_ring_atom, config), (0, 1))

    def test_features_can_be_disabled_independently(self):
        atom = self.atom_from_smiles("[NH4+]")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=True,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=True,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        self.assertEqual(get_atom_scalar_features(atom, config), (0, 0))

    def test_all_scalar_features_can_be_disabled(self):
        atom = self.atom_from_smiles("C")
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
        )
        self.assertEqual(get_atom_scalar_features(atom, config), ())

    def test_molecule_context_is_required_for_hydrogen_and_ring_features(self):
        with self.assertRaisesRegex(ValueError, "associated with an RDKit molecule"):
            get_atom_scalar_features(Chem.Atom(6))

    def test_detached_atom_supports_context_independent_features(self):
        atom = Chem.Atom(6)
        config = AtomicFeaturesConfig(
            include_attached_hydrogens=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        features = get_atom_scalar_features(atom, config)
        self.assertEqual(features[:2], (0, 0))
        self.assertAlmostEqual(features[2], 12.011, places=3)
        self.assertEqual(features[3], 0)

    def test_invalid_atom_and_config_types_are_rejected(self):
        with self.assertRaises(TypeError):
            get_atom_scalar_features(object())
        atom = self.atom_from_smiles("C")
        with self.assertRaises(TypeError):
            get_atom_scalar_features(atom, {})


if __name__ == "__main__":
    unittest.main()
