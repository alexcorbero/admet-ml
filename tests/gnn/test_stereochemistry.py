"""Unit tests for atom chirality and CIP one-hot encoding (T2.3)."""

import unittest

from rdkit import Chem

from src.gnn.atom_features import get_chirality_one_hot, get_cip_one_hot
from src.gnn.config import AtomicFeaturesConfig


class AtomStereochemistryOneHotTests(unittest.TestCase):
    @staticmethod
    def atom_from_smiles(smiles: str, atom_index: int = 1):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise AssertionError(f"RDKit could not parse test SMILES: {smiles}")
        return molecule.GetAtomWithIdx(atom_index)

    def test_unspecified_chirality_is_one_hot(self):
        atom = Chem.MolFromSmiles("CC").GetAtomWithIdx(0)
        self.assertEqual(
            get_chirality_one_hot(atom),
            (1, 0, 0, 0),
        )

    def test_tetrahedral_clockwise_chirality_is_one_hot(self):
        atom = self.atom_from_smiles("N[C@@H](C)C(=O)O")
        self.assertEqual(
            get_chirality_one_hot(atom),
            (0, 1, 0, 0),
        )

    def test_tetrahedral_counterclockwise_chirality_is_one_hot(self):
        atom = self.atom_from_smiles("N[C@H](C)C(=O)O")
        self.assertEqual(
            get_chirality_one_hot(atom),
            (0, 0, 1, 0),
        )

    def test_other_chiral_tag_is_supported(self):
        molecule = Chem.MolFromSmiles("C")
        atom = molecule.GetAtomWithIdx(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_OTHER)
        self.assertEqual(
            get_chirality_one_hot(atom),
            (0, 0, 0, 1),
        )

    def test_custom_chirality_vocabulary_order_is_respected(self):
        atom = self.atom_from_smiles("N[C@@H](C)C(=O)O")
        config = AtomicFeaturesConfig(
            chirality_types=("CHI_OTHER", "CHI_TETRAHEDRAL_CW")
        )
        self.assertEqual(get_chirality_one_hot(atom, config), (0, 1))

    def test_chirality_can_be_disabled(self):
        atom = Chem.MolFromSmiles("CC").GetAtomWithIdx(0)
        config = AtomicFeaturesConfig(include_chirality=False)
        self.assertEqual(get_chirality_one_hot(atom, config), ())

    def test_unconfigured_chiral_tag_is_rejected(self):
        atom = self.atom_from_smiles("N[C@H](C)C(=O)O")
        config = AtomicFeaturesConfig(chirality_types=("CHI_UNSPECIFIED",))
        with self.assertRaisesRegex(ValueError, "CHI_TETRAHEDRAL_CCW"):
            get_chirality_one_hot(atom, config)

    def test_r_cip_label_is_one_hot(self):
        atom = self.atom_from_smiles("N[C@H](C)C(=O)O")
        self.assertEqual(get_cip_one_hot(atom), (1, 0, 0))

    def test_s_cip_label_is_one_hot(self):
        atom = self.atom_from_smiles("N[C@@H](C)C(=O)O")
        self.assertEqual(get_cip_one_hot(atom), (0, 1, 0))

    def test_missing_cip_label_maps_to_unspecified(self):
        atom = Chem.MolFromSmiles("CC").GetAtomWithIdx(0)
        self.assertEqual(get_cip_one_hot(atom), (0, 0, 1))

    def test_detached_atom_has_unspecified_cip_label(self):
        self.assertEqual(get_cip_one_hot(Chem.Atom(6)), (0, 0, 1))

    def test_custom_cip_vocabulary_order_is_respected(self):
        atom = self.atom_from_smiles("N[C@@H](C)C(=O)O")
        config = AtomicFeaturesConfig(cip_types=("UNSPECIFIED", "S", "R"))
        self.assertEqual(get_cip_one_hot(atom, config), (0, 1, 0))

    def test_cip_can_be_disabled(self):
        atom = self.atom_from_smiles("N[C@H](C)C(=O)O")
        config = AtomicFeaturesConfig(include_cip=False)
        self.assertEqual(get_cip_one_hot(atom, config), ())

    def test_unconfigured_cip_label_is_rejected(self):
        atom = self.atom_from_smiles("N[C@H](C)C(=O)O")
        config = AtomicFeaturesConfig(cip_types=("S", "UNSPECIFIED"))
        with self.assertRaisesRegex(ValueError, "R"):
            get_cip_one_hot(atom, config)

    def test_cip_assignment_does_not_mutate_source_molecule(self):
        atom = self.atom_from_smiles("N[C@@H](C)C(=O)O")
        atom.ClearProp("_CIPCode")
        self.assertFalse(atom.HasProp("_CIPCode"))
        self.assertEqual(get_cip_one_hot(atom), (0, 1, 0))
        self.assertFalse(atom.HasProp("_CIPCode"))

    def test_invalid_atom_and_config_types_are_rejected(self):
        with self.assertRaises(TypeError):
            get_chirality_one_hot(object())
        with self.assertRaises(TypeError):
            get_cip_one_hot(object())
        atom = Chem.MolFromSmiles("CC").GetAtomWithIdx(0)
        with self.assertRaises(TypeError):
            get_chirality_one_hot(atom, {})
        with self.assertRaises(TypeError):
            get_cip_one_hot(atom, {})


if __name__ == "__main__":
    unittest.main()
