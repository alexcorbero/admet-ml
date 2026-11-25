"""Unit tests for RDKit hybridization one-hot encoding (T2.2)."""

import unittest

from rdkit import Chem

from src.gnn.atom_features import get_hybridization_one_hot
from src.gnn.config import AtomicFeaturesConfig


class HybridizationOneHotTests(unittest.TestCase):
    @staticmethod
    def atom_from_smiles(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise AssertionError(f"RDKit could not parse test SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        return mol.GetAtomWithIdx(0)

    def test_sp3_hybridization_is_one_hot(self):
        atom = self.atom_from_smiles("C")
        self.assertEqual(
            get_hybridization_one_hot(atom),
            (0, 0, 0, 1, 0, 0, 0),
        )

    def test_sp2_hybridization_is_one_hot(self):
        atom = self.atom_from_smiles("C=C")
        self.assertEqual(
            get_hybridization_one_hot(atom),
            (0, 0, 1, 0, 0, 0, 0),
        )

    def test_sp_hybridization_is_one_hot(self):
        atom = self.atom_from_smiles("C#C")
        self.assertEqual(
            get_hybridization_one_hot(atom),
            (0, 1, 0, 0, 0, 0, 0),
        )

    def test_unspecified_hybridization_is_one_hot(self):
        atom = Chem.Atom(6)
        self.assertEqual(
            get_hybridization_one_hot(atom),
            (0, 0, 0, 0, 0, 0, 1),
        )

    def test_custom_vocabulary_order_is_respected(self):
        atom = self.atom_from_smiles("C=C")
        config = AtomicFeaturesConfig(hybridization_types=("SP3", "SP2", "SP"))
        self.assertEqual(
            get_hybridization_one_hot(atom, config),
            (0, 1, 0),
        )

    def test_hybridization_can_be_disabled(self):
        atom = self.atom_from_smiles("C")
        config = AtomicFeaturesConfig(include_hybridization=False)
        self.assertEqual(get_hybridization_one_hot(atom, config), ())

    def test_unconfigured_hybridization_is_rejected(self):
        atom = self.atom_from_smiles("C=C")
        config = AtomicFeaturesConfig(hybridization_types=("SP3",))
        with self.assertRaisesRegex(ValueError, "SP2"):
            get_hybridization_one_hot(atom, config)

    def test_invalid_atom_type_is_rejected(self):
        with self.assertRaises(TypeError):
            get_hybridization_one_hot(object())

    def test_invalid_config_type_is_rejected(self):
        atom = self.atom_from_smiles("C")
        with self.assertRaises(TypeError):
            get_hybridization_one_hot(atom, {})


if __name__ == "__main__":
    unittest.main()
