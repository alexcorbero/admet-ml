"""Unit tests for RDKit covalent edge extraction (T3.1)."""

import unittest

import numpy as np
from rdkit import Chem

from src.gnn.edge_features import build_covalent_edge_index, build_edge_features
from src.gnn.config import GraphConfig


class CovalentEdgeIndexTests(unittest.TestCase):
    def test_each_rdkit_bond_produces_one_endpoint_pair_in_bond_order(self):
        molecule = Chem.MolFromSmiles("CC(=O)O")
        edge_index = build_covalent_edge_index(molecule)

        expected = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in molecule.GetBonds()
        ]
        self.assertEqual(edge_index.shape, (2, molecule.GetNumBonds()))
        self.assertEqual(edge_index.dtype, np.dtype(np.int64))
        self.assertEqual(
            list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
            expected,
        )

    def test_explicit_hydrogen_bonds_are_included_as_rdkit_bonds(self):
        molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
        edge_index = build_covalent_edge_index(molecule)

        self.assertEqual(molecule.GetNumAtoms(), 5)
        self.assertEqual(molecule.GetNumBonds(), 4)
        self.assertEqual(edge_index.shape, (2, 4))
        self.assertTrue(np.all(edge_index[0] == 0))
        self.assertEqual(set(edge_index[1].tolist()), {1, 2, 3, 4})

    def test_bond_endpoints_follow_current_rdkit_atom_indices(self):
        molecule = Chem.MolFromSmiles("CCO")
        order = [2, 0, 1]
        renumbered = Chem.RenumberAtoms(molecule, order)

        edge_index = build_covalent_edge_index(renumbered)
        expected = np.asarray(
            [
                [bond.GetBeginAtomIdx() for bond in renumbered.GetBonds()],
                [bond.GetEndAtomIdx() for bond in renumbered.GetBonds()],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(edge_index, expected)
        self.assertTrue(np.all(edge_index >= 0))
        self.assertTrue(np.all(edge_index < renumbered.GetNumAtoms()))

    def test_single_atom_without_bonds_returns_empty_edge_index(self):
        molecule = Chem.MolFromSmiles("[He]")
        edge_index = build_covalent_edge_index(molecule)

        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_index.dtype, np.dtype(np.int64))

    def test_edge_extraction_does_not_require_a_conformer(self):
        molecule = Chem.MolFromSmiles("C=C")
        self.assertEqual(molecule.GetNumConformers(), 0)

        edge_index = build_covalent_edge_index(molecule)

        self.assertEqual(edge_index.shape, (2, 1))
        self.assertEqual(edge_index.tolist(), [[0], [1]])


    def test_public_builder_exposes_covalent_topology_and_empty_attributes(self):
        molecule = Chem.MolFromSmiles("CCO")
        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=None,
            config=GraphConfig(edge_mode="covalent"),
        )

        self.assertEqual(edge_index.shape, (2, molecule.GetNumBonds()))
        self.assertEqual(edge_attr.shape, (molecule.GetNumBonds(), 0))
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))

    def test_unimplemented_edge_modes_fail_explicitly(self):
        molecule = Chem.MolFromSmiles("CC")
        for edge_mode in ("radius", "hybrid"):
            with self.subTest(edge_mode=edge_mode), self.assertRaises(NotImplementedError):
                build_edge_features(
                    molecule,
                    conf_id=None,
                    config=GraphConfig(edge_mode=edge_mode),
                )

    def test_public_builder_requires_graph_config(self):
        molecule = Chem.MolFromSmiles("CC")
        with self.assertRaises(TypeError):
            build_edge_features(molecule, conf_id=None, config={})

    def test_non_rdkit_molecules_are_rejected(self):
        for value in (None, "CC", object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                build_covalent_edge_index(value)


if __name__ == "__main__":
    unittest.main()
