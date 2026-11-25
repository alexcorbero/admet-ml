"""Unit tests for RDKit covalent, radius, and hybrid edge topology (T3.1-T3.3)."""

import unittest

import numpy as np
from rdkit import Chem

from src.gnn.config import GraphConfig
from src.gnn.edge_features import (
    build_covalent_edge_index,
    build_edge_features,
    build_radius_edge_index,
)


def molecule_with_conformer(smiles, positions, conf_id=0):
    molecule = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index, position in enumerate(positions):
        conformer.SetAtomPosition(atom_index, position)
    conformer.SetId(conf_id)
    assigned_id = molecule.AddConformer(conformer, assignId=False)
    return molecule, assigned_id


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

    def test_public_builder_requires_graph_config(self):
        molecule = Chem.MolFromSmiles("CC")
        with self.assertRaises(TypeError):
            build_edge_features(molecule, conf_id=None, config={})

    def test_non_rdkit_molecules_are_rejected(self):
        for value in (None, "CC", object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                build_covalent_edge_index(value)


class RadiusEdgeIndexTests(unittest.TestCase):
    def test_only_pairs_within_cutoff_are_returned(self):
        molecule, conf_id = molecule_with_conformer(
            "CCC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)],
        )

        edge_index = build_radius_edge_index(molecule, conf_id, cutoff=1.5)

        np.testing.assert_array_equal(edge_index, np.asarray([[0], [1]]))
        self.assertEqual(edge_index.dtype, np.dtype(np.int64))

    def test_cutoff_boundary_is_inclusive(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        )

        edge_index = build_radius_edge_index(molecule, conf_id, cutoff=2.0)

        np.testing.assert_array_equal(edge_index, np.asarray([[0], [1]]))

    def test_selected_rdkit_conf_id_is_used_not_conformer_position(self):
        molecule, first_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            conf_id=11,
        )
        second = Chem.Conformer(molecule.GetNumAtoms())
        second.SetAtomPosition(0, (0.0, 0.0, 0.0))
        second.SetAtomPosition(1, (10.0, 0.0, 0.0))
        second.SetId(23)
        second_id = molecule.AddConformer(second, assignId=False)

        self.assertEqual(first_id, 11)
        self.assertEqual(second_id, 23)
        first_edges = build_radius_edge_index(molecule, first_id, cutoff=2.0)
        second_edges = build_radius_edge_index(molecule, second_id, cutoff=2.0)

        self.assertEqual(first_edges.tolist(), [[0], [1]])
        self.assertEqual(second_edges.shape, (2, 0))

    def test_each_unordered_pair_is_emitted_once_without_reverse_edge(self):
        molecule, conf_id = molecule_with_conformer(
            "CCC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        )

        edge_index = build_radius_edge_index(molecule, conf_id, cutoff=1.1)

        self.assertEqual(edge_index.tolist(), [[0, 1], [1, 2]])
        self.assertTrue(np.all(edge_index[0] < edge_index[1]))

    def test_no_neighbors_returns_empty_edge_index(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        )

        edge_index = build_radius_edge_index(molecule, conf_id, cutoff=1.0)

        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_index.dtype, np.dtype(np.int64))

    def test_invalid_conformer_id_and_missing_conformer_are_rejected(self):
        molecule = Chem.MolFromSmiles("CC")
        for conf_id in (None, -1, "0", True):
            with self.subTest(conf_id=conf_id), self.assertRaises(ValueError):
                build_radius_edge_index(molecule, conf_id, cutoff=2.0)

    def test_negative_conf_id_is_rejected_when_conformers_exist(self):
        molecule, valid_conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            conf_id=5,
        )
        self.assertEqual(valid_conf_id, 5)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_radius_edge_index(molecule, -1, cutoff=2.0)

    def test_non_finite_conformer_coordinates_are_rejected(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0)],
        )

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            build_radius_edge_index(molecule, conf_id, cutoff=2.0)

    def test_cutoff_must_be_finite_and_positive(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )
        for cutoff in (0.0, -1.0, float("nan"), float("inf"), True, "bad"):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                build_radius_edge_index(molecule, conf_id, cutoff)

    def test_public_builder_uses_configured_radius_cutoff(self):
        molecule, conf_id = molecule_with_conformer(
            "CCC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
        )

        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=conf_id,
            config=GraphConfig(edge_mode="radius", cutoff=1.5),
        )

        self.assertEqual(edge_index.tolist(), [[0], [1]])
        self.assertEqual(edge_attr.shape, (1, 0))
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))

    def test_radius_mode_requires_a_conformer_id(self):
        molecule = Chem.MolFromSmiles("CC")
        with self.assertRaisesRegex(ValueError, "conf_id is required"):
            build_edge_features(
                molecule,
                conf_id=None,
                config=GraphConfig(edge_mode="radius"),
            )



class HybridEdgeIndexTests(unittest.TestCase):
    def test_hybrid_keeps_all_covalent_bonds_and_adds_spatial_neighbors(self):
        molecule, conf_id = molecule_with_conformer(
            "CCO",
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )

        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=conf_id,
            config=GraphConfig(edge_mode="hybrid", cutoff=2.0),
        )

        # Bond 0-1 lies outside the cutoff but must be retained. Pair 0-2
        # is spatially close despite not being covalently bonded.
        np.testing.assert_array_equal(
            edge_index,
            np.asarray([[0, 1, 0], [1, 2, 2]], dtype=np.int64),
        )
        self.assertEqual(edge_attr.shape, (3, 0))
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))

    def test_hybrid_requires_the_selected_conformer(self):
        molecule = Chem.MolFromSmiles("CC")
        with self.assertRaisesRegex(ValueError, "conf_id is required"):
            build_edge_features(
                molecule,
                conf_id=None,
                config=GraphConfig(edge_mode="hybrid"),
            )

    def test_hybrid_concatenates_sources_before_t34_deduplication(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )

        edge_index, _ = build_edge_features(
            molecule,
            conf_id=conf_id,
            config=GraphConfig(edge_mode="hybrid", cutoff=2.0),
        )

        # The same pair is present in both source topologies. T3.4 owns
        # duplicate removal and bidirectional expansion.
        self.assertEqual(edge_index.tolist(), [[0, 0], [1, 1]])




if __name__ == "__main__":
    unittest.main()
