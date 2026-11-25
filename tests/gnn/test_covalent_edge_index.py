"""Unit tests for RDKit covalent, radius, and hybrid edge features (T3.1-T3.5)."""

import unittest
from unittest.mock import patch

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

    def test_public_builder_returns_bidirectional_covalent_edges(self):
        molecule = Chem.MolFromSmiles("CCO")
        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=None,
            config=GraphConfig(edge_mode="covalent"),
        )

        np.testing.assert_array_equal(
            edge_index,
            np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        )
        self.assertEqual(edge_attr.shape, (4, 25))
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

        self.assertEqual(edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(edge_attr.shape, (2, 25))
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

        # Bond 0-1 lies outside the cutoff but is retained. Pair 0-2 is
        # spatially close despite not being covalently bonded. Every pair
        # appears once in each direction.
        np.testing.assert_array_equal(
            edge_index,
            np.asarray(
                [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
                dtype=np.int64,
            ),
        )
        self.assertEqual(edge_attr.shape, (6, 25))
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))

    def test_hybrid_requires_the_selected_conformer(self):
        molecule = Chem.MolFromSmiles("CC")
        with self.assertRaisesRegex(ValueError, "conf_id is required"):
            build_edge_features(
                molecule,
                conf_id=None,
                config=GraphConfig(edge_mode="hybrid"),
            )

    def test_hybrid_deduplicates_overlap_and_expands_both_directions(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )

        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=conf_id,
            config=GraphConfig(edge_mode="hybrid", cutoff=2.0),
        )

        # The pair is present in both sources but occurs only once per
        # direction in the final edge index.
        self.assertEqual(edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(edge_attr.shape, (2, 25))


    def test_public_builder_removes_self_loops_and_reversed_duplicates(self):
        molecule = Chem.MolFromSmiles("CC")
        raw_edges = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.int64)

        with patch(
            "src.gnn.edge_features.build_covalent_edge_index",
            return_value=raw_edges,
        ):
            edge_index, edge_attr = build_edge_features(
                molecule,
                conf_id=None,
                config=GraphConfig(edge_mode="covalent"),
            )

        self.assertEqual(edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(edge_attr.shape, (2, 25))

    def test_radius_public_builder_preserves_empty_bidirectional_shape(self):
        molecule, conf_id = molecule_with_conformer(
            "CC",
            [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        )

        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=conf_id,
            config=GraphConfig(edge_mode="radius", cutoff=1.0),
        )

        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_index.dtype, np.dtype(np.int64))
        self.assertEqual(edge_attr.shape, (0, 25))


class ChemicalEdgeFeatureTests(unittest.TestCase):
    def _features_by_edge(self, molecule, config):
        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=None,
            config=config,
        )
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))
        self.assertEqual(edge_attr.shape, (edge_index.shape[1], 25))
        self.assertTrue(np.isfinite(edge_attr).all())
        return {
            (int(source), int(target)): features
            for (source, target), features in zip(edge_index.T, edge_attr)
        }

    def test_single_double_triple_and_aromatic_bonds_are_one_hot(self):
        cases = (
            ("CC", 2),
            ("C=C", 3),
            ("C#C", 4),
            ("c1ccccc1", 13),
        )
        for smiles, bond_type_column in cases:
            with self.subTest(smiles=smiles):
                molecule = Chem.MolFromSmiles(smiles)
                features = self._features_by_edge(
                    molecule,
                    GraphConfig(edge_mode="covalent"),
                )
                forward = features[(0, 1)]
                reverse = features[(1, 0)]

                self.assertEqual(forward[0], 1.0)  # is_bonded
                self.assertEqual(float(forward[1:22].sum()), 1.0)
                self.assertEqual(forward[bond_type_column], 1.0)
                np.testing.assert_array_equal(forward, reverse)

    def test_aromaticity_conjugation_and_ring_membership_are_encoded(self):
        aromatic = self._features_by_edge(
            Chem.MolFromSmiles("c1ccccc1"),
            GraphConfig(edge_mode="covalent"),
        )[(0, 1)]
        self.assertEqual(aromatic[22], 1.0)
        self.assertEqual(aromatic[23], 1.0)
        self.assertEqual(aromatic[24], 1.0)

        conjugated = self._features_by_edge(
            Chem.MolFromSmiles("C=CC=C"),
            GraphConfig(edge_mode="covalent"),
        )[(0, 1)]
        self.assertEqual(conjugated[23], 1.0)
        self.assertEqual(conjugated[22], 0.0)
        self.assertEqual(conjugated[24], 0.0)

        ring = self._features_by_edge(
            Chem.MolFromSmiles("C1CCCCC1"),
            GraphConfig(edge_mode="covalent"),
        )[(0, 1)]
        self.assertEqual(ring[24], 1.0)
        self.assertEqual(ring[22], 0.0)

    def test_spatial_only_edges_have_zero_chemical_attributes(self):
        molecule, conf_id = molecule_with_conformer(
            "CCO",
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        )
        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=conf_id,
            config=GraphConfig(edge_mode="hybrid", cutoff=2.0),
        )
        edge_features = {
            (int(source), int(target)): features
            for (source, target), features in zip(edge_index.T, edge_attr)
        }

        self.assertEqual(edge_features[(0, 1)][0], 1.0)
        self.assertEqual(edge_features[(1, 2)][0], 1.0)
        np.testing.assert_array_equal(edge_features[(0, 2)], np.zeros(25, dtype=np.float32))
        np.testing.assert_array_equal(edge_features[(2, 0)], np.zeros(25, dtype=np.float32))

    def test_both_directions_receive_identical_bond_features(self):
        features = self._features_by_edge(
            Chem.MolFromSmiles("CC"),
            GraphConfig(edge_mode="covalent"),
        )
        np.testing.assert_array_equal(features[(0, 1)], features[(1, 0)])

    def test_bond_features_can_be_disabled_without_changing_topology(self):
        molecule = Chem.MolFromSmiles("CC")
        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=None,
            config=GraphConfig(
                edge_mode="covalent",
                include_bond_features=False,
            ),
        )

        self.assertEqual(edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(edge_attr.shape, (2, 0))
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))

    def test_empty_topology_returns_empty_chemical_feature_matrix(self):
        molecule = Chem.MolFromSmiles("[He]")
        edge_index, edge_attr = build_edge_features(
            molecule,
            conf_id=None,
            config=GraphConfig(edge_mode="covalent"),
        )

        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_attr.shape, (0, 25))
        self.assertEqual(edge_attr.dtype, np.dtype(np.float32))




if __name__ == "__main__":
    unittest.main()
