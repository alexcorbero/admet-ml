"""Tests for the T7.1 scalable-column schema."""

import unittest
from types import SimpleNamespace

import numpy as np

from src.gnn.config import AtomicFeaturesConfig, GNNConfig, GraphConfig
from src.gnn.normalizer import (
    GraphFeatureNormalizer,
    get_scalable_feature_columns,
    get_scalable_feature_names,
)


class ScalableFeatureColumnsTests(unittest.TestCase):
    def test_default_columns_include_only_numeric_raw_features(self):
        columns = get_scalable_feature_columns(GNNConfig())

        self.assertEqual(
            columns["x_chem"],
            (
                "formal_charge",
                "heavy_atom_degree",
                "attached_hydrogens",
                "atomic_mass",
            ),
        )
        self.assertEqual(columns["x_dg"], ())
        self.assertEqual(columns["edge_attr"][0], "distance")
        self.assertEqual(len(columns["edge_attr"]), 33)
        self.assertNotIn("z", columns["x_chem"])

    def test_excludes_flags_masks_and_disabled_columns(self):
        config = GNNConfig(
            graph=GraphConfig(
                include_bond_features=True,
                include_distance=False,
                include_rbf=False,
            ),
            atomic_features=AtomicFeaturesConfig(
                include_formal_charge=False,
                include_heavy_atom_degree=False,
                include_attached_hydrogens=False,
                include_atomic_mass=False,
                density_enabled=True,
                curvature_enabled=True,
            ),
        )

        columns = get_scalable_feature_columns(config)

        self.assertEqual(columns["x_chem"], ())
        self.assertTrue(columns["x_dg"])
        self.assertTrue(all(not name.endswith("curvature_valid") for name in columns["x_dg"]))
        self.assertEqual(columns["edge_attr"], ())
        self.assertNotIn("is_bonded", columns["edge_attr"])
        self.assertNotIn("bond_is_aromatic", columns["edge_attr"])

    def test_alias_has_same_contract(self):
        config = GNNConfig()
        self.assertEqual(
            get_scalable_feature_names(config),
            get_scalable_feature_columns(config),
        )

    def test_rejects_invalid_config(self):
        with self.assertRaises(TypeError):
            get_scalable_feature_columns(None)

    def test_fit_uses_only_train_graph_feature_values(self):
        config = GNNConfig(
            graph=GraphConfig(
                include_bond_features=False,
                include_distance=True,
                include_rbf=False,
            ),
            atomic_features=AtomicFeaturesConfig(
                include_aromaticity=False,
                include_ring_membership=False,
                include_hybridization=False,
                include_chirality=False,
                include_cip=False,
            ),
        )
        train_graphs = [
            SimpleNamespace(
                x_chem=np.array([[1.0, 2.0, 3.0, 10.0], [3.0, 4.0, 5.0, 14.0]]),
                x_dg=np.empty((2, 0)),
                edge_attr=np.array([[1.0], [3.0]]),
                y=np.array([999.0]),
            ),
            SimpleNamespace(
                x_chem=np.array([[5.0, 6.0, 7.0, 18.0]]),
                x_dg=np.empty((1, 0)),
                edge_attr=np.array([[5.0]]),
                y=np.array([-999.0]),
            ),
        ]

        normalizer = GraphFeatureNormalizer.from_config(config)
        result = normalizer.fit(train_graphs)

        self.assertIs(result, normalizer)
        self.assertTrue(normalizer.fitted)
        self.assertEqual(normalizer.statistics["x_chem"].count, 3)
        np.testing.assert_allclose(
            normalizer.statistics["x_chem"].mean,
            [3.0, 4.0, 5.0, 14.0],
        )
        np.testing.assert_allclose(
            normalizer.statistics["x_chem"].std,
            np.sqrt([8.0 / 3.0] * 3 + [32.0 / 3.0]),
        )
        np.testing.assert_allclose(normalizer.statistics["edge_attr"].mean, [3.0])
        np.testing.assert_allclose(
            normalizer.statistics["edge_attr"].std,
            [np.sqrt(8.0 / 3.0)],
        )
        self.assertNotIn("x_dg", normalizer.statistics)

    def test_fit_rejects_empty_or_nonfinite_training_values(self):
        config = GNNConfig(
            graph=GraphConfig(include_bond_features=False, include_rbf=False),
            atomic_features=AtomicFeaturesConfig(
                include_aromaticity=False,
                include_ring_membership=False,
                include_hybridization=False,
                include_chirality=False,
                include_cip=False,
            ),
        )
        normalizer = GraphFeatureNormalizer.from_config(config)
        with self.assertRaisesRegex(ValueError, "at least one graph"):
            normalizer.fit([])

        bad_graph = SimpleNamespace(
            x_chem=np.array([[np.nan, 2.0, 3.0, 10.0]]),
            x_dg=np.empty((1, 0)),
            edge_attr=np.array([[1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            normalizer.fit([bad_graph])


if __name__ == "__main__":
    unittest.main()
