"""Tests for the T7.1 scalable-column schema."""

import unittest

from src.gnn.config import AtomicFeaturesConfig, GNNConfig, GraphConfig
from src.gnn.normalizer import (
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


if __name__ == "__main__":
    unittest.main()
