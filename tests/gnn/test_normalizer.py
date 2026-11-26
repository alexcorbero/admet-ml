"""Tests for the T7.1 scalable-column schema."""

import unittest
from types import SimpleNamespace

import numpy as np

from src.gnn.config import AtomicFeaturesConfig, GNNConfig, GraphConfig
from src.gnn.density_features import get_density_feature_names
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

    def test_fit_excludes_invalid_curvature_rows_from_statistics(self):
        atomic_config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
            density_enabled=True,
            curvature_enabled=True,
            density_elements=("C",),
            dg_kernel_configs=((2.0, 1.0),),
        )
        config = GNNConfig(
            graph=GraphConfig(
                include_bond_features=False,
                include_distance=False,
                include_rbf=False,
            ),
            atomic_features=atomic_config,
        )
        names = get_density_feature_names(atomic_config)
        x_dg = np.zeros((2, len(names)))
        curvature_name = next(name for name in names if name.startswith("H_"))
        mask_name = f"curvature_valid_{curvature_name[2:]}"
        curvature_index = names.index(curvature_name)
        mask_index = names.index(mask_name)
        x_dg[:, curvature_index] = [100.0, 1.0]
        for name_index, name in enumerate(names):
            if name.startswith("curvature_valid_"):
                x_dg[:, name_index] = 1.0
        x_dg[:, mask_index] = [0.0, 1.0]

        statistics = GraphFeatureNormalizer.from_config(config).fit(
            [
                SimpleNamespace(
                    x_chem=np.empty((2, 0)),
                    x_dg=x_dg,
                    edge_attr=np.empty((2, 0)),
                )
            ]
        ).statistics["x_dg"]

        statistic_index = statistics.columns.index(curvature_name)
        self.assertEqual(statistics.count[statistic_index], 1)
        self.assertEqual(statistics.mean[statistic_index], 1.0)
        self.assertEqual(statistics.std[statistic_index], 0.0)

    def test_fit_rejects_non_binary_curvature_masks(self):
        atomic_config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
            density_enabled=True,
            curvature_enabled=True,
            density_elements=("C",),
            dg_kernel_configs=((2.0, 1.0),),
        )
        config = GNNConfig(
            graph=GraphConfig(
                include_bond_features=False,
                include_distance=False,
                include_rbf=False,
            ),
            atomic_features=atomic_config,
        )
        names = get_density_feature_names(atomic_config)
        x_dg = np.zeros((1, len(names)))
        x_dg[:, names.index("curvature_valid_number_from_C_kernel_exponential_0")] = 2.0
        graph = SimpleNamespace(
            x_chem=np.empty((1, 0)),
            x_dg=x_dg,
            edge_attr=np.empty((1, 0)),
        )
        with self.assertRaisesRegex(ValueError, "binary values"):
            GraphFeatureNormalizer.from_config(config).fit([graph])

    def test_transform_graph_scales_selected_columns_without_mutating_input(self):
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
        graph = SimpleNamespace(
            x_chem=np.array([[1.0, 2.0, 3.0, 10.0], [3.0, 4.0, 5.0, 14.0]]),
            x_dg=np.empty((2, 0)),
            edge_attr=np.array([[1.0], [3.0]]),
            x=np.array([[1.0, 2.0, 3.0, 10.0], [3.0, 4.0, 5.0, 14.0]]),
        )
        normalizer = GraphFeatureNormalizer.from_config(config).fit([graph])

        transformed = normalizer.transform_graph(graph)

        np.testing.assert_allclose(
            transformed.x_chem,
            [[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]],
        )
        np.testing.assert_allclose(transformed.edge_attr, [[-1.0], [1.0]])
        np.testing.assert_allclose(transformed.x, transformed.x_chem)
        np.testing.assert_allclose(graph.x_chem, [[1.0, 2.0, 3.0, 10.0], [3.0, 4.0, 5.0, 14.0]])

    def test_transform_graph_preserves_invalid_curvature_neutral_value_and_mask(self):
        atomic_config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
            density_enabled=True,
            curvature_enabled=True,
            density_elements=("C",),
            dg_kernel_configs=((2.0, 1.0),),
        )
        config = GNNConfig(
            graph=GraphConfig(
                include_bond_features=False,
                include_distance=False,
                include_rbf=False,
            ),
            atomic_features=atomic_config,
        )
        names = get_density_feature_names(atomic_config)
        x_dg = np.zeros((2, len(names)))
        curvature_name = next(name for name in names if name.startswith("H_"))
        mask_name = f"curvature_valid_{curvature_name[2:]}"
        curvature_index = names.index(curvature_name)
        mask_index = names.index(mask_name)
        x_dg[:, curvature_index] = [0.0, 20.0]
        for name_index, name in enumerate(names):
            if name.startswith("curvature_valid_"):
                x_dg[:, name_index] = 1.0
        x_dg[:, mask_index] = [0.0, 1.0]
        graph = SimpleNamespace(
            x_chem=np.empty((2, 0)),
            x_dg=x_dg,
            edge_attr=np.empty((2, 0)),
            x=x_dg.copy(),
        )

        transformed = GraphFeatureNormalizer.from_config(config).fit([graph]).transform_graph(graph)

        self.assertEqual(transformed.x_dg[0, curvature_index], 0.0)
        self.assertEqual(transformed.x_dg[1, curvature_index], 0.0)
        np.testing.assert_array_equal(
            transformed.x_dg[:, mask_index], x_dg[:, mask_index]
        )

    def test_transform_requires_fitted_statistics(self):
        normalizer = GraphFeatureNormalizer.from_config(GNNConfig())
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            normalizer.transform_graph(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
