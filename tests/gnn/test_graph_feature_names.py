"""Tests for the T5.1 graph feature-name composition."""

import importlib.util
import unittest

from src.gnn.config import AtomicFeaturesConfig, GNNConfig, GraphConfig


RDKit_AVAILABLE = importlib.util.find_spec("rdkit") is not None

if RDKit_AVAILABLE:
    from src.gnn.atom_features import get_atom_feature_names
    from src.gnn.density_features import get_density_feature_names
    from src.gnn.edge_features import get_edge_feature_names
    from src.gnn.graph_builder import (
        get_graph_feature_names,
        get_node_feature_names,
    )


@unittest.skipUnless(RDKit_AVAILABLE, "RDKit is required by the feature modules")
class GraphFeatureNameTests(unittest.TestCase):
    def test_node_names_concatenate_chemical_and_dg_schemas(self):
        config = GNNConfig(
            atomic_features=AtomicFeaturesConfig(
                density_enabled=True,
                density_elements=("C",),
                dg_kernel_configs=((3.0, 0.5),),
            )
        )

        self.assertEqual(
            get_node_feature_names(config),
            (
                *get_atom_feature_names(config.atomic_features),
                *get_density_feature_names(config.atomic_features),
            ),
        )
        self.assertNotIn("z", get_node_feature_names(config))

    def test_graph_names_are_grouped_by_feature_tensor(self):
        config = GNNConfig(
            graph=GraphConfig(rbf_num_centers=3),
            atomic_features=AtomicFeaturesConfig(
                include_formal_charge=False,
                density_enabled=True,
                density_elements=("O",),
                dg_kernel_configs=((2.0, 0.8),),
            ),
        )

        names = get_graph_feature_names(config)

        self.assertEqual(
            names,
            {
                "x": get_node_feature_names(config),
                "x_chem": get_atom_feature_names(config.atomic_features),
                "x_dg": get_density_feature_names(config.atomic_features),
                "edge_attr": get_edge_feature_names(config.graph),
            },
        )

    def test_invalid_config_type_is_rejected(self):
        with self.assertRaises(TypeError):
            get_node_feature_names(None)
        with self.assertRaises(TypeError):
            get_graph_feature_names(None)


if __name__ == "__main__":
    unittest.main()
