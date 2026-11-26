"""Tests for the T5.2-T5.3 raw graph composition contract."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.gnn.config import GNNConfig
from src.gnn.graph_builder import build_graph


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def long(self):
        return self

    def float(self):
        return self


class _FakeData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GraphBuilderTests(unittest.TestCase):
    def test_build_graph_composes_raw_feature_tensors(self):
        mol = Mock()
        mol.GetNumAtoms.return_value = 2
        config = GNNConfig()

        atom_module = types.ModuleType("src.gnn.atom_features")
        atom_module.build_atom_features = Mock(
            return_value=(
                np.asarray([6, 8], dtype=np.int64),
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
        )
        density_module = types.ModuleType("src.gnn.density_features")
        density_module.build_density_features = Mock(
            return_value=np.asarray([[5.0], [6.0]], dtype=np.float32)
        )
        edge_module = types.ModuleType("src.gnn.edge_features")
        edge_module.build_edge_features = Mock(
            return_value=(
                np.asarray([[0, 1], [1, 0]], dtype=np.int64),
                np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            )
        )
        coordinates_module = types.ModuleType("src.features.descriptors_3d")
        coordinates_module.get_coordinates = Mock(
            return_value=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        )

        torch_module = types.ModuleType("torch")
        torch_module.from_numpy = lambda values: _FakeTensor(values)
        torch_module.tensor = lambda values, dtype=None: _FakeTensor(
            np.asarray(values)
        )
        torch_module.float32 = object()
        torch_module.long = object()
        pyg_module = types.ModuleType("torch_geometric")
        pyg_data_module = types.ModuleType("torch_geometric.data")
        pyg_data_module.Data = _FakeData

        with patch.dict(
            sys.modules,
            {
                "src.gnn.atom_features": atom_module,
                "src.gnn.density_features": density_module,
                "src.gnn.edge_features": edge_module,
                "src.features.descriptors_3d": coordinates_module,
                "torch": torch_module,
                "torch_geometric": pyg_module,
                "torch_geometric.data": pyg_data_module,
            },
        ):
            data = build_graph(
                mol,
                conf_id=7,
                config=config,
                y=-1.25,
                smiles="CO",
                mol_id=42,
            )

        np.testing.assert_array_equal(data.z.values, [6, 8])
        np.testing.assert_array_equal(
            data.x.values,
            [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]],
        )
        np.testing.assert_array_equal(data.x_chem.values, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(data.x_dg.values, [[5], [6]])
        np.testing.assert_array_equal(
            data.pos.values,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(data.edge_index.values, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(data.edge_attr.values, [[1, 0], [1, 0]])
        self.assertEqual(data.num_nodes, 2)
        np.testing.assert_array_equal(data.y.values, [-1.25])
        self.assertEqual(data.smiles, "CO")
        self.assertEqual(data.mol_id, "42")
        np.testing.assert_array_equal(data.conf_id.values, [7])

        atom_module.build_atom_features.assert_called_once_with(
            mol, config.atomic_features
        )
        density_module.build_density_features.assert_called_once_with(
            mol, 7, config.atomic_features
        )
        edge_module.build_edge_features.assert_called_once_with(
            mol, 7, config.graph
        )
        coordinates_module.get_coordinates.assert_called_once_with(mol, 7)

    def test_invalid_config_type_is_rejected_before_dependency_imports(self):
        with self.assertRaises(TypeError):
            build_graph(None, conf_id=0, config=None)


if __name__ == "__main__":
    unittest.main()
