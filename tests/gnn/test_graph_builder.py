"""Tests for the T5.2-T5.4 raw graph composition contract."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.gnn.config import GNNConfig
from src.gnn.graph_builder import _validate_graph_data, build_graph


class _FakeTensor:
    def __init__(self, values, dtype=None):
        self.values = values
        self.dtype = dtype if dtype is not None else values.dtype

    @property
    def shape(self):
        return self.values.shape

    def long(self):
        self.dtype = "long"
        return self

    def float(self):
        self.dtype = "float32"
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
        atom_module.get_atom_feature_names = Mock(
            return_value=("chem_0", "chem_1")
        )
        density_module.get_density_feature_names = Mock(
            return_value=("dg_0",)
        )
        edge_module.get_edge_feature_names = Mock(
            return_value=("edge_0", "edge_1")
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
            np.asarray(values), dtype=dtype
        )
        torch_module.float32 = "float32"
        torch_module.long = "long"
        torch_module.isfinite = lambda tensor: np.isfinite(tensor.values)
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

    def test_target_is_required(self):
        with self.assertRaisesRegex(TypeError, "y is required"):
            build_graph(None, conf_id=0, config=GNNConfig(), y=None)

    def test_target_argument_cannot_be_omitted(self):
        with self.assertRaises(TypeError):
            build_graph(None, conf_id=0, config=GNNConfig())

    def _valid_data(self):
        return _FakeData(
            num_nodes=2,
            z=_FakeTensor(np.asarray([6, 8]), dtype="long"),
            x=_FakeTensor(np.ones((2, 3), dtype=np.float32), dtype="float32"),
            x_chem=_FakeTensor(
                np.ones((2, 2), dtype=np.float32), dtype="float32"
            ),
            x_dg=_FakeTensor(
                np.ones((2, 1), dtype=np.float32), dtype="float32"
            ),
            pos=_FakeTensor(
                np.ones((2, 3), dtype=np.float32), dtype="float32"
            ),
            edge_index=_FakeTensor(
                np.asarray([[0, 1], [1, 0]]), dtype="long"
            ),
            edge_attr=_FakeTensor(
                np.ones((2, 2), dtype=np.float32), dtype="float32"
            ),
            conf_id=_FakeTensor(np.asarray([7]), dtype="long"),
        )

    def _validation_modules(self):
        atom_module = types.ModuleType("src.gnn.atom_features")
        atom_module.get_atom_feature_names = Mock(
            return_value=("chem_0", "chem_1")
        )
        density_module = types.ModuleType("src.gnn.density_features")
        density_module.get_density_feature_names = Mock(
            return_value=("dg_0",)
        )
        edge_module = types.ModuleType("src.gnn.edge_features")
        edge_module.get_edge_feature_names = Mock(
            return_value=("edge_0", "edge_1")
        )
        return {
            "src.gnn.atom_features": atom_module,
            "src.gnn.density_features": density_module,
            "src.gnn.edge_features": edge_module,
        }

    def _fake_torch_for_validation(self):
        return types.SimpleNamespace(
            long="long",
            float32="float32",
            isfinite=lambda tensor: np.isfinite(tensor.values),
        )

    def test_rejects_wrong_graph_shape(self):
        data = self._valid_data()
        data.x = _FakeTensor(
            np.ones((2, 2), dtype=np.float32), dtype="float32"
        )
        with patch.dict(sys.modules, self._validation_modules()):
            with self.assertRaisesRegex(ValueError, "x must have shape"):
                _validate_graph_data(
                    data,
                    GNNConfig(),
                    self._fake_torch_for_validation(),
                )

    def test_rejects_wrong_graph_dtype(self):
        data = self._valid_data()
        data.x = _FakeTensor(
            np.ones((2, 3), dtype=np.float64), dtype="float64"
        )
        with patch.dict(sys.modules, self._validation_modules()):
            with self.assertRaisesRegex(ValueError, "x must have dtype"):
                _validate_graph_data(
                    data,
                    GNNConfig(),
                    self._fake_torch_for_validation(),
                )

    def test_rejects_non_finite_graph_values(self):
        data = self._valid_data()
        data.pos = _FakeTensor(
            np.asarray(
                [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
                dtype=np.float32,
            ),
            dtype="float32",
        )
        with patch.dict(sys.modules, self._validation_modules()):
            with self.assertRaisesRegex(ValueError, "pos contains"):
                _validate_graph_data(
                    data,
                    GNNConfig(),
                    self._fake_torch_for_validation(),
                )


if __name__ == "__main__":
    unittest.main()
