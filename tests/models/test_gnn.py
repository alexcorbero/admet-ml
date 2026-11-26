"""Tests for the T8.1 atomic-number embedding."""

import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

from src.gnn.config import GNNConfig, ModelConfig

try:
    from src.gnn.edge_features import get_edge_feature_names
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    get_edge_feature_names = None
from src.models.gnn import (
    AtomicNumberEmbedding,
    GINEMessagePassing,
    GINEMessagePassingStack,
    GlobalPooling,
    ScalarReadout,
    NodeFeatureEncoder,
)

try:
    import torch_geometric
except ModuleNotFoundError:
    torch_geometric = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AtomicNumberEmbeddingTests(unittest.TestCase):
    def test_maps_atomic_numbers_to_hidden_vectors(self):
        embedding = AtomicNumberEmbedding(hidden_dim=16)
        result = embedding(torch.tensor([1, 6, 8, 118], dtype=torch.long))

        self.assertEqual(tuple(result.shape), (4, 16))
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(embedding.embedding.num_embeddings, 119)
        self.assertTrue(torch.equal(embedding.embedding.weight[0], torch.zeros(16)))

    def test_from_config_uses_model_hidden_dimension(self):
        embedding = AtomicNumberEmbedding.from_config(
            GNNConfig(model=ModelConfig(hidden_dim=7))
        )

        self.assertEqual(embedding.hidden_dim, 7)
        self.assertEqual(
            tuple(embedding(torch.tensor([6], dtype=torch.long)).shape),
            (1, 7),
        )

    def test_rejects_invalid_atomic_number_inputs(self):
        embedding = AtomicNumberEmbedding(hidden_dim=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            embedding(torch.tensor([[6]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "between 1 and 118"):
            embedding(torch.tensor([0, 119], dtype=torch.long))


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class NodeFeatureEncoderTests(unittest.TestCase):
    def test_combines_z_embedding_and_x_into_hidden_vectors(self):
        config = GNNConfig(model=ModelConfig(hidden_dim=8))
        encoder = NodeFeatureEncoder(config)
        x = torch.zeros((3, 5), dtype=torch.float32)

        result = encoder(torch.tensor([6, 8, 17], dtype=torch.long), x)

        self.assertEqual(tuple(result.shape), (3, 8))
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(encoder.projection.in_features, 8 + x.shape[1])

    def test_rejects_node_feature_shape_mismatch(self):
        config = GNNConfig(model=ModelConfig(hidden_dim=8))
        encoder = NodeFeatureEncoder(config)
        encoder(
            torch.tensor([6], dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "feature columns"):
            encoder(
                torch.tensor([6], dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.float32),
            )


@unittest.skipUnless(
    torch is not None
    and torch_geometric is not None
    and get_edge_feature_names is not None,
    "PyTorch, PyTorch Geometric, and RDKit are not installed",
)
class GINEMessagePassingTests(unittest.TestCase):
    def test_uses_edge_index_and_edge_attr_to_return_node_states(self):
        config = GNNConfig()
        edge_dim = len(get_edge_feature_names(config.graph))
        layer = GINEMessagePassing(config.model.hidden_dim, edge_dim)
        x = torch.randn((3, config.model.hidden_dim), dtype=torch.float32)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.randn((4, edge_dim), dtype=torch.float32)

        result = layer(x, edge_index, edge_attr)

        self.assertEqual(tuple(result.shape), (3, config.model.hidden_dim))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_rejects_edge_attribute_shape_mismatch(self):
        layer = GINEMessagePassing(hidden_dim=8, edge_dim=3)
        with self.assertRaisesRegex(ValueError, "3 columns"):
            layer(
                torch.zeros((2, 8)),
                torch.tensor([[0], [1]], dtype=torch.long),
                torch.zeros((1, 2)),
            )


@unittest.skipUnless(
    torch is not None
    and torch_geometric is not None
    and get_edge_feature_names is not None,
    "PyTorch, PyTorch Geometric, and RDKit are not installed",
)
class GINEMessagePassingStackTests(unittest.TestCase):
    def test_applies_configured_layers_norm_activation_and_dropout(self):
        config = GNNConfig(
            model=ModelConfig(
                hidden_dim=8,
                num_message_passing_layers=2,
                dropout=0.0,
            )
        )
        edge_dim = len(get_edge_feature_names(config.graph))
        stack = GINEMessagePassingStack.from_config(config, edge_dim=edge_dim)
        x = torch.randn((3, 8), dtype=torch.float32)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.randn((4, edge_dim), dtype=torch.float32)

        result = stack(x, edge_index, edge_attr)

        self.assertEqual(len(stack.layers), 2)
        self.assertEqual(len(stack.norms), 2)
        self.assertEqual(len(stack.activations), 2)
        self.assertEqual(len(stack.dropouts), 2)
        self.assertEqual(tuple(result.shape), (3, 8))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_rejects_unsupported_stack_options(self):
        config = GNNConfig(model=ModelConfig(hidden_dim=8, num_message_passing_layers=1))
        with self.assertRaisesRegex(ValueError, "normalization"):
            GINEMessagePassingStack(config, edge_dim=3, normalization="batch")
        with self.assertRaisesRegex(ValueError, "activation"):
            GINEMessagePassingStack(config, edge_dim=3, activation="gelu")

    def test_applies_configured_residual_connection(self):
        config = GNNConfig(
            model=ModelConfig(
                hidden_dim=4,
                num_message_passing_layers=1,
                dropout=0.0,
                residual_connections=True,
            )
        )
        stack = GINEMessagePassingStack.from_config(
            config, edge_dim=len(get_edge_feature_names(config.graph))
        )
        x = torch.randn((2, 4), dtype=torch.float32)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_attr = torch.randn((1, stack.edge_dim), dtype=torch.float32)

        with patch.object(stack.layers[0], "forward", return_value=torch.zeros_like(x)):
            result = stack(x, edge_index, edge_attr)

        self.assertTrue(torch.allclose(result, x))


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class GlobalPoolingTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[1.0, 5.0], [3.0, 1.0], [10.0, 2.0]], dtype=torch.float32
        )
        self.batch = torch.tensor([0, 0, 1], dtype=torch.long)

    def test_supports_configured_reductions(self):
        expected = {
            "sum": torch.tensor([[4.0, 6.0], [10.0, 2.0]]),
            "mean": torch.tensor([[2.0, 3.0], [10.0, 2.0]]),
            "max": torch.tensor([[3.0, 5.0], [10.0, 2.0]]),
            "mean_max": torch.tensor(
                [[2.0, 3.0, 3.0, 5.0], [10.0, 2.0, 10.0, 2.0]]
            ),
        }
        for pooling, target in expected.items():
            with self.subTest(pooling=pooling):
                result = GlobalPooling(pooling)(self.x, self.batch)
                self.assertTrue(torch.allclose(result, target))

    def test_validates_batch_shape_and_dtype(self):
        pooling = GlobalPooling.from_config(GNNConfig())
        with self.assertRaisesRegex(ValueError, "matching x"):
            pooling(self.x, torch.tensor([0], dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "torch.long"):
            pooling(self.x, torch.tensor([0, 0, 1], dtype=torch.int32))


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class ScalarReadoutTests(unittest.TestCase):
    def test_returns_one_regression_value_per_graph(self):
        config = GNNConfig(
            model=ModelConfig(hidden_dim=6, readout_hidden_dim=4, pooling="mean_max")
        )
        readout = ScalarReadout.from_config(config)
        embeddings = torch.randn((3, 12), dtype=torch.float32)

        predictions = readout(embeddings)

        self.assertEqual(readout.input_dim, 12)
        self.assertEqual(readout.hidden_dim, 4)
        self.assertEqual(tuple(predictions.shape), (3, 1))
        self.assertTrue(torch.isfinite(predictions).all().item())

    def test_rejects_wrong_embedding_shape(self):
        readout = ScalarReadout(input_dim=5, hidden_dim=3)
        with self.assertRaisesRegex(ValueError, "shape"):
            readout(torch.zeros((2, 4), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
