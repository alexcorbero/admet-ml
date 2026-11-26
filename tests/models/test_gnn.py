"""Tests for the T8.1 atomic-number embedding."""

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

from src.gnn.config import GNNConfig, ModelConfig
from src.models.gnn import (
    AtomicNumberEmbedding,
    GINEMessagePassing,
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
    torch is not None and torch_geometric is not None,
    "PyTorch and PyTorch Geometric are not installed",
)
class GINEMessagePassingTests(unittest.TestCase):
    def test_uses_edge_index_and_edge_attr_to_return_node_states(self):
        config = GNNConfig()
        # Default GraphConfig: distance plus 32 Gaussian RBF features.
        edge_dim = 33
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


if __name__ == "__main__":
    unittest.main()
