"""Tests for the T8.1 atomic-number embedding."""

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

from src.gnn.config import GNNConfig, ModelConfig
from src.gnn.graph_builder import get_node_feature_names
from src.models.gnn import AtomicNumberEmbedding, NodeFeatureEncoder


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
        x = torch.zeros((3, len(get_node_feature_names(config))), dtype=torch.float32)

        result = encoder(torch.tensor([6, 8, 17], dtype=torch.long), x)

        self.assertEqual(tuple(result.shape), (3, 8))
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertEqual(encoder.projection.in_features, 8 + x.shape[1])

    def test_rejects_node_feature_shape_mismatch(self):
        config = GNNConfig(model=ModelConfig(hidden_dim=8))
        encoder = NodeFeatureEncoder(config)
        with self.assertRaisesRegex(ValueError, "feature columns"):
            encoder(
                torch.tensor([6], dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
