"""Tests for the T8.1 atomic-number embedding."""

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

from src.gnn.config import GNNConfig, ModelConfig
from src.models.gnn import AtomicNumberEmbedding


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


if __name__ == "__main__":
    unittest.main()
