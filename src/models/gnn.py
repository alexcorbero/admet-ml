"""Atomic-number embedding for the G7 molecular GNN.

Only the atom embedding belongs to T8.1. Node encoding, message passing,
pooling, and readout are intentionally implemented in later tasks.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


_MAX_ATOMIC_NUMBER = 118
_NUM_ATOMIC_EMBEDDINGS = _MAX_ATOMIC_NUMBER + 1
_EmbeddingBase = nn.Module if nn is not None else object


class AtomicNumberEmbedding(_EmbeddingBase):
    """Learned embedding for RDKit atomic numbers.

    Index zero is reserved, while real chemical atomic numbers occupy
    indices 1 through 118. The module maps ``z`` with shape ``[N]`` to a
    dense tensor with shape ``[N, hidden_dim]``.
    """

    max_atomic_number = _MAX_ATOMIC_NUMBER

    def __init__(self, hidden_dim: int) -> None:
        if nn is None:
            raise ImportError(
                "AtomicNumberEmbedding requires PyTorch; install the model dependencies."
            )
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be a positive integer.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(
            num_embeddings=_NUM_ATOMIC_EMBEDDINGS,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )

    @classmethod
    def from_config(cls, config) -> "AtomicNumberEmbedding":
        """Build the embedding using ``config.model.hidden_dim``."""
        hidden_dim = getattr(getattr(config, "model", None), "hidden_dim", None)
        if hidden_dim is None:
            raise TypeError("config must expose model.hidden_dim.")
        return cls(hidden_dim)

    def forward(self, z):
        """Embed a one-dimensional tensor of atomic numbers."""
        if torch is None:  # pragma: no cover - constructor raises first
            raise ImportError(
                "AtomicNumberEmbedding requires PyTorch; install the model dependencies."
            )
        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor.")
        if z.ndim != 1:
            raise ValueError(f"z must have shape [N]; got {tuple(z.shape)}.")
        if z.dtype not in (torch.int32, torch.int64):
            raise TypeError("z must use an integer dtype.")
        if z.numel() and (
            bool(torch.any(z < 1))
            or bool(torch.any(z > _MAX_ATOMIC_NUMBER))
        ):
            raise ValueError(
                f"z values must be between 1 and {_MAX_ATOMIC_NUMBER}."
            )
        return self.embedding(z)

