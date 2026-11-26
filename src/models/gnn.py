"""Atomic-number embedding and node encoder for the G7 molecular GNN.

The atom embedding belongs to T8.1 and the node encoder to T8.2. Message
passing, pooling, and readout are intentionally implemented in later tasks.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None

from src.gnn.config import GNNConfig


_MAX_ATOMIC_NUMBER = 118
_NUM_ATOMIC_EMBEDDINGS = _MAX_ATOMIC_NUMBER + 1
_EmbeddingBase = nn.Module if nn is not None else object
_NodeEncoderBase = nn.Module if nn is not None else object
_MessagePassingBase = nn.Module if nn is not None else object


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


class NodeFeatureEncoder(_NodeEncoderBase):
    """Combine atomic-number embeddings with the graph node feature tensor.

    This module only prepares per-node representations. It does not inspect
    edges or perform message passing; those responsibilities belong to T8.3+
    tasks.
    """

    def __init__(self, config: GNNConfig) -> None:
        if nn is None:
            raise ImportError(
                "NodeFeatureEncoder requires PyTorch; install the model dependencies."
            )
        if not isinstance(config, GNNConfig):
            raise TypeError("config must be a GNNConfig instance.")
        super().__init__()
        self.hidden_dim = config.model.hidden_dim
        self.num_node_features: int | None = None
        self.atom_embedding = AtomicNumberEmbedding(self.hidden_dim)
        # The model layer must not depend on graph_builder's feature schema.
        # LazyLinear binds to the actual graph tensor on first use instead.
        self.projection = nn.LazyLinear(self.hidden_dim)

    @classmethod
    def from_config(cls, config: GNNConfig) -> "NodeFeatureEncoder":
        """Build the node encoder from a validated GNN configuration."""
        return cls(config)

    def forward(self, z, x):
        """Return one encoded vector per node from ``z`` and ``x``."""
        if torch is None:  # pragma: no cover - constructor raises first
            raise ImportError(
                "NodeFeatureEncoder requires PyTorch; install the model dependencies."
            )
        if not isinstance(z, torch.Tensor) or not isinstance(x, torch.Tensor):
            raise TypeError("z and x must be torch.Tensor instances.")
        if z.ndim != 1:
            raise ValueError(f"z must have shape [N]; got {tuple(z.shape)}.")
        if x.ndim != 2:
            raise ValueError(f"x must have shape [N, F]; got {tuple(x.shape)}.")
        if x.shape[0] != z.shape[0]:
            raise ValueError("z and x must contain the same number of nodes.")
        if self.num_node_features is None:
            self.num_node_features = int(x.shape[1])
        elif x.shape[1] != self.num_node_features:
            raise ValueError(
                f"x must have {self.num_node_features} feature columns; "
                f"got {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype.")
        atom_vectors = self.atom_embedding(z)
        return self.projection(torch.cat((atom_vectors, x), dim=1))


# Short alias for callers that use the architectural term directly.
NodeEncoder = NodeFeatureEncoder


class GINEMessagePassing(_MessagePassingBase):
    """One edge-aware GINE message-passing layer.

    The layer deliberately represents a single convolution only. Layer
    stacking, normalization, dropout, and residual connections are assigned
    to T8.4 and T8.5.
    """

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        if nn is None:
            raise ImportError(
                "GINEMessagePassing requires PyTorch; install the model dependencies."
            )
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if isinstance(edge_dim, bool) or not isinstance(edge_dim, int) or edge_dim <= 0:
            raise ValueError("edge_dim must be a positive integer.")
        try:
            from torch_geometric.nn import GINEConv
        except ModuleNotFoundError as exc:
            raise ImportError(
                "GINEMessagePassing requires torch-geometric; install the model dependencies."
            ) from exc
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.convolution = GINEConv(
            nn=nn.Linear(hidden_dim, hidden_dim),
            edge_dim=edge_dim,
        )

    def forward(self, x, edge_index, edge_attr):
        """Propagate node states using the supplied connectivity and edges."""
        if torch is None:  # pragma: no cover - constructor raises first
            raise ImportError(
                "GINEMessagePassing requires PyTorch; install the model dependencies."
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_attr, torch.Tensor):
            raise TypeError("edge_index and edge_attr must be torch.Tensor instances.")
        if x.ndim != 2 or x.shape[1] != self.hidden_dim:
            raise ValueError(
                f"x must have shape [N, {self.hidden_dim}]; got {tuple(x.shape)}."
            )
        if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError(
                f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}."
            )
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long.")
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr must have one row per edge.")
        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"edge_attr must have {self.edge_dim} columns; "
                f"got {edge_attr.shape[1]}."
            )
        if not x.is_floating_point() or not edge_attr.is_floating_point():
            raise TypeError("x and edge_attr must use floating-point dtypes.")
        return self.convolution(x, edge_index, edge_attr)


# Descriptive alias for callers that use the convolution terminology.
GINEConvLayer = GINEMessagePassing

