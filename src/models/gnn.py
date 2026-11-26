"""Node encoders and edge-aware message passing for the G7 molecular GNN."""

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
_MessagePassingStackBase = nn.Module if nn is not None else object
_PoolingBase = nn.Module if nn is not None else object
_ReadoutBase = nn.Module if nn is not None else object
_MolecularGNNBase = nn.Module if nn is not None else object


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


class GINEMessagePassingStack(_MessagePassingStackBase):
    """Stack GINE layers with normalization, activation, and dropout.

    This is deliberately only the sequential message-passing trunk. Residual
    connections, graph pooling, and readout are separate model concerns and
    are implemented by later tasks.
    """

    def __init__(
        self,
        config: GNNConfig,
        edge_dim: int,
        *,
        normalization: str = "layer",
        activation: str = "relu",
    ) -> None:
        if nn is None:
            raise ImportError(
                "GINEMessagePassingStack requires PyTorch; install the model dependencies."
            )
        if not isinstance(config, GNNConfig):
            raise TypeError("config must be a GNNConfig instance.")
        if isinstance(edge_dim, bool) or not isinstance(edge_dim, int) or edge_dim <= 0:
            raise ValueError("edge_dim must be a positive integer.")
        if not isinstance(normalization, str) or normalization.lower() != "layer":
            raise ValueError("normalization must be 'layer'.")
        if not isinstance(activation, str) or activation.lower() != "relu":
            raise ValueError("activation must be 'relu'.")

        super().__init__()
        self.hidden_dim = config.model.hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = config.model.num_message_passing_layers
        self.residual_connections = config.model.residual_connections
        self.normalization = normalization.lower()
        self.activation = activation.lower()
        self.layers = nn.ModuleList(
            GINEMessagePassing(self.hidden_dim, edge_dim)
            for _ in range(self.num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        )
        self.activations = nn.ModuleList(
            nn.ReLU() for _ in range(self.num_layers)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(config.model.dropout) for _ in range(self.num_layers)
        )

    @classmethod
    def from_config(cls, config: GNNConfig, edge_dim: int) -> "GINEMessagePassingStack":
        """Build the configured message-passing stack."""
        return cls(config, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        """Apply all configured GINE layers to node states."""
        for layer, norm, activation, dropout in zip(
            self.layers,
            self.norms,
            self.activations,
            self.dropouts,
        ):
            residual = x
            x = dropout(activation(norm(layer(x, edge_index, edge_attr))))
            if self.residual_connections:
                x = x + residual
        return x


class GlobalPooling(_PoolingBase):
    """Aggregate node states into one representation per graph.

    ``batch`` assigns every node to a graph. The supported reductions mirror
    :class:`ModelConfig`: ``sum``, ``mean``, ``max``, and ``mean_max``.
    This module only creates graph embeddings; prediction readout belongs to
    T8.7.
    """

    _SUPPORTED = frozenset(("sum", "mean", "max", "mean_max"))

    def __init__(self, pooling: str = "mean") -> None:
        if nn is None:
            raise ImportError(
                "GlobalPooling requires PyTorch; install the model dependencies."
            )
        if not isinstance(pooling, str) or pooling.lower() not in self._SUPPORTED:
            raise ValueError(
                "pooling must be one of ['max', 'mean', 'mean_max', 'sum']."
            )
        super().__init__()
        self.pooling = pooling.lower()

    @classmethod
    def from_config(cls, config: GNNConfig) -> "GlobalPooling":
        """Build pooling from ``config.model.pooling``."""
        if not isinstance(config, GNNConfig):
            raise TypeError("config must be a GNNConfig instance.")
        return cls(config.model.pooling)

    def forward(self, x, batch):
        """Return one pooled row per graph in the batch."""
        if torch is None:  # pragma: no cover - constructor raises first
            raise ImportError(
                "GlobalPooling requires PyTorch; install the model dependencies."
            )
        if not isinstance(x, torch.Tensor) or not isinstance(batch, torch.Tensor):
            raise TypeError("x and batch must be torch.Tensor instances.")
        if x.ndim != 2:
            raise ValueError(f"x must have shape [N, H]; got {tuple(x.shape)}.")
        if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
            raise ValueError("batch must have shape [N] matching x.")
        if batch.dtype != torch.long:
            raise TypeError("batch must use torch.long.")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype.")
        if batch.numel() == 0:
            return x.new_empty((0, x.shape[1]))
        if bool(torch.any(batch < 0)):
            raise ValueError("batch graph indices must be non-negative.")

        num_graphs = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=num_graphs).to(dtype=x.dtype)
        pooled_sum = x.new_zeros((num_graphs, x.shape[1]))
        pooled_sum.index_add_(0, batch, x)
        if self.pooling == "sum":
            return pooled_sum

        if self.pooling in ("mean", "mean_max"):
            pooled_mean = pooled_sum / counts.clamp_min(1).unsqueeze(1)
            if self.pooling == "mean":
                return pooled_mean

        pooled_max = x.new_full((num_graphs, x.shape[1]), float("-inf"))
        for graph_index in range(num_graphs):
            graph_nodes = x[batch == graph_index]
            if graph_nodes.numel():
                pooled_max[graph_index] = graph_nodes.max(dim=0).values
        if self.pooling == "max":
            return pooled_max
        return torch.cat((pooled_mean, pooled_max), dim=1)


class ScalarReadout(_ReadoutBase):
    """Map a graph embedding to one scalar prediction per graph.

    The readout is an MLP for the regression target. It consumes already
    pooled graph embeddings; graph pooling itself remains the responsibility
    of :class:`GlobalPooling`.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        if nn is None:
            raise ImportError(
                "ScalarReadout requires PyTorch; install the model dependencies."
            )
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_config(
        cls,
        config: GNNConfig,
        input_dim: int | None = None,
    ) -> "ScalarReadout":
        """Build the scalar readout from model configuration."""
        if not isinstance(config, GNNConfig):
            raise TypeError("config must be a GNNConfig instance.")
        if input_dim is None:
            pooling_width = 2 if config.model.pooling == "mean_max" else 1
            input_dim = config.model.hidden_dim * pooling_width
        return cls(input_dim, config.model.readout_hidden_dim)

    def forward(self, graph_embeddings):
        """Return predictions with shape ``[num_graphs]``."""
        if torch is None:  # pragma: no cover - constructor raises first
            raise ImportError(
                "ScalarReadout requires PyTorch; install the model dependencies."
            )
        if not isinstance(graph_embeddings, torch.Tensor):
            raise TypeError("graph_embeddings must be a torch.Tensor.")
        if graph_embeddings.ndim != 2 or graph_embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"graph_embeddings must have shape [B, {self.input_dim}]; "
                f"got {tuple(graph_embeddings.shape)}."
            )
        if not graph_embeddings.is_floating_point():
            raise TypeError("graph_embeddings must use a floating-point dtype.")
        # GraphBuilder/PyG batching stores targets as [B]. Squeezing only the
        # final singleton output dimension keeps loss computation elementwise
        # and prevents accidental [B, B] broadcasting against ``batch.y``.
        return self.network(graph_embeddings).squeeze(-1)


class MolecularGNN(_MolecularGNNBase):
    """Compose node encoding, message passing, and pooling for ``encode``.

    The returned tensor is a graph-level embedding. Prediction through the
    readout and the public ``forward`` training path are intentionally left
    to their later tasks.
    """

    def __init__(self, config: GNNConfig, edge_dim: int) -> None:
        if nn is None:
            raise ImportError(
                "MolecularGNN requires PyTorch; install the model dependencies."
            )
        if not isinstance(config, GNNConfig):
            raise TypeError("config must be a GNNConfig instance.")
        super().__init__()
        self.config = config
        self.node_encoder = NodeFeatureEncoder(config)
        self.message_passing = GINEMessagePassingStack.from_config(config, edge_dim)
        self.pooling = GlobalPooling.from_config(config)
        self.readout = ScalarReadout.from_config(config)

    @classmethod
    def from_config(cls, config: GNNConfig, edge_dim: int) -> "MolecularGNN":
        """Build the model components from a validated configuration."""
        return cls(config, edge_dim)

    def encode(self, data):
        """Return one pooled embedding per graph in a PyG-like data object."""
        if torch is None:  # pragma: no cover - constructor raises first
            raise ImportError(
                "MolecularGNN requires PyTorch; install the model dependencies."
            )
        if data is None:
            raise TypeError("data must provide z, x, edge_index, and edge_attr.")
        required = ("z", "x", "edge_index", "edge_attr")
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(
                "data is missing required fields: " + ", ".join(missing) + "."
            )
        z, x = data.z, data.x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        node_states = self.node_encoder(z, x)
        node_states = self.message_passing(
            node_states,
            data.edge_index,
            data.edge_attr,
        )
        return self.pooling(node_states, batch)

    def forward(self, data):
        """Return one scalar prediction per graph and retain autograd flow."""
        return self.readout(self.encode(data))


# Descriptive alias for callers that use the convolution terminology.
GINEConvLayer = GINEMessagePassing
GINEStack = GINEMessagePassingStack
MessagePassingStack = GINEMessagePassingStack
GraphPooling = GlobalPooling
GraphReadout = ScalarReadout
Readout = ScalarReadout
GNNEncoder = MolecularGNN

