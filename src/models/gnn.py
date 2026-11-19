"""
GINE-based molecular graph neural network for ADMET regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ..gnn.config import GNNConfig
from ..gnn.graph_builder import get_node_feature_names
from ..gnn.edge_features import get_edge_feature_names


class MolecularGNN(nn.Module):
    """
    Molecular GINE encoder with a graph-level regression head.

    The model combines:
        - an embedding of the atomic number (`data.z`),
        - chemical and G6 node features (`data.x`),
        - chemical and geometric edge features (`data.edge_attr`).

    `encode()` returns one learned embedding per molecule. `forward()`
    maps that embedding to a scalar prediction, e.g. aqueous solubility.
    """

    ATOMIC_NUMBER_VOCAB_SIZE = 119  # Atomic numbers 0..118.
    ATOMIC_EMBEDDING_DIM = 32

    def __init__(self, config: GNNConfig):
        super().__init__()

        model_config = config.model

        node_feature_dim = len(get_node_feature_names(config))
        edge_feature_dim = len(get_edge_feature_names(config))

        hidden_dim = model_config.hidden_dim
        message_hidden_dim = model_config.message_mlp_hidden_dim
        dropout = model_config.dropout

        # Learn an element embedding and combine it with explicit node features.
        self.atom_embedding = nn.Embedding(
            num_embeddings=self.ATOMIC_NUMBER_VOCAB_SIZE,
            embedding_dim=self.ATOMIC_EMBEDDING_DIM,
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(
                node_feature_dim + self.ATOMIC_EMBEDDING_DIM,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Each GINE layer projects edge_attr to hidden_dim internally.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(model_config.num_message_passing_layers):
            message_mlp = nn.Sequential(
                nn.Linear(hidden_dim, message_hidden_dim),
                nn.ReLU(),
                nn.Linear(message_hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINEConv(
                    nn=message_mlp,
                    train_eps=True,
                    edge_dim=edge_feature_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.residual_connections = model_config.residual_connections
        self.pooling = model_config.pooling

        if self.pooling == "mean_max":
            pooled_dim = 2 * hidden_dim
        else:
            pooled_dim = hidden_dim

        self.readout = nn.Sequential(
            nn.Linear(pooled_dim, model_config.readout_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_config.readout_hidden_dim, 1),
        )

    def encode(self, data) -> torch.Tensor:
        """Return a graph-level embedding for each molecule in the batch."""

        atom_embedding = self.atom_embedding(data.z)
        node_input = torch.cat((data.x, atom_embedding), dim=-1)
        x = self.node_encoder(node_input)

        for conv, norm in zip(self.convs, self.norms):
            updated = conv(
                x,
                data.edge_index,
                data.edge_attr,
            )
            updated = norm(updated)
            updated = F.relu(updated)
            updated = F.dropout(
                updated,
                p=self.dropout,
                training=self.training,
            )

            if self.residual_connections:
                x = x + updated
            else:
                x = updated

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        if self.pooling == "mean_max":
            graph_embedding = torch.cat(
                (
                    global_mean_pool(x, batch),
                    global_max_pool(x, batch),
                ),
                dim=-1,
            )
        elif self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            graph_embedding = global_add_pool(x, batch)
        elif self.pooling == "max":
            graph_embedding = global_max_pool(x, batch)

        return graph_embedding

    def forward(self, data) -> torch.Tensor:
        """Return one scalar prediction per molecule in the batch."""

        graph_embedding = self.encode(data)
        return self.readout(graph_embedding).squeeze(-1)
