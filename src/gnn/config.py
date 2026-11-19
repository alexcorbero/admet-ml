from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Literal


EdgeMode = Literal["hybrid", "covalent", "radius"]
KernelName = Literal["exponential", "lorentz"]
PoolingName = Literal["mean", "sum"]


@dataclass(frozen=True)
class GraphConfig:
    """How to construct one PyG graph from an RDKit 3D conformer.

    hybrid: union of RDKit bonds and spatial neighbors within cutoff.
    covalent: only RDKit chemical bonds.
    radius: only spatial neighbors within cutoff.

    Edges are bidirectional; duplicates are removed. No self loops.
    A covalent bond is kept in 'hybrid' even if its length exceeds cutoff.
    """

    edge_mode: EdgeMode = "hybrid"
    radius_cutoff_angstrom: float = 5.0
    max_neighbors: int | None = None  # None: do not truncate neighbors
    rbf_num_centers: int = 32
    include_bond_attributes: bool = True
    include_distances: bool = True


@dataclass(frozen=True)
class AtomicFeaturesConfig:
    """Atom attributes and local density/differential-geometry inputs.

    G6 inputs are atom-level values *before* molecular aggregation.
    eta_ij = tau * (vdW_radius_i + vdW_radius_j).

    The first exponential G6 configuration (order=3.5, tau=0.3)
    was not informative with our atom-centered vdW exclusion; therefore
    the default below uses the previously usable (2.5, 1.3) setting.

    Keep the same explicit-H policy for all saved conformers; never
    add/remove H after graph coordinates and atom features are aligned.
    """

    include_basic_chemistry: bool = True
    include_density: bool = True
    include_curvature: bool = True

    elements: tuple[str, ...] = (
        "H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Na"
    )
    other_label: str = "Other"
    density_types: tuple[str, ...] = ("number",)
    kernel: KernelName = "exponential"
    kernel_configs: tuple[tuple[float, float], ...] = ((2.5, 1.3),)
    sigma_angstrom: float = 0.0
    gradient_epsilon: float = 1e-8
    relative_gradient_epsilon: float = 1e-6

    # Keep a mask for undefined curvature; its numeric value will be
    # replaced by 0 AFTER the mask is computed, never silently discarded.
    include_validity_masks: bool = True
    geometry_transform: Literal["signed_log1p", "none"] = "signed_log1p"
    normalization: Literal["train_standard", "none"] = "train_standard"

@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the edge-aware message-passing encoder and readout."""

    hidden_dim: int = 128
    num_message_passing_layers: int = 4
    message_mlp_hidden_dim: int = 128
    dropout: float = 0.15
    residual_connections: bool = True
    pooling: PoolingName = "mean"
    readout_hidden_dim: int = 128

@dataclass(frozen=True)
class TrainingConfig:
    """Training defaults; scaffold splits come from the existing pipeline."""

    batch_size: int = 32
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: Literal["mae", "mse", "huber"] = "huber"
    huber_delta: float = 1.0
    early_stopping_patience: int = 25
    gradient_clip_norm: float = 5.0
    random_seed: int = 42
    num_workers: int = 0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    checkpoint_dir: str = "models/gnn"

@dataclass(frozen=True)
class GNNConfig:
    """Single, serializable configuration for G7 experiments."""

    graph: GraphConfig = field(default_factory=GraphConfig)
    atomic_features: AtomicFeaturesConfig = field(default_factory=AtomicFeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    graph_cache_dir: str = "data/graphs/aqsoldb"

    def to_dict(self) -> dict:
        """Return a plain dictionary suitable for JSON/YAML experiment logs."""
        return asdict(self)
