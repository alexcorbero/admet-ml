"""Configuration for G7 graph connectivity and edge attributes.

This module intentionally contains only the graph-level edge configuration.
It does not import RDKit, PyTorch, or PyTorch Geometric.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


_EDGE_MODES = {"covalent", "radius", "hybrid"}


@dataclass(frozen=True)
class GraphConfig:
    """Validated configuration for raw G7 graph connectivity.

    The configuration is deliberately limited to edge construction. Node
    feature, model, training, and serialization configuration belongs to
    later tasks.
    """

    edge_mode: str = "hybrid"
    cutoff: float = 5.0

    include_bond_features: bool = True
    include_distance: bool = True
    include_rbf: bool = True
    rbf_num_centers: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.edge_mode, str):
            raise ValueError("edge_mode must be a string.")
        edge_mode = self.edge_mode.lower()

        if edge_mode not in _EDGE_MODES:
            raise ValueError(
                f"edge_mode must be one of {sorted(_EDGE_MODES)}; "
                f"got {self.edge_mode!r}."
            )
        try:
            cutoff = float(self.cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError("cutoff must be finite and greater than zero.") from exc
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and greater than zero.")
        boolean_fields = (
            "include_bond_features",
            "include_distance",
            "include_rbf",
        )
        for field_name in boolean_fields:
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")

        if (
            isinstance(self.rbf_num_centers, bool)
            or not isinstance(self.rbf_num_centers, int)
            or self.rbf_num_centers < 2
        ):
            raise ValueError("rbf_num_centers must be an integer greater than or equal to 2.")

        object.__setattr__(self, "edge_mode", edge_mode)
        object.__setattr__(self, "cutoff", cutoff)
