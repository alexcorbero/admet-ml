"""Configuration for G7 graph connectivity and edge attributes.

This module intentionally contains only the graph-level edge configuration.
It does not import RDKit, PyTorch, or PyTorch Geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


_EDGE_MODES = {"covalent", "radius", "hybrid"}
_KERNELS = {"exponential", "lorentz"}


def _as_float_tuple(
    values: tuple[float, ...] | list[float],
    name: str,
) -> tuple[float, ...]:
    """Convert a sequence to finite floats and reject empty sequences."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if not result:
        raise ValueError(f"{name} must not be empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


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
    include_kernel: bool = True
    rbf_centers: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(value) for value in range(11))
    )
    rbf_gamma: float = 4.0
    kernel: str = "exponential"
    kernel_scales: tuple[float, ...] = (0.3, 1.3)

    def __post_init__(self) -> None:
        if not isinstance(self.edge_mode, str):
            raise ValueError("edge_mode must be a string.")
        if not isinstance(self.kernel, str):
            raise ValueError("kernel must be a string.")

        edge_mode = self.edge_mode.lower()
        kernel = self.kernel.lower()

        if edge_mode not in _EDGE_MODES:
            raise ValueError(
                f"edge_mode must be one of {sorted(_EDGE_MODES)}; "
                f"got {self.edge_mode!r}."
            )
        if not math.isfinite(float(self.cutoff)) or float(self.cutoff) <= 0:
            raise ValueError("cutoff must be finite and greater than zero.")
        if kernel not in _KERNELS:
            raise ValueError(
                f"kernel must be one of {sorted(_KERNELS)}; "
                f"got {self.kernel!r}."
            )

        boolean_fields = (
            "include_bond_features",
            "include_distance",
            "include_rbf",
            "include_kernel",
        )
        for field_name in boolean_fields:
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")

        rbf_centers = _as_float_tuple(self.rbf_centers, "rbf_centers")
        kernel_scales = _as_float_tuple(self.kernel_scales, "kernel_scales")

        if any(value < 0 for value in rbf_centers):
            raise ValueError("rbf_centers must contain non-negative distances.")
        if any(value <= 0 for value in kernel_scales):
            raise ValueError("kernel_scales must contain values greater than zero.")
        if not math.isfinite(float(self.rbf_gamma)) or float(self.rbf_gamma) <= 0:
            raise ValueError("rbf_gamma must be finite and greater than zero.")

        object.__setattr__(self, "edge_mode", edge_mode)
        object.__setattr__(self, "cutoff", float(self.cutoff))
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "rbf_centers", rbf_centers)
        object.__setattr__(self, "kernel_scales", kernel_scales)
        object.__setattr__(self, "rbf_gamma", float(self.rbf_gamma))
