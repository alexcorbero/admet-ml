"""Configuration for G7 graph construction.

This module intentionally contains only the graph-level configuration.  It
does not import RDKit, PyTorch, or PyTorch Geometric so that configurations
can be created and validated before the scientific/runtime dependencies are
loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping


_EDGE_MODES = {"covalent", "radius", "hybrid"}
_KERNELS = {"exponential", "lorentz"}
_HYDROGEN_POLICIES = {"explicit"}
_CONFORMER_POLICIES = {"lowest_energy"}
_DENSITY_TYPES = {"number", "charge"}


def _as_float_tuple(values: tuple[float, ...] | list[float], name: str) -> tuple[float, ...]:
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


def _as_string_tuple(values: tuple[str, ...] | list[str], name: str) -> tuple[str, ...]:
    """Convert a sequence to non-empty strings and reject duplicates."""
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if any(not value for value in result):
        raise ValueError(f"{name} must not contain empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


@dataclass(frozen=True)
class GraphConfig:
    """Validated, serializable configuration for raw G7 graph generation.

    The defaults describe the documented baseline: a hybrid graph with
    invariant distance features, exponential edge kernels, explicit
    hydrogens, one lowest-energy conformer, and optional local DG features.
    Node- and model-level settings belong to the later configuration classes.
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

    density_enabled: bool = False
    curvature_enabled: bool = False
    density_kernel: str = "exponential"
    density_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    density_types: tuple[str, ...] = ("number",)

    hydrogen_policy: str = "explicit"
    conformer_policy: str = "lowest_energy"
    use_pos_as_input: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.edge_mode, str):
            raise ValueError("edge_mode must be a string.")
        if not isinstance(self.kernel, str):
            raise ValueError("kernel must be a string.")
        if not isinstance(self.density_kernel, str):
            raise ValueError("density_kernel must be a string.")
        if not isinstance(self.hydrogen_policy, str):
            raise ValueError("hydrogen_policy must be a string.")
        if not isinstance(self.conformer_policy, str):
            raise ValueError("conformer_policy must be a string.")

        edge_mode = self.edge_mode.lower()
        kernel = self.kernel.lower()
        density_kernel = self.density_kernel.lower()

        if edge_mode not in _EDGE_MODES:
            raise ValueError(
                f"edge_mode must be one of {sorted(_EDGE_MODES)}; got {self.edge_mode!r}."
            )
        if not math.isfinite(float(self.cutoff)) or float(self.cutoff) <= 0:
            raise ValueError("cutoff must be finite and greater than zero.")
        if kernel not in _KERNELS:
            raise ValueError(
                f"kernel must be one of {sorted(_KERNELS)}; got {self.kernel!r}."
            )
        if density_kernel not in _KERNELS:
            raise ValueError(
                "density_kernel must be one of "
                f"{sorted(_KERNELS)}; got {self.density_kernel!r}."
            )
        if self.hydrogen_policy not in _HYDROGEN_POLICIES:
            raise ValueError("hydrogen_policy must be 'explicit'.")
        if self.conformer_policy not in _CONFORMER_POLICIES:
            raise ValueError("conformer_policy must be 'lowest_energy'.")
        if self.use_pos_as_input:
            raise ValueError("use_pos_as_input must remain False for the GINE baseline.")

        rbf_centers = _as_float_tuple(self.rbf_centers, "rbf_centers")
        kernel_scales = _as_float_tuple(self.kernel_scales, "kernel_scales")
        density_scales = _as_float_tuple(self.density_scales, "density_scales")
        density_types = _as_string_tuple(self.density_types, "density_types")

        if any(value <= 0 for value in kernel_scales):
            raise ValueError("kernel_scales must contain values greater than zero.")
        if any(value <= 0 for value in density_scales):
            raise ValueError("density_scales must contain values greater than zero.")
        if any(value < 0 for value in rbf_centers):
            raise ValueError("rbf_centers must contain non-negative distances.")
        if not math.isfinite(float(self.rbf_gamma)) or float(self.rbf_gamma) <= 0:
            raise ValueError("rbf_gamma must be finite and greater than zero.")
        if any(value not in _DENSITY_TYPES for value in density_types):
            raise ValueError(
                "density_types must contain only 'number' and/or 'charge'."
            )
        if self.curvature_enabled and not self.density_enabled:
            raise ValueError("curvature_enabled requires density_enabled=True.")

        object.__setattr__(self, "edge_mode", edge_mode)
        object.__setattr__(self, "cutoff", float(self.cutoff))
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "density_kernel", density_kernel)
        object.__setattr__(self, "rbf_centers", rbf_centers)
        object.__setattr__(self, "kernel_scales", kernel_scales)
        object.__setattr__(self, "density_scales", density_scales)
        object.__setattr__(self, "density_types", density_types)
        object.__setattr__(self, "rbf_gamma", float(self.rbf_gamma))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this configuration."""
        return asdict(self)

    def to_json(self, *, sort_keys: bool = True) -> str:
        """Serialize this configuration as JSON."""
        return json.dumps(self.to_dict(), sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GraphConfig":
        """Build a configuration from a mapping, validating all fields."""
        return cls(**dict(values))

    @classmethod
    def from_json(cls, value: str) -> "GraphConfig":
        """Build a configuration from JSON, validating all fields."""
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("GraphConfig JSON must encode an object.")
        return cls.from_dict(payload)
