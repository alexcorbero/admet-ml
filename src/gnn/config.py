"""Configuration for G7 graph connectivity and edge attributes.

This module intentionally contains only the graph-level edge configuration.
It does not import RDKit, PyTorch, or PyTorch Geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


_EDGE_MODES = {"covalent", "radius", "hybrid"}
_DG_KERNELS = {"exponential", "lorentz"}
_DENSITY_TYPES = {"number", "charge"}

_DEFAULT_HYBRIDIZATION_TYPES = (
    "S",
    "SP",
    "SP2",
    "SP3",
    "SP3D",
    "SP3D2",
    "UNSPECIFIED",
)
_DEFAULT_CHIRALITY_TYPES = (
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
)
_DEFAULT_DENSITY_ELEMENTS = (
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "Na",
    "Other",
)


def _as_string_tuple(
    values: tuple[str, ...] | list[str],
    name: str,
) -> tuple[str, ...]:
    """Convert a sequence to unique, non-empty strings."""
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if any(not value for value in result):
        raise ValueError(f"{name} must not contain empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


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


@dataclass(frozen=True)
class AtomicFeaturesConfig:
    """Configuration for local chemical, density, and curvature features.

    This class defines feature schema and DG parameters only. Feature
    extraction remains the responsibility of later G7 feature modules.
    """

    # Local chemical node features from the G7 specification.
    include_formal_charge: bool = True
    include_heavy_atom_degree: bool = True
    include_attached_hydrogens: bool = True
    include_atomic_mass: bool = True
    include_aromaticity: bool = True
    include_ring_membership: bool = True
    include_hybridization: bool = True
    include_chirality: bool = True
    hybridization_types: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_HYBRIDIZATION_TYPES
    )
    chirality_types: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_CHIRALITY_TYPES
    )

    # Optional local density / differential-geometry features.  The default
    # corresponds to G7-A; enabling density gives G7-B and enabling curvature
    # as well gives G7-C without changing GraphConfig.
    density_enabled: bool = False
    curvature_enabled: bool = False
    dg_kernel: str = "exponential"
    dg_orders: tuple[float, ...] = (3.5, 2.5)
    dg_tau: tuple[float, ...] = (0.3, 1.3)
    density_types: tuple[str, ...] = ("number",)
    density_elements: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_DENSITY_ELEMENTS
    )
    other_label: str = "Other"
    include_curvature_mask: bool = True
    curvature_neutral_value: float = 0.0

    def __post_init__(self) -> None:
        boolean_fields = (
            "include_formal_charge",
            "include_heavy_atom_degree",
            "include_attached_hydrogens",
            "include_atomic_mass",
            "include_aromaticity",
            "include_ring_membership",
            "include_hybridization",
            "include_chirality",
            "density_enabled",
            "curvature_enabled",
            "include_curvature_mask",
        )
        for field_name in boolean_fields:
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean.")

        if not isinstance(self.dg_kernel, str):
            raise ValueError("dg_kernel must be a string.")
        dg_kernel = self.dg_kernel.lower()
        if dg_kernel not in _DG_KERNELS:
            raise ValueError(
                f"dg_kernel must be one of {sorted(_DG_KERNELS)}; "
                f"got {self.dg_kernel!r}."
            )
        if self.curvature_enabled and not self.density_enabled:
            raise ValueError("curvature_enabled requires density_enabled=True.")

        hybridization_types = _as_string_tuple(
            self.hybridization_types,
            "hybridization_types",
        )
        chirality_types = _as_string_tuple(
            self.chirality_types,
            "chirality_types",
        )
        density_types = _as_string_tuple(self.density_types, "density_types")
        density_elements = _as_string_tuple(
            self.density_elements,
            "density_elements",
        )

        if any(value not in _DENSITY_TYPES for value in density_types):
            raise ValueError("density_types must contain only 'number' and/or 'charge'.")
        if self.other_label not in density_elements:
            raise ValueError("density_elements must include other_label.")

        dg_orders = _as_float_tuple(self.dg_orders, "dg_orders")
        dg_tau = _as_float_tuple(self.dg_tau, "dg_tau")
        if len(dg_orders) != len(dg_tau):
            raise ValueError("dg_orders and dg_tau must have the same length.")
        if any(value < 2.0 for value in dg_orders):
            raise ValueError("dg_orders must be greater than or equal to 2.")
        if any(value <= 0 for value in dg_tau):
            raise ValueError("dg_tau must contain values greater than zero.")
        try:
            curvature_neutral_value = float(self.curvature_neutral_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("curvature_neutral_value must be finite.") from exc
        if not math.isfinite(curvature_neutral_value):
            raise ValueError("curvature_neutral_value must be finite.")

        object.__setattr__(self, "dg_kernel", dg_kernel)
        object.__setattr__(self, "hybridization_types", hybridization_types)
        object.__setattr__(self, "chirality_types", chirality_types)
        object.__setattr__(self, "density_types", density_types)
        object.__setattr__(self, "density_elements", density_elements)
        object.__setattr__(self, "dg_orders", dg_orders)
        object.__setattr__(self, "dg_tau", dg_tau)
        object.__setattr__(self, "curvature_neutral_value", curvature_neutral_value)
