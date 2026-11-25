"""Configuration objects for the G7 molecular graph pipeline.

This module defines lightweight configuration for graph connectivity,
chemical atom features, density, differential geometry, and (in later tasks)
model and training settings. It intentionally depends only on the Python
standard library and does not import RDKit, PyTorch, or PyTorch Geometric.
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
_DEFAULT_CIP_TYPES = ("R", "S", "UNSPECIFIED")

# Initial G6 vocabulary. "Other" is deliberately kept outside this tuple
# and is configured separately by AtomicFeaturesConfig.other_label.
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
)

# Element symbols are kept locally so configuration validation does not
# introduce a dependency on RDKit's periodic table.
_CHEMICAL_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
    "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
    "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk "
    "Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)

_DG_KERNEL_CONFIGS_UNSET = object()

# Initial G6 starting points only; these are not optimized for AqSolDB.
# Each pair is (order, tau), with eta_ij = tau * (r_i(vdW) + r_j(vdW)).
_DEFAULT_DG_KERNEL_CONFIGS = {
    "exponential": (
        (3.5, 0.3),
        (2.5, 1.3),
    ),
    "lorentz": (
        (3.0, 1.3),
        (6.5, 0.3),
    ),
}


def _as_string_tuple(
    values: tuple[str, ...] | list[str],
    name: str,
) -> tuple[str, ...]:
    """Validate and normalize a non-empty collection of unique strings."""
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list of strings.")
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain only non-empty strings.")

    result = tuple(value.strip() for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _as_dg_kernel_configs(
    values: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Validate each DG (order, tau) pair independently."""
    if not isinstance(values, (tuple, list)):
        raise ValueError("dg_kernel_configs must be a tuple or list of (order, tau) pairs.")

    configs: list[tuple[float, float]] = []
    for index, pair in enumerate(values):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"dg_kernel_configs[{index}] must contain exactly (order, tau)."
            )
        order_value, tau_value = pair
        if isinstance(order_value, bool) or isinstance(tau_value, bool):
            raise ValueError("DG order and tau values must be numeric, not boolean.")
        try:
            order = float(order_value)
            tau = float(tau_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("DG order and tau values must be numeric.") from exc

        if not math.isfinite(order) or not math.isfinite(tau):
            raise ValueError("Each DG order and tau value must be finite.")
        if order < 2.0:
            raise ValueError("Every DG order must be greater than or equal to 2.")
        if tau <= 0.0:
            raise ValueError("Every DG tau value must be greater than zero.")

        configs.append((order, tau))

    result = tuple(configs)
    if len(set(result)) != len(result):
        raise ValueError("dg_kernel_configs must not contain duplicate (order, tau) pairs.")
    return result


@dataclass(frozen=True)
class GraphConfig:
    """Validated configuration for raw G7 graph connectivity and edge features."""

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
                raise ValueError(&"{field_name} must be a boolean.")

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
    """Configuration for local atom chemistry, density, and DG features.

    If ``dg_kernel_configs`` is omitted, the initial defaults associated
    with ``dg_kernel`` are selected. After construction the field is always
    a tuple of explicit ``(order, tau)`` pairs. Supplying a tuple explicitly
    overrides those defaults.
    """

    # Local chemical node features.
    include_formal_charge: bool = True
    include_heavy_atom_degree: bool = True
    include_attached_hydrogens: bool = True
     include_atomic_mass: bool = True
    include_aromaticity: bool = True
    include_ring_membership: bool = True
    include_hybridization: bool = True
    hybridization_types: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_HYBRIDIZATION_TYPES
    )
    include_chirality: bool = True
    chirality_types: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_CHIRALITY_TYPES
    )
    include_cip: bool = True
    cip_types: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_CIP_TYPES
    )

    # Optional local density / differential-geometry features.
    # G7-A: both disabled; G7-B: density only; G7-C: density + curvature.
    density_enabled: bool = False
    curvature_enabled: bool = False
    dg_kernel: str = "exponential"
    # Explicit (order, tau) pairs; tau scales the sum of the two vdW radii
    # to obtain eta_ij. Defaults are initial G6 values, not AqSolDB-tuned.
    dg_kernel_configs: tuple[tuple[float, float], ...] = field(
        default=_DG_KERNEL_CONFIGS_UNSET  # type: ignore[arg-type]
    )
    density_types: tuple[str, ...] = ("number")
    density_elements: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_DENSITY_ELEMENTS
    )
    other_label: str = "Other"
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
            "include_cip",
            "density_enabled",
            "curvature_enabled",
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
        chirality_types = _as_string_tuple(self.chirality_types, "chirality_types")
        cip_types = _as_string_tuple(self.cip_types, "cip_types")
        density_types = _as_string_tuple(self.density_types, "density_types")
        density_elements = _as_string_tuple(self.density_elements, "density_elements")

        allowed_cip_types = set(_DEFAULT_CIP_TYPES)
        if any(value not in allowed_cip_types for value in cip_types):
            raise ValueError("cip_types may contain only 'R', 'S', and 'UNSPECIFIED'.")
        if self.other_label is None or not isinstance(self.other_label, str) or not self.other_label.strip():
            raise ValueError("other_label must be a non-empty string.")
        other_label = self.other_label.strip()
        if other_label in density_elements:
            raise ValueError("other_label must not be included in density_elements.")
        unknown_elements = set(density_elements) - _CHEMICAL_ELEMENTS
        if unknown_elements:
            raise ValueError(
                "density_elements must contain only real chemical element symbols; "
                f"unknown values: {sorted(unknown_elements)}."
            )

        if any(value not in _DENSITY_TYPES for value in density_types):
            raise ValueError("density_types must contain only 'number' and/or 'charge'.")

        if self.dg_kernel_configs is _DG_KERNEL_CONFIGS_UNSET:
            dg_kernel_configs = _DEFAULT_DG_KERNEL_CONFIGS[dg_kernel]
        else:
            dg_kernel_configs = _as_dg_kernel_configs(self.dg_kernel_configs)

        if (self.density_enabled or self.curvature_enabled) and not dg_kernel_configs:
            raise ValueError(
                "dg_kernel_configs must contain at least one pair when density or curvature is enabled."
            )
        # Validate defaults and explicit configurations through the same path.
        dg_kernel_configs = _as_dg_kernel_configs(dg_kernel_configs)

        try:
            curvature_neutral_value = float(self.curvature_neutral_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("curvature_neutral_value must be finite.") from exc
        if not math.isfinite(curvature_neutral_value):
            raise ValueError("curvature_neutral_value must be finite.")

        object.__setattr__(self, "dg_kernel", dg_kernel)
        object.__setattr__(self, "dg_kernel_configs", dg_kernel_configs)
        object.__setattr__(self, "hybridization_types", hybridization_types)
        object.__setattr__(self, "chirality_types", chirality_types)
        object.__setattr__(self, "cip_types", cip_types)
        object.__setattr__(self, "density_types", density_types)
        object.__setattr__(self, "density_elements", density_elements)
        object.__setattr__(self, "other_label", other_label)
        object.__setattr__(self, "curvature_neutral_value", curvature_neutral_value)
