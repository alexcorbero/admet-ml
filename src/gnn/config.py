"""Configuration objects for the G7 molecular graph pipeline.

This module defines lightweight, validated configuration objects for graph
connectivity, chemical atom features, density, differential geometry,
model architecture, and training. GNNConfig composes these objects and
provides JSON serialization. It intentionally depends only on the Python
standard library and does not import RDKit, PyTorch, or PyTorch Geometric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any


_EDGE_MODES = {"covalent", "radius", "hybrid"}
_DG_KERNELS = {"exponential", "lorentz"}
_DENSITY_TYPES = {"number", "charge"}
_DENSITY_TRANSFORMS = {"identity", "signed_log1p"}

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
        if isinstance(self.cutoff, bool):
            raise ValueError("cutoff must be a finite numeric value greater than zero.")
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
    density_types: tuple[str, ...] = ("number",)
    density_elements: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_DENSITY_ELEMENTS
    )
    other_label: str = "Other"
    curvature_neutral_value: float = 0.0
    density_transform: str = "identity"

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

        if not isinstance(self.density_transform, str):
            raise ValueError("density_transform must be a string.")
        density_transform = self.density_transform.lower()
        if density_transform not in _DENSITY_TRANSFORMS:
            raise ValueError(
                "density_transform must be one of "
                f"{sorted(_DENSITY_TRANSFORMS)}; got {self.density_transform!r}."
            )

        hybridization_types = _as_string_tuple(
            self.hybridization_types,
            "hybridization_types",
        )
        chirality_types = _as_string_tuple(self.chirality_types, "chirality_types")
        cip_types = _as_string_tuple(self.cip_types, "cip_types")
        density_types = _as_string_tuple(self.density_types, "density_types")
        density_elements = _as_string_tuple(self.density_elements, "density_elements")

        allowed_chirality_types = set(_DEFAULT_CHIRALITY_TYPES)
        if any(value not in allowed_chirality_types for value in chirality_types):
            raise ValueError(
                "chirality_types may contain only the supported RDKit chiral tags."
            )
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

        if isinstance(self.curvature_neutral_value, bool):
            raise ValueError("curvature_neutral_value must be a finite numeric value.")
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
        object.__setattr__(self, "density_transform", density_transform)


@dataclass(frozen=True)
class ModelConfig:
    """Validated architecture settings for the G7 molecular GNN.

    This config describes model dimensions and high-level architecture
    choices only. It does not construct layers or depend on a tensor/GNN
    framework; model implementation belongs to the model phase.
    """

    hidden_dim: int = 128
    num_message_passing_layers: int = 4
    dropout: float = 0.1
    residual_connections: bool = True
    pooling: str = "mean"
    readout_hidden_dim: int = 128

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_dim",
            "num_message_passing_layers",
            "readout_hidden_dim",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ValueError("dropout must be a finite number in [0, 1).")
        dropout = float(self.dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a finite number in [0, 1).")

        if not isinstance(self.residual_connections, bool):
            raise ValueError("residual_connections must be a boolean.")

        if not isinstance(self.pooling, str):
            raise ValueError("pooling must be a string.")
        pooling = self.pooling.lower()
        allowed_pooling = {"sum", "mean", "max", "mean_max"}
        if pooling not in allowed_pooling:
            raise ValueError(
                f"pooling must be one of {sorted(allowed_pooling)}; got {self.pooling!r}."
            )

        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "pooling", pooling)

@dataclass(frozen=True)
class TrainingConfig:
    """Validated hyperparameters for the G7 training procedure.

    The documented training protocol fixes AdamW as the optimizer and
    validation MAE as the checkpoint-selection metric. The run seed and
    device are supplied by the execution context.
    """

    batch_size: int = 32
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss: str = "huber"
    huber_delta: float = 1.0
    early_stopping_patience: int = 20
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

        numeric_fields = ("learning_rate", "weight_decay", "huber_delta")
        numeric_values: dict[str, float] = {}
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a finite numeric value.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{field_name} must be finite.")
            numeric_values[field_name] = numeric_value

        if numeric_values["learning_rate"] <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        if numeric_values["weight_decay"] < 0.0:
            raise ValueError("weight_decay must be greater than or equal to zero.")
        if numeric_values["huber_delta"] <= 0.0:
            raise ValueError("huber_delta must be greater than zero.")

        if not isinstance(self.loss, str):
            raise ValueError("loss must be a string.")
        loss = self.loss.strip().lower()
        allowed_losses = {"huber", "mae", "mse"}
        if loss not in allowed_losses:
            raise ValueError(
                f"loss must be one of {sorted(allowed_losses)}; got {self.loss!r}."
            )

        gradient_clip_norm = self.gradient_clip_norm
        if gradient_clip_norm is not None:
            if (
                isinstance(gradient_clip_norm, bool)
                or not isinstance(gradient_clip_norm, (int, float))
            ):
                raise ValueError(
                    "gradient_clip_norm must be None or a finite positive number."
                )
            gradient_clip_norm = float(gradient_clip_norm)
            if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
                raise ValueError(
                    "gradient_clip_norm must be None or a finite positive number."
                )

        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "gradient_clip_norm", gradient_clip_norm)
        for field_name, value in numeric_values.items():
            object.__setattr__(self, field_name, value)


def _json_compatible(value: Any) -> Any:
    """Convert dataclass-derived values into JSON-native containers."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


@dataclass(frozen=True)
class GNNConfig:
    """Complete, validated composition of the four G7 configuration objects.

    This object contains configuration only. Run metadata, normalizer
    statistics, checkpoints, and training history remain separate artifacts.
    """

    graph: GraphConfig = field(default_factory=GraphConfig)
    atomic_features: AtomicFeaturesConfig = field(default_factory=AtomicFeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        expected_types = (
            ("graph", GraphConfig),
            ("atomic_features", AtomicFeaturesConfig),
            ("model", ModelConfig),
            ("training", TrainingConfig),
        )
        for field_name, expected_type in expected_types:
            if not isinstance(getattr(self, field_name), expected_type):
                raise ValueError(
                    f"{field_name} must be an instance of {expected_type.__name__}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete configuration using JSON-native containers."""
        return _json_compatible(asdict(self))

    @property
    def graph_config_id(self) -> str:
        """Return the reproducible ID for the graph-construction configuration."""
        return get_graph_config_id(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GNNConfig:
        """Build a validated configuration from a complete mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("GNN configuration must be a mapping.")

        config_types = {
            "graph": GraphConfig,
            "atomic_features": AtomicFeaturesConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
        }
        supplied_sections = set(data)
        expected_sections = set(config_types)
        if supplied_sections != expected_sections:
            missing = sorted(str(key) for key in expected_sections - supplied_sections)
            unexpected = sorted(str(key) for key in supplied_sections - expected_sections)
            raise ValueError(
                "GNN configuration sections do not match the schema; "
                f"missing={missing}, unexpected={unexpected}."
            )

        nested_configs: dict[str, Any] = {}
        for section_name, config_type in config_types.items():
            section_data = data[section_name]
            if not isinstance(section_data, Mapping):
                raise ValueError(f"{section_name} must be a mapping.")

            expected_fields = {item.name for item in fields(config_type)}
            supplied_fields = set(section_data)
            if supplied_fields != expected_fields:
                missing = sorted(str(key) for key in expected_fields - supplied_fields)
                unexpected = sorted(str(key) for key in supplied_fields - expected_fields)
                raise ValueError(
                    f"{section_name} fields do not match the schema; "
                    f"missing={missing}, unexpected={unexpected}."
                )
            try:
                nested_configs[section_name] = config_type(**dict(section_data))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {section_name} configuration: {exc}"
                ) from exc

        return cls(**nested_configs)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete configuration as deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, content: str) -> GNNConfig:
        """Deserialize and validate a JSON configuration string."""
        if not isinstance(content, str):
            raise ValueError("JSON configuration content must be a string.")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON configuration.") from exc
        return cls.from_dict(data)

    def save_json(self, path: str | Path) -> Path:
        """Write the complete configuration to a UTF-8 JSON file."""
        target = Path(path)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> GNNConfig:
        """Load and validate a configuration from a UTF-8 JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def get_graph_config_id(config: GNNConfig) -> str:
    """Return a stable SHA-256 ID for the raw graph configuration.

    Only settings that affect raw graph construction are included: graph
    connectivity/edge features and atomic/density features. Model and
    training settings deliberately do not affect this cache identifier.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    config_dict = config.to_dict()
    payload = {
        "graph": config_dict["graph"],
        "atomic_features": config_dict["atomic_features"],
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

