"""Dataset-level graph generation for G7.

This module turns already prepared molecule records into raw PyG graphs and
persists/loads those raw graphs. Failure accounting, cache validation, and
subset selection remain separate responsibilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import GNNConfig
from .graph_builder import build_graph


@dataclass(frozen=True)
class GraphBuildSuccess:
    """Metadata for one record converted successfully into a graph."""

    record_index: int
    mol_id: Any
    smiles: Any
    graph: Any


@dataclass(frozen=True)
class GraphBuildFailure:
    """Explicit failure information for one record that was excluded."""

    record_index: int
    mol_id: Any
    smiles: Any
    error_type: str
    error_message: str


@dataclass(frozen=True)
class GraphBuildReport:
    """Graphs and per-record outcomes from a dataset generation run."""

    graphs: list[Any]
    successes: list[GraphBuildSuccess]
    failures: list[GraphBuildFailure]


@dataclass(frozen=True)
class GraphDatasetArtifact:
    """Persisted raw graphs together with caller-provided metadata."""

    graphs: list[Any]
    metadata: dict[str, Any]


def build_graph_dataset(
    records: Iterable[Mapping[str, Any]],
    config: GNNConfig,
    *,
    molecule_key: str = "mol",
    conf_id_key: str = "conf_id",
    target_key: str = "y",
    smiles_key: str = "smiles",
    mol_id_key: str = "mol_id",
) -> list[Any]:
    """Build raw graphs for an ordered iterable of molecule records.

    Each record must contain a molecule, its selected conformer ID, target,
    SMILES, and stable molecule ID. Records are converted in input order and
    each graph is built through :func:`src.gnn.graph_builder.build_graph`.

    Exceptions from graph construction intentionally propagate. Recording
    successes and failures is assigned to T6.3, while persistence and cache
    handling are assigned to later tasks.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    field_names = {
        "molecule": molecule_key,
        "conf_id": conf_id_key,
        "target": target_key,
        "smiles": smiles_key,
        "mol_id": mol_id_key,
    }
    if any(not isinstance(name, str) or not name for name in field_names.values()):
        raise ValueError("record field names must be non-empty strings.")

    graphs: list[Any] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"record {record_index} must be a mapping; "
                f"got {type(record).__name__}."
            )

        missing = [
            label
            for label, field_name in field_names.items()
            if field_name not in record
        ]
        if missing:
            raise ValueError(
                f"record {record_index} is missing required fields: "
                f"{', '.join(missing)}."
            )

        graphs.append(
            build_graph(
                record[molecule_key],
                conf_id=record[conf_id_key],
                config=config,
                y=record[target_key],
                smiles=record[smiles_key],
                mol_id=record[mol_id_key],
            )
        )

    return graphs


def build_graph_dataset_with_report(
    records: Iterable[Mapping[str, Any]],
    config: GNNConfig,
    *,
    molecule_key: str = "mol",
    conf_id_key: str = "conf_id",
    target_key: str = "y",
    smiles_key: str = "smiles",
    mol_id_key: str = "mol_id",
) -> GraphBuildReport:
    """Build graphs while recording successes and molecule-level failures.

    A malformed record or graph-construction exception excludes only that
    record and is captured in ``failures`` with its identifying metadata.
    Configuration and field-name errors remain fatal because they invalidate
    the whole generation run. No persistence or cache handling is performed.
    """
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    field_names = {
        "molecule": molecule_key,
        "conf_id": conf_id_key,
        "target": target_key,
        "smiles": smiles_key,
        "mol_id": mol_id_key,
    }
    if any(not isinstance(name, str) or not name for name in field_names.values()):
        raise ValueError("record field names must be non-empty strings.")

    graphs: list[Any] = []
    successes: list[GraphBuildSuccess] = []
    failures: list[GraphBuildFailure] = []

    for record_index, record in enumerate(records):
        record_mol_id = record.get(mol_id_key) if isinstance(record, Mapping) else None
        record_smiles = record.get(smiles_key) if isinstance(record, Mapping) else None
        try:
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record must be a mapping; got {type(record).__name__}."
                )

            missing = [
                label
                for label, field_name in field_names.items()
                if field_name not in record
            ]
            if missing:
                raise ValueError(
                    "record is missing required fields: "
                    f"{', '.join(missing)}."
                )

            graph = build_graph(
                record[molecule_key],
                conf_id=record[conf_id_key],
                config=config,
                y=record[target_key],
                smiles=record[smiles_key],
                mol_id=record[mol_id_key],
            )
        except Exception as exc:
            failures.append(
                GraphBuildFailure(
                    record_index=record_index,
                    mol_id=record_mol_id,
                    smiles=record_smiles,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        graphs.append(graph)
        successes.append(
            GraphBuildSuccess(
                record_index=record_index,
                mol_id=record[mol_id_key],
                smiles=record[smiles_key],
                graph=graph,
            )
        )

    return GraphBuildReport(
        graphs=graphs,
        successes=successes,
        failures=failures,
    )


def save_graph_dataset(
    graphs: Iterable[Any],
    path: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Persist raw graphs and optional metadata to a torch archive.

    The graphs are saved without normalization. Cache compatibility checks,
    including configuration-ID validation, belong to T6.5.
    """
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    from pathlib import Path

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graphs": list(graphs),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    torch.save(payload, target)
    return target


def load_graph_dataset(path: str) -> GraphDatasetArtifact:
    """Load a raw graph archive saved by :func:`save_graph_dataset`.

    This validates only the persistence envelope. It deliberately does not
    decide whether a cache is compatible with the current configuration.
    """
    from pathlib import Path

    import torch

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with torch versions that do not expose weights_only.
        payload = torch.load(source, map_location="cpu")

    if not isinstance(payload, Mapping):
        raise ValueError("graph archive must contain a mapping payload.")
    graphs = payload.get("graphs")
    metadata = payload.get("metadata")
    if not isinstance(graphs, list):
        raise ValueError("graph archive field 'graphs' must be a list.")
    if not isinstance(metadata, Mapping):
        raise ValueError("graph archive field 'metadata' must be a mapping.")

    return GraphDatasetArtifact(
        graphs=graphs,
        metadata=dict(metadata),
    )
