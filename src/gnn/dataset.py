"""Dataset-level graph generation for G7.

This module turns already prepared molecule records into raw PyG graphs,
persists/loads those raw graphs, validates cache compatibility, and selects
subsets by stable molecule IDs.
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


def select_graph_subset(
    graphs: Iterable[Any],
    split_ids: Iterable[Any],
    *,
    id_attribute: str = "mol_id",
) -> list[Any]:
    """Select cached graphs whose stable IDs belong to a split.

    Selection preserves the order of ``graphs`` so that filtering a raw cache
    cannot silently reorder the dataset. Every requested split ID must occur
    exactly once in the cache; missing IDs and duplicate IDs are rejected
    explicitly. The default ``mol_id`` attribute is the stable graph ID
    attached by :func:`src.gnn.graph_builder.build_graph`.
    """
    if not isinstance(id_attribute, str) or not id_attribute:
        raise ValueError("id_attribute must be a non-empty string.")

    cached_graphs = list(graphs)
    requested_ids = list(split_ids)
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("split_ids must not contain duplicate IDs.")

    graph_by_id: dict[Any, Any] = {}
    for graph_index, graph in enumerate(cached_graphs):
        graph_id = getattr(graph, id_attribute, None)
        if graph_id is None:
            raise ValueError(
                f"graph {graph_index} is missing required ID attribute "
                f"'{id_attribute}'."
            )
        if graph_id in graph_by_id:
            raise ValueError(f"duplicate graph ID in cache: {graph_id!r}.")
        graph_by_id[graph_id] = graph

    missing_ids = [split_id for split_id in requested_ids if split_id not in graph_by_id]
    if missing_ids:
        raise ValueError(f"split IDs are missing from cache: {missing_ids!r}.")

    requested_id_set = set(requested_ids)
    return [
        graph
        for graph in cached_graphs
        if getattr(graph, id_attribute) in requested_id_set
    ]


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


def load_graph_dataset(
    path: str,
    *,
    config: GNNConfig | None = None,
) -> GraphDatasetArtifact:
    """Load a raw graph archive saved by :func:`save_graph_dataset`.

    This validates the persistence envelope. When ``config`` is supplied, the
    cache is also checked against its graph-construction configuration.
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

    artifact = GraphDatasetArtifact(
        graphs=graphs,
        metadata=dict(metadata),
    )
    if config is not None:
        validate_graph_dataset_compatibility(artifact, config)
    return artifact


def validate_graph_dataset_compatibility(
    artifact: GraphDatasetArtifact,
    config: GNNConfig,
) -> None:
    """Raise ``ValueError`` when an artifact is incompatible with ``config``.

    The graph configuration ID is the canonical compatibility key. A missing
    ID is incompatible rather than being treated as a legacy compatible
    cache. Model and training settings are intentionally not checked here.
    """
    if not isinstance(artifact, GraphDatasetArtifact):
        raise TypeError("artifact must be a GraphDatasetArtifact instance.")
    if not isinstance(config, GNNConfig):
        raise TypeError("config must be a GNNConfig instance.")

    cache_id = artifact.metadata.get("graph_config_id")
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError(
            "graph cache is incompatible: metadata is missing graph_config_id."
        )
    expected_id = config.graph_config_id
    if cache_id != expected_id:
        raise ValueError(
            "graph cache is incompatible: graph_config_id mismatch; "
            f"expected {expected_id}, got {cache_id}."
        )
