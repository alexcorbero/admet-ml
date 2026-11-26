"""Dataset-level graph generation for G7.

This module only turns already prepared molecule records into raw PyG graphs.
Failure accounting, persistence, cache validation, and subset selection are
deliberately left to their later dataset tasks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .config import GNNConfig
from .graph_builder import build_graph


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
