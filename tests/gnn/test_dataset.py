"""Tests for dataset generation, persistence, and cache compatibility."""

import unittest
import pickle
import tempfile
import types
from unittest.mock import Mock, patch
from pathlib import Path

from src.gnn.config import GNNConfig
from src.gnn.dataset import (
    build_graph_dataset,
    build_graph_dataset_with_report,
    load_graph_dataset,
    save_graph_dataset,
    select_graph_subset,
    validate_raw_graph_dataset,
    validate_graph_dataset_compatibility,
    GraphDatasetArtifact,
)


class GraphDatasetTests(unittest.TestCase):
    def test_builds_graphs_in_record_order(self):
        config = GNNConfig()
        records = [
            {
                "mol": "molecule-a",
                "conf_id": 3,
                "y": -1.2,
                "smiles": "CO",
                "mol_id": "a",
            },
            {
                "mol": "molecule-b",
                "conf_id": 5,
                "y": -0.8,
                "smiles": "CCO",
                "mol_id": "b",
            },
        ]
        built = [object(), object()]

        with patch("src.gnn.dataset.build_graph", side_effect=built) as builder:
            result = build_graph_dataset(records, config)

        self.assertEqual(result, built)
        self.assertEqual(builder.call_count, 2)
        builder.assert_any_call(
            "molecule-a",
            conf_id=3,
            config=config,
            y=-1.2,
            smiles="CO",
            mol_id="a",
        )
        builder.assert_any_call(
            "molecule-b",
            conf_id=5,
            config=config,
            y=-0.8,
            smiles="CCO",
            mol_id="b",
        )

    def test_supports_custom_record_field_names(self):
        record = {
            "molecule": "mol",
            "conformer": 9,
            "target": 0.25,
            "canonical_smiles": "C",
            "stable_id": "row-1",
        }

        with patch("src.gnn.dataset.build_graph", return_value="graph") as builder:
            result = build_graph_dataset(
                [record],
                GNNConfig(),
                molecule_key="molecule",
                conf_id_key="conformer",
                target_key="target",
                smiles_key="canonical_smiles",
                mol_id_key="stable_id",
            )

        self.assertEqual(result, ["graph"])
        builder.assert_called_once_with(
            "mol",
            conf_id=9,
            config=unittest.mock.ANY,
            y=0.25,
            smiles="C",
            mol_id="row-1",
        )

    def test_rejects_non_mapping_and_missing_fields(self):
        with self.assertRaises(TypeError):
            build_graph_dataset(["not-a-record"], GNNConfig())

        incomplete = {
            "mol": "mol",
            "conf_id": 0,
            "y": 1.0,
            "smiles": "C",
        }
        with self.assertRaisesRegex(ValueError, "mol_id"):
            build_graph_dataset([incomplete], GNNConfig())

    def test_propagates_graph_builder_failures(self):
        with patch(
            "src.gnn.dataset.build_graph",
            side_effect=RuntimeError("invalid conformer"),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid conformer"):
                build_graph_dataset(
                    [
                        {
                            "mol": "mol",
                            "conf_id": 0,
                            "y": 1.0,
                            "smiles": "C",
                            "mol_id": "row-1",
                        }
                    ],
                    GNNConfig(),
                )

    def test_rejects_invalid_config_before_processing_records(self):
        builder = Mock()
        with patch("src.gnn.dataset.build_graph", builder):
            with self.assertRaises(TypeError):
                build_graph_dataset([], None)
        builder.assert_not_called()

    def test_report_keeps_successes_and_records_failures(self):
        records = [
            {
                "mol": "valid",
                "conf_id": 0,
                "y": 1.0,
                "smiles": "C",
                "mol_id": "ok",
            },
            {
                "mol": "invalid",
                "conf_id": 1,
                "y": 2.0,
                "smiles": "N",
                "mol_id": "bad",
            },
        ]
        with patch(
            "src.gnn.dataset.build_graph",
            side_effect=["graph-ok", ValueError("bad conformer")],
        ):
            report = build_graph_dataset_with_report(records, GNNConfig())

        self.assertEqual(report.graphs, ["graph-ok"])
        self.assertEqual(len(report.successes), 1)
        self.assertEqual(report.successes[0].record_index, 0)
        self.assertEqual(report.successes[0].mol_id, "ok")
        self.assertEqual(report.successes[0].graph, "graph-ok")
        self.assertEqual(len(report.failures), 1)
        failure = report.failures[0]
        self.assertEqual(failure.record_index, 1)
        self.assertEqual(failure.mol_id, "bad")
        self.assertEqual(failure.smiles, "N")
        self.assertEqual(failure.error_type, "ValueError")
        self.assertEqual(failure.error_message, "bad conformer")

    def test_report_records_malformed_records_without_stopping(self):
        records = [
            "not-a-record",
            {
                "mol": "valid",
                "conf_id": 0,
                "y": 1.0,
                "smiles": "C",
                "mol_id": "ok",
            },
        ]
        with patch("src.gnn.dataset.build_graph", return_value="graph"):
            report = build_graph_dataset_with_report(records, GNNConfig())

        self.assertEqual(report.graphs, ["graph"])
        self.assertEqual(report.failures[0].record_index, 0)
        self.assertEqual(report.failures[0].error_type, "TypeError")
        self.assertIn("mapping", report.failures[0].error_message)

    def test_save_and_load_preserves_raw_graphs_and_metadata(self):
        saved_payloads = {}

        def fake_save(payload, path):
            saved_payloads[Path(path)] = payload
            Path(path).write_bytes(pickle.dumps(payload))

        def fake_load(path, **_kwargs):
            return pickle.loads(Path(path).read_bytes())

        fake_torch = types.SimpleNamespace(save=fake_save, load=fake_load)
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "nested" / "graphs.pt"
            raw_graphs = [
                types.SimpleNamespace(normalization_state="raw"),
                types.SimpleNamespace(normalization_state="raw"),
            ]
            with patch.dict("sys.modules", {"torch": fake_torch}):
                saved_path = save_graph_dataset(
                    raw_graphs,
                    str(archive),
                    metadata={"graph_config_id": "abc", "n": 2, "raw_graphs": True},
                )
                artifact = load_graph_dataset(str(archive))

        self.assertEqual(saved_path, archive)
        self.assertEqual(artifact.graphs, raw_graphs)
        self.assertEqual(
            artifact.metadata,
            {"graph_config_id": "abc", "n": 2, "raw_graphs": True},
        )
        self.assertEqual(saved_payloads[archive]["graphs"], raw_graphs)
        self.assertTrue(saved_payloads[archive]["metadata"]["raw_graphs"])

    def test_persistence_rejects_invalid_envelope_and_metadata(self):
        fake_torch = types.SimpleNamespace(
            save=lambda payload, path: None,
            load=lambda path, **_kwargs: {"graphs": "not-a-list", "metadata": {}},
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "graphs.*list"):
                load_graph_dataset("graphs.pt")

        with self.assertRaises(TypeError):
            save_graph_dataset([], "graphs.pt", metadata=["invalid"])

    def test_compatible_cache_matches_graph_config_id(self):
        config = GNNConfig()
        artifact = GraphDatasetArtifact(
            graphs=["graph"],
            metadata={"graph_config_id": config.graph_config_id},
        )
        validate_graph_dataset_compatibility(artifact, config)

    def test_incompatible_cache_is_rejected(self):
        config = GNNConfig()
        artifact = GraphDatasetArtifact(
            graphs=["graph"],
            metadata={"graph_config_id": "different"},
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_graph_dataset_compatibility(artifact, config)

    def test_cache_without_config_id_is_rejected(self):
        artifact = GraphDatasetArtifact(graphs=["graph"], metadata={})
        with self.assertRaisesRegex(ValueError, "missing graph_config_id"):
            validate_graph_dataset_compatibility(artifact, GNNConfig())

    def test_load_with_config_validates_compatibility(self):
        config = GNNConfig()
        payload = {
            "graphs": [types.SimpleNamespace(normalization_state="raw")],
            "metadata": {
                "graph_config_id": config.graph_config_id,
                "raw_graphs": True,
            },
        }
        fake_torch = types.SimpleNamespace(
            load=lambda path, **_kwargs: payload,
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            artifact = load_graph_dataset("graphs.pt", config=config)
        self.assertEqual(artifact.graphs[0].normalization_state, "raw")

    def test_save_rejects_false_raw_graph_marker(self):
        with self.assertRaisesRegex(ValueError, "raw_graphs"):
            save_graph_dataset(
                [types.SimpleNamespace(normalization_state="raw")],
                "graphs.pt",
                metadata={"raw_graphs": False},
            )

        with self.assertRaisesRegex(ValueError, "explicitly True"):
            save_graph_dataset(
                [types.SimpleNamespace(normalization_state="raw")],
                "graphs.pt",
                metadata=None,
            )

        with self.assertRaisesRegex(ValueError, "graph 0 is normalized"):
            save_graph_dataset(
                [
                    types.SimpleNamespace(
                        normalization_state="raw",
                        normalized=True,
                    )
                ],
                "graphs.pt",
                metadata={"raw_graphs": True},
            )

    def test_raw_cache_requires_marker_and_rejects_normalized_graphs(self):
        with self.assertRaisesRegex(ValueError, "raw_graphs=True"):
            validate_raw_graph_dataset(
                GraphDatasetArtifact(graphs=["graph"], metadata={})
            )

        with self.assertRaisesRegex(ValueError, "missing normalization_state"):
            validate_raw_graph_dataset(
                GraphDatasetArtifact(
                    graphs=[types.SimpleNamespace()],
                    metadata={"raw_graphs": True},
                )
            )

        normalized_graph = types.SimpleNamespace(
            normalization_state="raw",
            normalized=True,
        )
        with self.assertRaisesRegex(ValueError, "graph 0 is normalized"):
            validate_raw_graph_dataset(
                GraphDatasetArtifact(
                    graphs=[normalized_graph],
                    metadata={"raw_graphs": True},
                )
            )

    def test_select_graph_subset_preserves_cache_order(self):
        graphs = [
            types.SimpleNamespace(mol_id="b"),
            types.SimpleNamespace(mol_id="a"),
            types.SimpleNamespace(mol_id="c"),
        ]

        selected = select_graph_subset(graphs, ["c", "b"])

        self.assertEqual([graph.mol_id for graph in selected], ["b", "c"])

    def test_select_graph_subset_rejects_missing_and_duplicate_ids(self):
        graphs = [types.SimpleNamespace(mol_id="a"), types.SimpleNamespace(mol_id="b")]

        with self.assertRaisesRegex(ValueError, "missing"):
            select_graph_subset(graphs, ["a", "missing"])
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            select_graph_subset(graphs, ["a", "a"])

    def test_select_graph_subset_rejects_duplicate_or_missing_graph_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate graph ID"):
            select_graph_subset(
                [
                    types.SimpleNamespace(mol_id="a"),
                    types.SimpleNamespace(mol_id="a"),
                ],
                ["a"],
            )
        with self.assertRaisesRegex(ValueError, "missing required ID"):
            select_graph_subset([object()], ["a"])


if __name__ == "__main__":
    unittest.main()
