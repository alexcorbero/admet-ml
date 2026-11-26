"""Tests for dataset-level raw graph generation (T6.2)."""

import unittest
from unittest.mock import Mock, patch

from src.gnn.config import GNNConfig
from src.gnn.dataset import build_graph_dataset, build_graph_dataset_with_report


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


if __name__ == "__main__":
    unittest.main()
