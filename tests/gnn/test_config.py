"""Unit tests for the G7 configuration contract (T1.6).

These tests intentionally use only the Python standard library. They cover
configuration validation and serialization; feature extraction, graph
construction, normalization, and model execution belong to later tasks.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.gnn.config import (
    AtomicFeaturesConfig,
    GNNConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
)


class GraphConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = GraphConfig()
        self.assertEqual(config.edge_mode, "hybrid")
        self.assertEqual(config.cutoff, 5.0)
        self.assertEqual(config.rbf_num_centers, 32)
        self.assertTrue(config.include_bond_features)
        self.assertTrue(config.include_distance)
        self.assertTrue(config.include_rbf)

    def test_edge_mode_is_normalized(self):
        self.assertEqual(GraphConfig(edge_mode="RADIUS").edge_mode, "radius")

    def test_invalid_edge_modes_are_rejected(self):
        for value in ("unknown", "", None, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GraphConfig(edge_mode=value)

    def test_cutoff_must_be_finite_and_positive_numeric(self):
        for value in (0, -1, float("nan"), float("inf"), float("-inf"), "invalid", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GraphConfig(cutoff=value)

    def test_edge_flags_must_be_booleans(self):
        for field_name in (
            "include_bond_features",
            "include_distance",
            "include_rbf",
        ):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                GraphConfig(**{field_name: 1})

    def test_rbf_center_count_must_be_integer_at_least_two(self):
        for value in (0, 1, 2.5, True, "32"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GraphConfig(rbf_num_centers=value)

    def test_config_is_immutable(self):
        config = GraphConfig()
        with self.assertRaises(AttributeError):
            config.cutoff = 4.0


class AtomicFeaturesConfigTests(unittest.TestCase):
    def test_defaults_define_g7_a(self):
        config = AtomicFeaturesConfig()
        self.assertFalse(config.density_enabled)
        self.assertFalse(config.curvature_enabled)
        self.assertTrue(config.include_chirality)
        self.assertTrue(config.include_cip)
        self.assertEqual(
            config.chirality_types,
            (
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ),
        )
        self.assertEqual(config.cip_types, ("R", "S", "UNSPECIFIED"))

    def test_g7_b_and_g7_c_are_valid(self):
        g7_b = AtomicFeaturesConfig(density_enabled=True)
        g7_c = AtomicFeaturesConfig(density_enabled=True, curvature_enabled=True)
        self.assertTrue(g7_b.density_enabled)
        self.assertFalse(g7_b.curvature_enabled)
        self.assertTrue(g7_c.density_enabled)
        self.assertTrue(g7_c.curvature_enabled)

    def test_dg_defaults_follow_selected_kernel(self):
        exponential = AtomicFeaturesConfig(dg_kernel="exponential")
        lorentz = AtomicFeaturesConfig(dg_kernel="lorentz")
        self.assertEqual(exponential.dg_kernel_configs, ((3.5, 0.3), (2.5, 1.3)))
        self.assertEqual(lorentz.dg_kernel_configs, ((3.0, 1.3), (6.5, 0.3)))

    def test_explicit_dg_configs_override_kernel_defaults(self):
        explicit = ((4.0, 0.5),)
        config = AtomicFeaturesConfig(dg_kernel="lorentz", dg_kernel_configs=explicit)
        self.assertEqual(config.dg_kernel_configs, explicit)

    def test_boolean_fields_are_strict(self):
        for field_name in (
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
        ):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                AtomicFeaturesConfig(**{field_name: 1})

    def test_curvature_requires_density(self):
        with self.assertRaises(ValueError):
            AtomicFeaturesConfig(curvature_enabled=True)

    def test_invalid_kernel_is_rejected(self):
        for value in ("gaussian", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AtomicFeaturesConfig(dg_kernel=value)

    def test_dg_configs_must_be_nonempty_when_dg_is_enabled(self):
        with self.assertRaises(ValueError):
            AtomicFeaturesConfig(density_enabled=True, dg_kernel_configs=())

    def test_each_dg_pair_is_validated(self):
        invalid_values = (
            ((1.9, 0.3),),
            ((3.0, 0.0),),
            ((3.0, -0.1),),
            ((float("nan"), 0.3),),
            ((3.0, float("inf")),),
            ((True, 0.3),),
            ((3.0, False),),
            ((3.0,),),
            ((3.0, 0.3, 1.0),),
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                AtomicFeaturesConfig(density_enabled=True, dg_kernel_configs=value)

    def test_duplicate_dg_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            AtomicFeaturesConfig(
                density_enabled=True,
                dg_kernel_configs=((3.0, 0.3), (3.0, 0.3)),
            )

    def test_density_types_must_be_nonempty_unique_and_allowed(self):
        for value in ((), ("number", "number"), ("mass",), ("charge", "mass")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AtomicFeaturesConfig(density_types=value)

    def test_density_elements_are_real_elements_and_exclude_other(self):
        for value in (("Other",), ("C", "Other"), ("Xx",), ("C", "C"), ()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AtomicFeaturesConfig(density_elements=value)

    def test_other_label_must_be_nonempty_and_separate(self):
        for value in ("", "  ", "C"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AtomicFeaturesConfig(other_label=value)

    def test_chirality_and_cip_vocabularies_are_validated(self):
        with self.assertRaises(ValueError):
            AtomicFeaturesConfig(chirality_types=("CHI_INVALID",))
        with self.assertRaises(ValueError):
            AtomicFeaturesConfig(cip_types=("R", "L"))

    def test_curvature_neutral_value_must_be_finite_numeric(self):
        for value in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AtomicFeaturesConfig(curvature_neutral_value=value)

    def test_config_is_immutable(self):
        config = AtomicFeaturesConfig()
        with self.assertRaises(AttributeError):
            config.density_enabled = True


class ModelConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = ModelConfig()
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.num_message_passing_layers, 4)
        self.assertEqual(config.pooling, "mean")

    def test_valid_pooling_options(self):
        for pooling in ("sum", "mean", "max", "mean_max"):
            with self.subTest(pooling=pooling):
                self.assertEqual(ModelConfig(pooling=pooling).pooling, pooling)

    def test_dimensions_must_be_positive_integers(self):
        for field_name in ("hidden_dim", "num_message_passing_layers", "readout_hidden_dim"):
            for value in (0, -1, 1.5, True):
                with self.subTest(field=field_name, value=value), self.assertRaises(ValueError):
                    ModelConfig(**{field_name: value})

    def test_dropout_must_be_finite_and_in_half_open_unit_interval(self):
        for value in (-0.1, 1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModelConfig(dropout=value)

    def test_residual_flag_and_pooling_are_validated(self):
        with self.assertRaises(ValueError):
            ModelConfig(residual_connections=1)
        with self.assertRaises(ValueError):
            ModelConfig(pooling="attention")


class TrainingConfigTests(unittest.TestCase):
    def test_defaults_and_supported_losses(self):
        config = TrainingConfig()
        self.assertEqual(config.loss, "huber")
        self.assertEqual(config.batch_size, 32)
        for loss in ("huber", "mae", "mse"):
            with self.subTest(loss=loss):
                self.assertEqual(TrainingConfig(loss=loss.upper()).loss, loss)

    def test_positive_integer_fields_reject_invalid_values(self):
        for field_name in ("batch_size", "max_epochs", "early_stopping_patience"):
            for value in (0, -1, 1.5, True):
                with self.subTest(field=field_name, value=value), self.assertRaises(ValueError):
                    TrainingConfig(**{field_name: value})

    def test_learning_rate_weight_decay_and_huber_delta(self):
        invalid = (
            ("learning_rate", 0),
            ("learning_rate", float("nan")),
            ("weight_decay", -0.1),
            ("weight_decay", float("inf")),
            ("huber_delta", 0),
            ("huber_delta", float("-inf")),
            ("learning_rate", True),
        )
        for field_name, value in invalid:
            with self.subTest(field=field_name, value=value), self.assertRaises(ValueError):
                TrainingConfig(**{field_name: value})
        self.assertEqual(TrainingConfig(weight_decay=0).weight_decay, 0.0)

    def test_loss_and_gradient_clipping_are_validated(self):
        for value in ("", "rmse", None):
            with self.subTest(loss=value), self.assertRaises(ValueError):
                TrainingConfig(loss=value)
        self.assertIsNone(TrainingConfig().gradient_clip_norm)
        self.assertEqual(TrainingConfig(gradient_clip_norm=1.5).gradient_clip_norm, 1.5)
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(clip=value), self.assertRaises(ValueError):
                TrainingConfig(gradient_clip_norm=value)

    def test_config_is_immutable(self):
        config = TrainingConfig()
        with self.assertRaises(AttributeError):
            config.max_epochs = 10


class GNNConfigTests(unittest.TestCase):
    def test_defaults_compose_all_four_configs(self):
        config = GNNConfig()
        self.assertIsInstance(config.graph, GraphConfig)
        self.assertIsInstance(config.atomic_features, AtomicFeaturesConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)

    def test_custom_nested_configs_are_preserved(self):
        config = GNNConfig(
            graph=GraphConfig(edge_mode="radius"),
            atomic_features=AtomicFeaturesConfig(density_enabled=True),
            model=ModelConfig(hidden_dim=64),
            training=TrainingConfig(batch_size=16),
        )
        restored = GNNConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)

    def test_nested_types_are_enforced(self):
        for field_name in ("graph", "atomic_features", "model", "training"):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                GNNConfig(**{field_name: {}})

    def test_dict_roundtrip_uses_json_native_lists(self):
        config = GNNConfig(
            atomic_features=AtomicFeaturesConfig(
                density_enabled=True,
                dg_kernel_configs=((3.5, 0.3), (2.5, 1.3)),
            )
        )
        payload = config.to_dict()
        self.assertIsInstance(payload["atomic_features"]["dg_kernel_configs"], list)
        self.assertIsInstance(payload["atomic_features"]["dg_kernel_configs"][0], list)
        self.assertEqual(GNNConfig.from_dict(payload), config)

    def test_json_roundtrip_is_deterministic(self):
        config = GNNConfig()
        serialized = config.to_json()
        self.assertEqual(serialized, config.to_json())
        self.assertEqual(GNNConfig.from_json(serialized), config)

    def test_file_roundtrip(self):
        config = GNNConfig(
            atomic_features=AtomicFeaturesConfig(
                density_enabled=True,
                curvature_enabled=True,
                dg_kernel="lorentz",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            self.assertEqual(config.save_json(path), path)
            self.assertTrue(path.is_file())
            self.assertEqual(GNNConfig.load_json(path), config)

    def test_invalid_or_incomplete_json_is_rejected(self):
        for value in ("{", "[]", "null", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GNNConfig.from_json(value)

    def test_missing_and_unknown_sections_are_rejected(self):
        payload = GNNConfig().to_dict()
        missing = dict(payload)
        missing.pop("training")
        with self.assertRaises(ValueError):
            GNNConfig.from_dict(missing)
        unexpected = dict(payload)
        unexpected["extra"] = {}
        with self.assertRaises(ValueError):
            GNNConfig.from_dict(unexpected)

    def test_missing_and_unknown_nested_fields_are_rejected(self):
        payload = GNNConfig().to_dict()
        missing = json.loads(json.dumps(payload))
        missing["graph"].pop("cutoff")
        with self.assertRaises(ValueError):
            GNNConfig.from_dict(missing)
        unexpected = json.loads(json.dumps(payload))
        unexpected["model"]["unknown"] = 1
        with self.assertRaises(ValueError):
            GNNConfig.from_dict(unexpected)

    def test_invalid_nested_values_are_rejected(self):
        payload = GNNConfig().to_dict()
        payload["graph"]["cutoff"] = 0
        with self.assertRaises(ValueError):
            GNNConfig.from_dict(payload)

    def test_nonfinite_json_values_are_rejected(self):
        payload = GNNConfig().to_dict()
        payload["model"]["dropout"] = float("nan")
        with self.assertRaises(ValueError):
            GNNConfig.from_json(json.dumps(payload))

    def test_config_is_immutable(self):
        config = GNNConfig()
        with self.assertRaises(AttributeError):
            config.graph = GraphConfig(edge_mode="radius")


if __name__ == "__main__":
    unittest.main()
