"""Tests for per-atom number density features (T4.3)."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.gnn.config import AtomicFeaturesConfig
from src.gnn.density_features import build_density_features


class DensityFeatureBuildTests(unittest.TestCase):
    def test_builds_one_column_per_source_and_kernel_in_name_order(self):
        coords = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        mapped_symbols = np.asarray(["C", "O", "Other"])
        radii = np.asarray([1.0, 1.1, 1.2])
        calls = []

        def fake_derivatives(**kwargs):
            calls.append(kwargs)
            value = float(len(calls))
            n_target = len(kwargs["evaluation_coords"])
            return (
                np.full(n_target, value),
                np.zeros((n_target, 3)),
                np.zeros((n_target, 3, 3)),
            )

        g6_module = types.ModuleType("src.features.dg_gl_features")
        g6_module.radial_kernel_derivatives = Mock()
        g6_module.density_field_derivatives = fake_derivatives

        coordinates_module = types.ModuleType("src.features.descriptors_3d")
        coordinates_module.get_coordinates = Mock(return_value=coords)

        source_module = types.ModuleType("src.features.density_features")
        source_module.get_mapped_atom_symbols = Mock(
            return_value=mapped_symbols
        )
        g6_module.get_vdw_radii = Mock(return_value=radii)

        mol = Mock()
        mol.GetNumAtoms.return_value = len(coords)
        config = AtomicFeaturesConfig(
            density_enabled=True,
            density_elements=("C", "O"),
            dg_kernel="lorentz",
            dg_kernel_configs=((3.0, 0.5), (5.0, 1.0)),
        )

        with patch.dict(
            sys.modules,
            {
                "src.features.dg_gl_features": g6_module,
                "src.features.descriptors_3d": coordinates_module,
                "src.features.density_features": source_module,
            },
        ):
            result = build_density_features(mol, conf_id=7, config=config)

        self.assertEqual(result.shape, (3, 6))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(
            result,
            np.asarray(
                [
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(len(calls), 6)

        # Columns are grouped by source channel, then kernel configuration.
        source_sizes = [call["source_coords"].shape[0] for call in calls]
        self.assertEqual(source_sizes, [1, 1, 1, 1, 1, 1])
        for call in calls:
            np.testing.assert_array_equal(
                call["source_weights"], np.ones(1)
            )
            self.assertEqual(call["kernel"], "lorentz")

    def test_disabled_density_returns_empty_node_matrix(self):
        mol = Mock()
        mol.GetNumAtoms.return_value = 4

        result = build_density_features(
            mol,
            conf_id=0,
            config=AtomicFeaturesConfig(),
        )

        self.assertEqual(result.shape, (4, 0))
        self.assertEqual(result.dtype, np.float32)

    def test_later_density_and_curvature_tasks_are_not_implemented_here(self):
        mol = Mock()
        mol.GetNumAtoms.return_value = 1

        charge_config = AtomicFeaturesConfig(
            density_enabled=True,
            density_types=("charge",),
        )
        with self.assertRaises(NotImplementedError):
            build_density_features(mol, conf_id=0, config=charge_config)

        curvature_config = AtomicFeaturesConfig(
            density_enabled=True,
            curvature_enabled=True,
        )
        with self.assertRaises(NotImplementedError):
            build_density_features(mol, conf_id=0, config=curvature_config)


if __name__ == "__main__":
    unittest.main()
