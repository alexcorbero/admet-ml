"""Tests for local density, curvature, validity, and neutral values (T4.3-T4.7)."""

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

    def test_charge_density_uses_gasteiger_weights_after_number_channels(self):
        coords = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        mapped_symbols = np.asarray(["C", "O", "Other"])
        radii = np.asarray([1.0, 1.1, 1.2])
        charges = np.asarray([0.1, -0.2, 0.3])
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
        g6_module.get_vdw_radii = Mock(return_value=radii)
        g6_module.get_gasteiger_charges = Mock(return_value=charges)

        coordinates_module = types.ModuleType("src.features.descriptors_3d")
        coordinates_module.get_coordinates = Mock(return_value=coords)
        source_module = types.ModuleType("src.features.density_features")
        source_module.get_mapped_atom_symbols = Mock(
            return_value=mapped_symbols
        )

        mol = Mock()
        mol.GetNumAtoms.return_value = len(coords)
        config = AtomicFeaturesConfig(
            density_enabled=True,
            density_types=("number", "charge"),
            density_elements=("C", "O"),
            dg_kernel_configs=((3.0, 0.5),),
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
        np.testing.assert_array_equal(
            result[:, 0], np.full(3, 1.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result[:, 3], np.full(3, 4.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            [call["source_weights"][0] for call in calls[:3]],
            [1.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(
            [call["source_weights"][0] for call in calls[3:]],
            [0.1, -0.2, 0.3],
        )
        g6_module.get_gasteiger_charges.assert_called_once_with(mol=mol)

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

    def test_curvature_values_are_interleaved_per_local_channel(self):
        coords = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        )
        mapped_symbols = np.asarray(["C", "O", "Other"])
        radii = np.asarray([1.0, 1.1, 1.2])

        g6_module = types.ModuleType("src.features.dg_gl_features")
        g6_module.radial_kernel_derivatives = Mock()
        g6_module.density_field_derivatives = Mock(
            return_value=(
                np.ones(3),
                np.zeros((3, 3)),
                np.zeros((3, 3, 3)),
            )
        )
        g6_module.get_vdw_radii = Mock(return_value=radii)

        def fake_curvature(**kwargs):
            target_indices = np.flatnonzero(
                mapped_symbols == kwargs["target_element"]
            )
            target_value = float(target_indices[0] + 1)
            return {
                "H": np.full(target_indices.size, target_value),
                "K": np.full(target_indices.size, target_value + 10),
                "kappa_min": np.full(target_indices.size, target_value + 20),
                "kappa_max": np.full(target_indices.size, target_value + 30),
                "valid": np.ones(target_indices.size, dtype=bool),
            }

        g6_module.element_interactive_curvature_at_atoms = fake_curvature

        coordinates_module = types.ModuleType("src.features.descriptors_3d")
        coordinates_module.get_coordinates = Mock(return_value=coords)
        source_module = types.ModuleType("src.features.density_features")
        source_module.get_mapped_atom_symbols = Mock(
            return_value=mapped_symbols
        )

        mol = Mock()
        mol.GetNumAtoms.return_value = len(coords)
        config = AtomicFeaturesConfig(
            density_enabled=True,
            curvature_enabled=True,
            density_elements=("C",),
            dg_kernel_configs=((3.0, 0.5),),
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

        # Two source channels (C and Other), one kernel, four curvature
        # values plus one validity mask per channel.
        self.assertEqual(result.shape, (3, 12))
        np.testing.assert_array_equal(
            result[0], [1, 1, 11, 21, 31, 1, 1, 1, 11, 21, 31, 1]
        )
        np.testing.assert_array_equal(
            result[1], [1, 2, 12, 22, 32, 1, 1, 2, 12, 22, 32, 1]
        )
        np.testing.assert_array_equal(
            result[2], [1, 3, 13, 23, 33, 1, 1, 3, 13, 23, 33, 1]
        )

    def test_invalid_curvature_uses_configured_neutral_value(self):
        coords = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        mapped_symbols = np.asarray(["C", "O"])
        radii = np.asarray([1.0, 1.1])

        g6_module = types.ModuleType("src.features.dg_gl_features")
        g6_module.radial_kernel_derivatives = Mock()
        g6_module.density_field_derivatives = Mock(
            return_value=(
                np.ones(2),
                np.zeros((2, 3)),
                np.zeros((2, 3, 3)),
            )
        )
        g6_module.get_vdw_radii = Mock(return_value=radii)

        def fake_curvature(**kwargs):
            target_indices = np.flatnonzero(
                mapped_symbols == kwargs["target_element"]
            )
            valid = kwargs["target_element"] == "C"
            values = np.full(
                target_indices.size,
                np.nan if not valid else 1.0,
            )
            return {
                "H": values.copy(),
                "K": values.copy(),
                "kappa_min": values.copy(),
                "kappa_max": values.copy(),
                "valid": np.full(target_indices.size, valid, dtype=bool),
            }

        g6_module.element_interactive_curvature_at_atoms = fake_curvature
        coordinates_module = types.ModuleType("src.features.descriptors_3d")
        coordinates_module.get_coordinates = Mock(return_value=coords)
        source_module = types.ModuleType("src.features.density_features")
        source_module.get_mapped_atom_symbols = Mock(
            return_value=mapped_symbols
        )

        mol = Mock()
        mol.GetNumAtoms.return_value = 2
        config = AtomicFeaturesConfig(
            density_enabled=True,
            curvature_enabled=True,
            density_elements=("C",),
            dg_kernel_configs=((3.0, 0.5),),
            curvature_neutral_value=7.5,
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

        self.assertEqual(result.shape, (2, 12))
        self.assertTrue(np.all(result[0, [5, 11]] == 1.0))
        self.assertTrue(np.all(result[1, [5, 11]] == 0.0))
        np.testing.assert_array_equal(
            result[1, [1, 2, 3, 4, 7, 8, 9, 10]],
            np.full(8, 7.5, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
