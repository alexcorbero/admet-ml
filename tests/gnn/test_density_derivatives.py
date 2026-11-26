"""Tests for reuse of the analytical G6 derivative primitives (T4.2)."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.gnn.density_features import (
    density_field_derivatives,
    radial_kernel_derivatives,
)


class G6DerivativeReuseTests(unittest.TestCase):
    def test_radial_derivatives_delegate_to_g6_without_reimplementation(self):
        expected = (
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        )
        g6_radial = Mock(return_value=expected)
        g6_density = Mock()
        g6_module = types.ModuleType("src.features.dg_gl_features")
        g6_module.radial_kernel_derivatives = g6_radial
        g6_module.density_field_derivatives = g6_density

        with patch.dict(sys.modules, {"src.features.dg_gl_features": g6_module}):
            result = radial_kernel_derivatives(
                distances=np.array([0.5]),
                scales=np.array([1.0]),
                kernel="lorentz",
                order=3.0,
            )

        self.assertIs(result, expected)
        self.assertEqual(g6_radial.call_count, 1)
        call_kwargs = g6_radial.call_args.kwargs
        np.testing.assert_array_equal(call_kwargs["distances"], np.array([0.5]))
        np.testing.assert_array_equal(call_kwargs["scales"], np.array([1.0]))
        self.assertEqual(call_kwargs["kernel"], "lorentz")
        self.assertEqual(call_kwargs["order"], 3.0)

    def test_density_derivatives_delegate_to_g6_without_reimplementation(self):
        expected = (
            np.array([1.0]),
            np.zeros((1, 3)),
            np.zeros((1, 3, 3)),
        )
        g6_radial = Mock()
        g6_density = Mock(return_value=expected)
        g6_module = types.ModuleType("src.features.dg_gl_features")
        g6_module.radial_kernel_derivatives = g6_radial
        g6_module.density_field_derivatives = g6_density

        evaluation_coords = np.zeros((1, 3))
        source_coords = np.ones((1, 3))
        source_weights = np.ones(1)
        pair_scales = np.ones((1, 1))
        allowed_mask = np.ones((1, 1), dtype=bool)

        with patch.dict(sys.modules, {"src.features.dg_gl_features": g6_module}):
            result = density_field_derivatives(
                evaluation_coords=evaluation_coords,
                source_coords=source_coords,
                source_weights=source_weights,
                pair_scales=pair_scales,
                allowed_mask=allowed_mask,
                kernel="exponential",
                order=3.5,
            )

        self.assertIs(result, expected)
        self.assertEqual(g6_density.call_count, 1)
        call_kwargs = g6_density.call_args.kwargs
        np.testing.assert_array_equal(
            call_kwargs["evaluation_coords"], evaluation_coords
        )
        np.testing.assert_array_equal(call_kwargs["source_coords"], source_coords)
        np.testing.assert_array_equal(call_kwargs["source_weights"], source_weights)
        np.testing.assert_array_equal(call_kwargs["pair_scales"], pair_scales)
        np.testing.assert_array_equal(call_kwargs["allowed_mask"], allowed_mask)
        self.assertEqual(call_kwargs["kernel"], "exponential")
        self.assertEqual(call_kwargs["order"], 3.5)


if __name__ == "__main__":
    unittest.main()
