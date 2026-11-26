"""Schema tests for local G7 density/DG node features (T4.1)."""

import unittest

from src.gnn.config import AtomicFeaturesConfig
from src.gnn.density_features import get_density_feature_names


class DensityFeatureNameTests(unittest.TestCase):
    def test_disabled_density_and_curvature_have_no_x_dg_features(self):
        self.assertEqual(get_density_feature_names(AtomicFeaturesConfig()), ())

    def test_density_names_include_configured_types_sources_other_and_kernels(self):
        config = AtomicFeaturesConfig(
            density_enabled=True,
            density_types=("number", "charge"),
            density_elements=("C", "O"),
            dg_kernel="lorentz",
            dg_kernel_configs=((3.0, 0.5), (5.0, 1.0)),
        )

        names = get_density_feature_names(config)

        self.assertEqual(
            names,
            (
                "rho_number_from_C_kernel_lorentz_0",
                "rho_number_from_C_kernel_lorentz_1",
                "rho_number_from_O_kernel_lorentz_0",
                "rho_number_from_O_kernel_lorentz_1",
                "rho_number_from_Other_kernel_lorentz_0",
                "rho_number_from_Other_kernel_lorentz_1",
                "rho_charge_from_C_kernel_lorentz_0",
                "rho_charge_from_C_kernel_lorentz_1",
                "rho_charge_from_O_kernel_lorentz_0",
                "rho_charge_from_O_kernel_lorentz_1",
                "rho_charge_from_Other_kernel_lorentz_0",
                "rho_charge_from_Other_kernel_lorentz_1",
            ),
        )
        self.assertEqual(len(names), len(set(names)))

    def test_curvature_names_are_local_and_include_validity_mask(self):
        config = AtomicFeaturesConfig(
            density_enabled=True,
            curvature_enabled=True,
            density_types=("number",),
            density_elements=("C",),
            dg_kernel_configs=((3.5, 0.3),),
        )

        self.assertEqual(
            get_density_feature_names(config),
            (
                "rho_number_from_C_kernel_exponential_0",
                "H_number_from_C_kernel_exponential_0",
                "K_number_from_C_kernel_exponential_0",
                "kappa_min_number_from_C_kernel_exponential_0",
                "kappa_max_number_from_C_kernel_exponential_0",
                "curvature_valid_number_from_C_kernel_exponential_0",
                "rho_number_from_Other_kernel_exponential_0",
                "H_number_from_Other_kernel_exponential_0",
                "K_number_from_Other_kernel_exponential_0",
                "kappa_min_number_from_Other_kernel_exponential_0",
                "kappa_max_number_from_Other_kernel_exponential_0",
                "curvature_valid_number_from_Other_kernel_exponential_0",
            ),
        )

    def test_invalid_config_type_is_rejected(self):
        with self.assertRaises(TypeError):
            get_density_feature_names(None)


if __name__ == "__main__":
    unittest.main()
