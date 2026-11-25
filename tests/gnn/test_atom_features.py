"""Tests for the T2.1 atom-feature schema and ordered feature names."""

import unittest

from src.gnn.atom_features import get_atom_feature_names
from src.gnn.config import AtomicFeaturesConfig


class AtomFeatureSchemaTests(unittest.TestCase):
    def test_default_schema_has_stable_scalar_then_one_hot_order(self):
        self.assertEqual(
            get_atom_feature_names(),
            (
                "formal_charge",
                "heavy_atom_degree",
                "attached_hydrogens",
                "atomic_mass",
                "aromaticity",
                "ring_membership",
                "hybridization_S",
                "hybridization_SP",
                "hybridization_SP2",
                "hybridization_SP3",
                "hybridization_SP3D",
                "hybridization_SP3D2",
                "hybridization_UNSPECIFIED",
                "chirality_CHI_UNSPECIFIED",
                "chirality_CHI_TETRAHEDRAL_CW",
                "chirality_CHI_TETRAHEDRAL_CCW",
                "chirality_CHI_OTHER",
                "cip_R",
                "cip_S",
                "cip_UNSPECIFIED",
            ),
        )

    def test_default_argument_matches_explicit_default_config(self):
        self.assertEqual(
            get_atom_feature_names(),
            get_atom_feature_names(AtomicFeaturesConfig()),
        )

    def test_disabled_scalar_and_categorical_groups_are_omitted(self):
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=False,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=False,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        self.assertEqual(get_atom_feature_names(config), ())

    def test_scalar_flags_are_applied_independently(self):
        config = AtomicFeaturesConfig(
            include_formal_charge=False,
            include_heavy_atom_degree=True,
            include_attached_hydrogens=False,
            include_atomic_mass=False,
            include_aromaticity=True,
            include_ring_membership=False,
            include_hybridization=False,
            include_chirality=False,
            include_cip=False,
        )
        self.assertEqual(
            get_atom_feature_names(config),
            ("heavy_atom_degree", "aromaticity"),
        )

    def test_custom_one_hot_vocabularies_preserve_configured_order(self):
        config = AtomicFeaturesConfig(
            hybridization_types=("SP3", "SP2"),
            chirality_types=("CHI_OTHER", "CHI_UNSPECIFIED"),
            cip_types=("UNSPECIFIED", "R"),
        )
        names = get_atom_feature_names(config)
        self.assertEqual(
            names[6:],
            (
                "hybridization_SP3",
                "hybridization_SP2",
                "chirality_CHI_OTHER",
                "chirality_CHI_UNSPECIFIED",
                "cip_UNSPECIFIED",
                "cip_R",
            ),
        )

    def test_chirality_and_cip_can_be_enabled_independently(self):
        chirality_only = AtomicFeaturesConfig(include_cip=False)
        cip_only = AtomicFeaturesConfig(include_chirality=False)

        self.assertTrue(
            any(name.startswith("chirality_") for name in get_atom_feature_names(chirality_only))
        )
        self.assertFalse(
            any(name.startswith("cip_") for name in get_atom_feature_names(chirality_only))
        )
        self.assertFalse(
            any(name.startswith("chirality_") for name in get_atom_feature_names(cip_only))
        )
        self.assertTrue(
            any(name.startswith("cip_") for name in get_atom_feature_names(cip_only))
        )

    def test_atomic_number_is_not_repeated_in_x_chem_names(self):
        names = get_atom_feature_names()
        self.assertNotIn("z", names)
        self.assertNotIn("atomic_number", names)

    def test_names_are_unique_and_returned_as_immutable_tuple(self):
        names = get_atom_feature_names()
        self.assertIsInstance(names, tuple)
        self.assertEqual(len(names), len(set(names)))

    def test_invalid_config_type_is_rejected(self):
        with self.assertRaises(TypeError):
            get_atom_feature_names({})


if __name__ == "__main__":
    unittest.main()
