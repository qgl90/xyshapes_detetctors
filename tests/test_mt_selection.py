from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import box

from xyshapes_detectors import (
    DEFAULT_COMPARISON, AcceptanceModel, GeometryComparison,
    LoadedMaskConfiguration, MTSelection, zero_field,
)
from xyshapes_detectors.layout import DetectorLayer


class MTSelectionTests(unittest.TestCase):
    def test_default_boundary_truth_table(self):
        mp = np.array([0, 0, 0, 1, 1, 2, 3, 3])
        ft = np.array([9, 10, 6, 5, 6, 9, 0, 12])
        np.testing.assert_array_equal(MTSelection().evaluate(mp, ft),
                                      [False, True, False, False, True, True, True, True])

    def test_invalid_thresholds(self):
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                MTSelection(min_mp=value)

    def test_comparison_defaults_are_the_six_holed_geometries(self):
        self.assertEqual(len(DEFAULT_COMPARISON), 6)
        self.assertTrue(all(label.endswith("_withHole") for label in DEFAULT_COMPARISON))
        self.assertTrue(all(identifier.endswith("_FT_withHole") for identifier in DEFAULT_COMPARISON.values()))

    def test_geometry_specific_selection_without_repropagation(self):
        shapes = [DetectorLayer(det, name, z, name, box(-10, -10, 10, 10), name)
                  for det, name, z in [("MP", "mp1", 10), ("FT", "ft1", 20), ("FT", "ft2", 30)]]
        config = LoadedMaskConfiguration("toy", "toy", "", Path("."), {},
                                         AcceptanceModel.from_layers(shapes, exact_surfaces=True))
        comparison = GeometryComparison(
            zero_field(), {"standard": config, "custom": config},
            mt_selection={"custom": MTSelection(min_mp=2, min_ft=3, mixed_min_mp=1, mixed_min_ft=2)},
        )
        result = comparison.label_tracks(dict(x=0, y=0, z=0, tx=0, ty=0, q_over_p=0))
        self.assertFalse(result["hasMT_standard"][0])
        self.assertTrue(result["hasMT_custom"][0])

    def test_unknown_selection_label_is_rejected(self):
        with self.assertRaises(ValueError):
            GeometryComparison(zero_field(), mt_selection={"typo": MTSelection()})
