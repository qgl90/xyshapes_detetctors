import unittest

import numpy as np
from shapely.geometry import box

from examples.toy_tv_eta_acceptance import count_tv_hits, eta_scan, mean_timing_resolution


class TVToyTests(unittest.TestCase):
    def test_timing_is_mean_per_track_and_zero_hits_are_undefined(self):
        counts = np.array([[0, 1, 4], [0, 0, 0], [3, 4, 9]])
        result = mean_timing_resolution(counts)
        self.assertAlmostEqual(result[0], 37.5)
        self.assertTrue(np.isnan(result[1]))
        self.assertNotAlmostEqual(result[0], 50 / np.sqrt(2.5))
        selected = mean_timing_resolution(counts, 4)
        self.assertAlmostEqual(selected[0], 25)
        self.assertAlmostEqual(selected[2], (25 + 50 / 3) / 2)

    def test_upstream_ignored_and_multiple_sensors_count_once(self):
        layers = [
            {"surfaces": [(-10, box(-100, -100, 100, 100))]},
            {"surfaces": [(10, box(0, -1, 2, 1)), (11, box(0, -1, 2, 1))]},
        ]
        np.testing.assert_array_equal(count_tv_hits(layers, [0.1, -0.1], [0, 0]), [1, 0])

    def test_sensor_evaluated_at_own_z_and_gaps_preserved(self):
        layers = [{"surfaces": [(10, box(9, -1, 11, 1)), (20, box(-2, -1, 2, 1))]}]
        np.testing.assert_array_equal(count_tv_hits(layers, [1, 0.5], [0, 0]), [1, 0])

    def test_eta_slope_definition(self):
        # eta=asinh(10) gives tan(theta)=0.1; all rays at z=10 have r=1.
        layers = [{"surfaces": [(10, box(-1.1, -1.1, 1.1, 1.1))]}]
        _, counts = eta_scan(layers, [np.arcsinh(10), 0.5], 32)
        self.assertTrue(np.all(counts[0] == 1))
        self.assertTrue(np.all(counts[1] == 0))


if __name__ == "__main__":
    unittest.main()
