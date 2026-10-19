import unittest
from unittest.mock import patch

import numpy as np

from xyshapes_detectors import (
    DetectorTransport, MTSelection, PolarityComparison, TrackState, UniformField,
    label_tracks_both_polarities, zero_field,
)
from xyshapes_detectors.propagation import propagate
from tests.test_comparison import toy_config


class PolarityComparisonTests(unittest.TestCase):
    def test_two_transports_not_two_per_geometry_and_identical_track_order(self):
        configs = {"one": toy_config("one"), "two": toy_config("two", up_z=120)}
        with patch("xyshapes_detectors.transport.propagate", wraps=propagate) as run:
            columns = label_tracks_both_polarities(
                dict(x=0, y=0, z=0, tx=[.01, 1], ty=0, q_over_p=0),
                field=zero_field(), configurations=configs,
            )
            self.assertEqual(run.call_count, 2)
        for label in configs:
            np.testing.assert_array_equal(columns[f"nTV_{label}_magup"],
                                          columns[f"nTV_{label}_magdown"])
            self.assertEqual(columns[f"nTV_{label}_magup"].shape, (2,))

    def test_full_vector_flip_and_explicit_map_polarity(self):
        field = UniformField(.1, -1, .2)
        configs = {"toy": toy_config("toy")}
        service = PolarityComparison(field, configs, map_polarity="magdown")
        self.assertEqual(service.polarity_factors, {"magup": -1, "magdown": 1})
        for polarity, factor in service.polarity_factors.items():
            actual = service.comparisons[polarity].services["toy"].field.b_at(0, 0, 10)
            np.testing.assert_allclose(actual, factor * field.b_at(0, 0, 10))
        inverse = PolarityComparison(field, configs, map_polarity="magup")
        self.assertEqual(inverse.polarity_factors, {"magup": 1, "magdown": -1})

    def test_both_signs_agree_with_individual_transports(self):
        config = toy_config("toy", up_width=.4)
        field = UniformField(.1, -1, .2)
        states = [TrackState(0, 0, .003, 0, 1/1000), TrackState(0, 0, -.003, 0, -1/1000)]
        columns = label_tracks_both_polarities(iter(states), field=field,
                                              configurations={"toy": config})
        for polarity, factor in [("magup", -1), ("magdown", 1)]:
            expected = DetectorTransport(config, field.scaled(factor)).propagate_batch(states).labels
            for key, values in expected.items():
                if not key.startswith("fired_"):
                    np.testing.assert_array_equal(columns[f"{key}_toy_{polarity}"], values)

    def test_invalid_map_label_or_ambiguous_field(self):
        with self.assertRaises(ValueError):
            PolarityComparison(zero_field(), map_polarity="unknown")
        with self.assertRaises(ValueError):
            label_tracks_both_polarities([], field=zero_field(), field_path="map.root")
