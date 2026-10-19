from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box

from examples.toy_tv_eta_acceptance import count_tv_hits
from xyshapes_detectors import (
    AcceptanceModel, DetectorTransport, GeometryComparison, LoadedMaskConfiguration,
    TrackState, TrajectoryError, available_configurations, label_tracks,
    load_mask_configuration, zero_field,
)
from xyshapes_detectors.layout import DetectorLayer
from xyshapes_detectors.propagation import propagate
from xyshapes_detectors.tv_geometry import tv_sensor_layers


ROOT = Path(__file__).resolve().parents[1]


def toy_config(name, up_width=10, up_z=100):
    layers = [DetectorLayer("TV", "upstream", -10, "upstream", box(-10, -10, 10, 10), "TV0")]
    for i, z in enumerate([10, 20, 30, 40], 1):
        layers.append(DetectorLayer("TV", f"TV{i}", z, f"TV{i}", box(-10, -10, 10, 10), f"TV{i}"))
    layers.append(DetectorLayer("UP", "UP1", up_z, "UP1", box(-up_width, -10, up_width, 10), "UP1"))
    model = AcceptanceModel.from_layers(layers, exact_surfaces=True)
    return LoadedMaskConfiguration(name, name, "test", Path("."), {}, model)


class ComparisonTests(unittest.TestCase):
    def test_single_propagation_different_z_and_suffixed_threshold_flags(self):
        choices = {"wide": toy_config("wide"), "small": toy_config("small", 1, 120)}
        states = dict(x=0, y=0, z=[0, 15], tx=[.05, 1], ty=0, q_over_p=0)
        with patch("xyshapes_detectors.transport.propagate", wraps=propagate) as run:
            result = label_tracks(states, field=zero_field(), configurations=choices)
            self.assertEqual(run.call_count, 1)
        np.testing.assert_array_equal(result["nTV_wide"], [4, 1])
        np.testing.assert_array_equal(result["hasTV_3Hits_wide"], [True, False])
        np.testing.assert_array_equal(result["hasTV_4Hits_small"], [True, False])
        np.testing.assert_array_equal(result["nUP_wide"], [1, 0])
        np.testing.assert_array_equal(result["nUP_small"], [0, 0])
        self.assertFalse(any(key == "nTV" for key in result))

    def test_scalar_origin_transport_ignores_upstream_planes(self):
        config = toy_config("origin")
        service = DetectorTransport(config, zero_field())
        state = TrackState(0, 0, .05, 0, 0)
        scalar = service.propagate(state).labels
        batch = service.propagate_batch([state]).labels
        self.assertEqual(scalar["nTV"], 4)
        self.assertTrue(scalar["hasTV_4Hits"])
        for key, value in scalar.items():
            self.assertEqual(value, batch[key][0], key)

    def test_incomplete_downstream_transport_is_not_a_mask_miss(self):
        service = GeometryComparison(zero_field(), {"toy": toy_config("toy")})
        with self.assertRaises(TrajectoryError):
            service.label_tracks([TrackState(0, 0, 0, 0, 0)], max_steps=0)

    def test_bad_labels_and_ambiguous_field_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            GeometryComparison(zero_field(), {"bad-label": toy_config("toy")})
        with self.assertRaises(ValueError):
            label_tracks([], field=zero_field(), field_path="map.root")


class SavedTVGeometryTests(unittest.TestCase):
    def test_canonical_geometry_counting_and_placements(self):
        layers = tv_sensor_layers()
        self.assertEqual(len(layers), 100)
        self.assertEqual(len({layer.station for layer in layers}), 50)
        self.assertAlmostEqual(min(r.z for r in layers), -287.996)
        self.assertAlmostEqual(max(r.z for r in layers), 558.146)
        self.assertEqual(sum("_left_" in r.name for r in layers), 50)
        self.assertEqual(sum("_right_" in r.name for r in layers), 50)

    def test_every_catalog_geometry_has_identical_sensor_masks(self):
        expected = {r.name: r for r in tv_sensor_layers()}
        for entry in available_configurations():
            with self.subTest(configuration=entry["id"]):
                for config in [load_mask_configuration(entry["id"]), load_mask_configuration(entry["path"])]:
                    stations = config.model.stations_of("TV")
                    self.assertEqual(len(stations), 50)
                    surfaces = [(name, z, geometry) for s in stations
                                for name, (z, geometry) in zip(s.layers, s.surfaces)]
                    self.assertEqual(len(surfaces), 100)
                    for name, z, geometry in surfaces:
                        self.assertAlmostEqual(z, expected[name].z)
                        self.assertTrue(geometry.equals(expected[name].geometry))

    def test_full_six_geometry_call_agrees_with_origin_sensor_toy(self):
        eta = np.array([1.5, 2, 3, 4.4, 4.6, 4.8, 5, 5.5])
        tx, ty = np.cos(.4)/np.sinh(eta), np.sin(.4)/np.sinh(eta)
        columns = label_tracks(dict(x=0, y=0, z=0, tx=tx, ty=ty, q_over_p=0), field=zero_field())
        layers = tv_sensor_layers()
        grouped = {}
        for layer in layers:
            grouped.setdefault(layer.station, {"surfaces": []})["surfaces"].append((layer.z, layer.geometry))
        expected = count_tv_hits(list(grouped.values()), tx, ty)
        for up in ["gdml", "y8", "y8x6"]:
            for ft in ["8_8_10", "10_10_12"]:
                suffix = f"up_{up}_ft_{ft}_withHole"
                np.testing.assert_array_equal(columns[f"nTV_{suffix}"], expected)
                np.testing.assert_array_equal(columns[f"hasTV_3Hits_{suffix}"], expected >= 3)
                np.testing.assert_array_equal(columns[f"hasTV_4Hits_{suffix}"], expected >= 4)
                self.assertIn(f"nUP_{suffix}", columns)
                self.assertIn(f"nFT_{suffix}", columns)

    def test_border_tracks_expose_up_and_ft_acceptance_differences(self):
        columns = label_tracks(dict(x=0, y=0, z=0, tx=[.25, .29, .1],
                                   ty=[.05, .05, .22], q_over_p=0), field=zero_field())
        nominal = columns["nUP_up_gdml_ft_8_8_10_withHole"]
        ycrop = columns["nUP_up_y8_ft_8_8_10_withHole"]
        xycrop = columns["nUP_up_y8x6_ft_8_8_10_withHole"]
        self.assertTrue(np.all(nominal >= ycrop))
        self.assertTrue(np.all(ycrop >= xycrop))
        self.assertTrue(np.any(nominal > xycrop))
        low = columns["nFT_up_gdml_ft_8_8_10_withHole"]
        baseline = columns["nFT_up_gdml_ft_10_10_12_withHole"]
        self.assertTrue(np.all(baseline >= low))
        self.assertTrue(np.any(baseline > low))
