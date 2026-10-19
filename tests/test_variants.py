import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

from examples.build_ft_with_hole_variants import build_variants
from xyshapes_detectors import (
    DetectorTransport, TrackState, ft_with_hole_layers,
    load_mask_configuration, mp_hole_envelope, zero_field,
)
from xyshapes_detectors.layout import DetectorLayer


def sample_layers():
    ft = box(-10, -10, 10, 10).difference(box(-1, -1, 1, 1))
    return [
        DetectorLayer("MP", "mp1", 5, "mp_left", unary_union([
            box(-6, -2, -2, 2), box(-4, -4, -2, 4),
        ]), "mp1"),
        DetectorLayer("MP", "mp2", 10, "mp_right", unary_union([
            box(2, -2, 6, 2), box(2, -4, 4, 4),
        ]), "mp2"),
        DetectorLayer("FT", "ft1", 20, "ft_x", ft, "ft1"),
        DetectorLayer("FT", "ft2", 30, "ft_u", affinity.rotate(ft, -5), "ft2"),
    ]


class FTWithHoleTests(unittest.TestCase):
    def test_solid_envelope_fills_gaps_and_preserves_stereo_borders(self):
        original = sample_layers()
        changed = ft_with_hole_layers(original)
        hole = mp_hole_envelope(original)
        active_mp = unary_union([layer.geometry for layer in original[:2]])
        gap_point = Point(0, 3)
        self.assertFalse(active_mp.contains(gap_point))
        self.assertTrue(hole.contains(gap_point))
        self.assertFalse(hole.covers(Point(5, 3)))
        for (x0, y0), (x1, y1) in zip(hole.exterior.coords[:-1], hole.exterior.coords[1:]):
            self.assertTrue(x0 == x1 or y0 == y1)
        for before, after in zip(original, changed):
            self.assertEqual((after.z, after.name, after.station),
                             (before.z, before.name, before.station))
            if before.detector == "MP":
                self.assertIs(before, after)
            else:
                self.assertTrue(after.geometry.equals(before.geometry.difference(hole)))
                self.assertEqual(after.geometry.intersection(hole).area, 0)
                self.assertEqual(after.geometry.bounds, before.geometry.bounds)
                self.assertFalse(after.geometry.contains(gap_point))
                self.assertFalse(after.geometry.contains(Point(0, 0)))
                self.assertIn("FT_withHole", after.xyshape)

    def test_missing_detectors_are_rejected(self):
        layers = sample_layers()
        with self.assertRaisesRegex(ValueError, "MP"):
            ft_with_hole_layers(layers[2:])
        with self.assertRaisesRegex(ValueError, "FT"):
            ft_with_hole_layers(layers[:2])

    def test_only_central_x_gap_is_filled(self):
        left = box(-8, -5, -2, 5).difference(box(-7, -1, -5, 1))
        right = box(2, -5, 8, 5)
        layers = [
            DetectorLayer("MP", "left", 10, "left", left),
            DetectorLayer("MP", "right", 20, "right", right),
        ]
        envelope = mp_hole_envelope(layers)
        self.assertTrue(envelope.contains(Point(0, 0)))
        self.assertFalse(envelope.contains(Point(-6, 0)))

    def test_saved_variants_load_by_id_and_path_and_reject_gap_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            shapes = root / "shared"
            source.mkdir()
            shapes.mkdir()
            layers = sample_layers()
            for layer in layers:
                with (shapes / f"{layer.xyshape}.pickle").open("wb") as handle:
                    pickle.dump(layer.geometry, handle)
            with (source / "layout.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("Use", "detector", "layer", "station", "z", "xyshape"))
                for layer in layers:
                    writer.writerow((1, layer.detector, layer.name, layer.station,
                                     layer.z, layer.xyshape))
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"configurations": [{
                "id": "source", "label": "source", "description": "fixture",
                "setup": "source", "shape_dirs": ["shared"],
            }]}))
            build_variants(catalog)
            build_variants(catalog)
            self.assertEqual(len(json.loads(catalog.read_text())["configurations"]), 2)
            by_id = load_mask_configuration("source_FT_withHole", catalog=catalog)
            by_path = load_mask_configuration(root / "source_FT_withHole")
            state = TrackState(0, 3, 0, 0, 0)
            base = load_mask_configuration("source", catalog=catalog)
            self.assertEqual(DetectorTransport(base, zero_field()).propagate(state).labels["nFT"], 2)
            shared_crossings = DetectorTransport(base, zero_field()).propagate_batch([state]).crossings
            for variant in (by_id, by_path):
                service = DetectorTransport(variant, zero_field())
                result = service.propagate(state)
                self.assertEqual(result.labels["nFT"], 0)
                self.assertEqual(result.labels["nMP"], 0)
                reused = service.labels_from_crossings(shared_crossings)
                self.assertEqual(reused["nFT"][0], result.labels["nFT"])


if __name__ == "__main__":
    unittest.main()
