import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Point

from xyshapes_detectors import box_with_hole, dump_geometries, load_geometry
from xyshapes_detectors.create_geom import all_geometries


class GeometryTests(unittest.TestCase):
    def test_box_without_hole(self):
        shape = box_with_hole(200, 100)
        self.assertEqual(shape.area, 20_000)
        self.assertTrue(shape.contains(Point(0, 0)))

    def test_rectangular_hole_and_pickle_round_trip(self):
        shape = box_with_hole(200, 100, hole_width=20, hole_height=10)
        with tempfile.TemporaryDirectory() as directory:
            path = dump_geometries({"Box": shape}, directory)[0]
            loaded = load_geometry(path)
        self.assertTrue(loaded.equals_exact(shape, tolerance=0))
        self.assertEqual(loaded.area, 19_800)
        self.assertFalse(loaded.contains(Point(0, 0)))

    def test_circular_hole(self):
        shape = box_with_hole(200, 100, hole_radius=10)
        self.assertTrue(shape.is_valid)
        self.assertFalse(shape.contains(Point(0, 0)))

    def test_invalid_box_dimensions(self):
        invalid = (
            {"width": 0, "height": 10},
            {"width": 10, "height": 10, "hole_width": 2},
            {"width": 10, "height": 10, "hole_radius": 5},
            {"width": 10, "height": 10, "hole_width": 10, "hole_height": 2},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                box_with_hole(**arguments)

    def test_all_built_in_geometries_round_trip(self):
        geometries = all_geometries()
        self.assertEqual(len(geometries), 27)
        with tempfile.TemporaryDirectory() as directory:
            paths = dump_geometries(geometries, directory)
            self.assertEqual(len(paths), 27)
            for path in paths:
                loaded = load_geometry(path)
                self.assertTrue(loaded.is_valid, Path(path).name)
                self.assertFalse(loaded.is_empty, Path(path).name)


if __name__ == "__main__":
    unittest.main()
