import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Point

from xyshapes_detectors import (
    box_with_hole,
    dump_geometries,
    load_geometry,
    polygon_from_points,
)
from xyshapes_detectors.create_geom import all_geometries
from xyshapes_detectors.create_geom_ft import (
    FT_BASELINE_LAYOUT,
    FT_BASELINE_LAYER_LAYOUT,
    FT_LOW_LAYOUT,
    FT_LOW_LAYER_LAYOUT,
    FT_MODULE_SIZE_X,
    ft_layer_geometries,
    ft_box_geometries,
)
from xyshapes_detectors.create_geom_muon import MUON_STATIONS, muon_geometries, muon_plane_layout
from xyshapes_detectors.create_geom_tv import tv_active_geometry, tv_plane_layout, tv_z_positions
from xyshapes_detectors.layout import detector_masks, layers_by_detector, load_layout
from xyshapes_detectors.viewlhcb_acceptance import (
    load_acceptance_json,
    write_acceptance_outputs,
)
from examples.create_up_v1_borders import crop_borders


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
            figure = path.with_suffix(".png")
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 0)
        self.assertTrue(loaded.equals_exact(shape, tolerance=0))
        self.assertEqual(loaded.area, 19_800)
        self.assertFalse(loaded.contains(Point(0, 0)))

    def test_circular_hole(self):
        shape = box_with_hole(200, 100, hole_radius=10)
        self.assertTrue(shape.is_valid)
        self.assertFalse(shape.contains(Point(0, 0)))

    def test_polygon_from_border_points(self):
        border = [(-4, -2), (3, -3), (5, 1), (1, 4), (-5, 2)]
        shape = polygon_from_points(border)
        self.assertTrue(shape.is_valid)
        self.assertTrue(shape.contains(Point(0, 0)))
        self.assertEqual(list(shape.exterior.coords)[0], list(shape.exterior.coords)[-1])

    def test_polygon_from_points_with_hole(self):
        border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        hole = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
        shape = polygon_from_points(border, holes=[hole])
        self.assertFalse(shape.contains(Point(0, 0)))
        self.assertEqual(len(shape.interiors), 1)

    def test_polygon_from_points_subtracts_hole_shape(self):
        border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        hole_shape = Point(0, 0).buffer(2)
        shape = polygon_from_points(border, hole=hole_shape)
        self.assertFalse(shape.contains(Point(0, 0)))
        self.assertEqual(len(shape.interiors), 1)
        self.assertAlmostEqual(shape.area, 400 - hole_shape.area)

    def test_hole_shape_must_be_inside_border(self):
        border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        with self.assertRaises(ValueError):
            polygon_from_points(border, hole=Point(10, 0).buffer(2))

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
        self.assertEqual(len(geometries), 20)
        with tempfile.TemporaryDirectory() as directory:
            paths = dump_geometries(geometries, directory)
            self.assertEqual(len(paths), 20)
            for path in paths:
                loaded = load_geometry(path)
                self.assertTrue(loaded.is_valid, Path(path).name)
                self.assertFalse(loaded.is_empty, Path(path).name)

    def test_ft_box_geometries_from_sketch_dimensions(self):
        geometries = ft_box_geometries()
        expected = {
            "FT_8Modules": (FT_MODULE_SIZE_X * 8, 4354.0),
            "FT_10Modules": (FT_MODULE_SIZE_X * 10, 4354.0),
            "FT_12Modules": (FT_MODULE_SIZE_X * 12, 4354.0),
        }
        for name, (width, height) in expected.items():
            with self.subTest(name=name):
                shape = geometries[name]
                self.assertEqual(
                    shape.bounds,
                    (-width / 2, -height / 2, width / 2, height / 2),
                )
                self.assertEqual(shape.area, width * height - 260.0 * 260.0)
                self.assertFalse(shape.contains(Point(0, 0)))

    def test_ft_scenario_layouts(self):
        self.assertEqual(FT_LOW_LAYOUT["T1"], (7931.0, "FT_8Modules"))
        self.assertEqual(FT_LOW_LAYOUT["T2"], (8613.0, "FT_8Modules"))
        self.assertEqual(FT_LOW_LAYOUT["T3"], (9298.0, "FT_10Modules"))
        self.assertEqual(FT_BASELINE_LAYOUT["T1"], (7931.0, "FT_10Modules"))
        self.assertEqual(FT_BASELINE_LAYOUT["T2"], (8613.0, "FT_10Modules"))
        self.assertEqual(FT_BASELINE_LAYOUT["T3"], (9298.0, "FT_12Modules"))

    def test_ft_xuvx_layer_shapes(self):
        geometries = ft_layer_geometries()
        self.assertEqual(len(geometries), 9)
        x8 = geometries["FT_8Modules_X"]
        u8 = geometries["FT_8Modules_U"]
        v8 = geometries["FT_8Modules_V"]
        self.assertEqual(x8.bounds, (-FT_MODULE_SIZE_X * 4, -2177.0, FT_MODULE_SIZE_X * 4, 2177.0))
        self.assertAlmostEqual(x8.area, u8.area)
        self.assertAlmostEqual(u8.area, v8.area)
        self.assertGreater(u8.bounds[2] - u8.bounds[0], x8.bounds[2] - x8.bounds[0])
        self.assertGreater(v8.bounds[2] - v8.bounds[0], x8.bounds[2] - x8.bounds[0])
        self.assertFalse(u8.contains(Point(0, 0)))

    def test_ft_xuvx_layout_options(self):
        self.assertEqual(len(FT_LOW_LAYER_LAYOUT), 12)
        self.assertEqual(len(FT_BASELINE_LAYER_LAYOUT), 12)
        self.assertEqual(FT_LOW_LAYER_LAYOUT["T1_X1"][1], "FT_8Modules_X")
        self.assertEqual(FT_LOW_LAYER_LAYOUT["T1_U"][1], "FT_8Modules_U")
        self.assertEqual(FT_LOW_LAYER_LAYOUT["T2_V"][1], "FT_8Modules_V")
        self.assertEqual(FT_LOW_LAYER_LAYOUT["T3_X2"][1], "FT_10Modules_X")
        self.assertEqual(FT_BASELINE_LAYER_LAYOUT["T1_X1"][1], "FT_10Modules_X")
        self.assertEqual(FT_BASELINE_LAYER_LAYOUT["T2_U"][1], "FT_10Modules_U")
        self.assertEqual(FT_BASELINE_LAYER_LAYOUT["T3_V"][1], "FT_12Modules_V")

    def test_tv_geometry_is_square_with_hole_rotated_45_degrees(self):
        shape = tv_active_geometry()
        self.assertTrue(shape.is_valid)
        self.assertAlmostEqual(shape.area, 50.0 * 50.0 - 7.2 * 7.2)
        self.assertFalse(shape.contains(Point(0, 0)))
        expected_bound = 25.0 * 2 ** 0.5
        self.assertAlmostEqual(shape.bounds[0], -expected_bound)
        self.assertAlmostEqual(shape.bounds[1], -expected_bound)
        self.assertAlmostEqual(shape.bounds[2], expected_bound)
        self.assertAlmostEqual(shape.bounds[3], expected_bound)

    def test_tv_plane_layout_uses_mockup_z_bounds(self):
        positions = tv_z_positions()
        self.assertEqual(len(positions), 20)
        self.assertEqual(positions[0], 0.0)
        self.assertEqual(positions[-1], 780.0)
        layout = tv_plane_layout()
        self.assertEqual(layout["TV00"], (0.0, "TV"))
        self.assertEqual(layout["TV19"], (780.0, "TV"))

    def test_run5_layout_loads_detector_layers(self):
        layers = load_layout("examples/run5_layout.csv")
        grouped = layers_by_detector(layers)
        self.assertEqual(len(grouped["TV"]), 20)
        self.assertEqual(len(grouped["UP"]), 4)
        self.assertEqual(len(grouped["MT"]), 3)
        self.assertEqual(grouped["TV"][0].z, 0.0)
        self.assertEqual(grouped["TV"][-1].z, 780.0)
        self.assertEqual(grouped["UP"][0].name, "UPaX")
        self.assertEqual(grouped["MT"][-1].xyshape, "FT_12Modules")

    def test_detector_masks_for_run5_layout(self):
        layers = load_layout("examples/run5_layout.csv")
        masks = detector_masks(layers, lambda z: (8.0, 0.0) if z < 1000 else (200.0, 0.0))
        self.assertEqual(masks, {
            "hasTV_Run5": True,
            "hasUP_Run5": True,
            "hasMT_Run5": True,
        })

        masks = detector_masks(layers, lambda z: (0.0, 0.0))
        self.assertEqual(masks, {
            "hasTV_Run5": False,
            "hasUP_Run5": False,
            "hasMT_Run5": False,
        })

    def test_up_v1_borders_crop_rule(self):
        shape = load_geometry("UP/UPaX.pickle")
        variant = crop_borders(shape)
        width = shape.bounds[2] - shape.bounds[0]
        height = shape.bounds[3] - shape.bounds[1]
        self.assertAlmostEqual(variant.bounds[2] - variant.bounds[0], width * 5 / 6)
        self.assertAlmostEqual(variant.bounds[3] - variant.bounds[1], height * 3 / 4)
        self.assertFalse(variant.contains(Point(0, 0)))
        self.assertTrue(variant.is_valid)

    def test_muon_station_geometries(self):
        geometries = muon_geometries()
        self.assertEqual(set(geometries), {"M2", "M3", "M4", "M5"})
        for station, dimensions in MUON_STATIONS.items():
            with self.subTest(station=station):
                shape = geometries[station]
                self.assertEqual(
                    shape.bounds,
                    (
                        -dimensions["xmax"],
                        -dimensions["ymax"],
                        dimensions["xmax"],
                        dimensions["ymax"],
                    ),
                )
                self.assertAlmostEqual(
                    shape.area,
                    4 * dimensions["xmax"] * dimensions["ymax"]
                    - 4 * dimensions["xmin"] * dimensions["ymin"],
                )
                self.assertFalse(shape.contains(Point(0, 0)))

    def test_muon_plane_layout(self):
        layout = muon_plane_layout()
        self.assertEqual(layout["M2"], (15270.0, "M2"))
        self.assertEqual(layout["M5"], (18870.0, "M5"))

    def test_viewlhcb_acceptance_json_writes_pickles_and_layout(self):
        payload = """{
          "surfaces": [
            {"detector": "TV", "layer": "TV_000", "z": 1.5, "bounds": [-2, -1, 2, 1]},
            {"detector": "FT", "layer": "T1", "z": 7931, "points": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}
          ]
        }"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "acceptance.json"
            source.write_text(payload, encoding="utf-8")
            surfaces = load_acceptance_json(source)
            rows = write_acceptance_outputs(
                surfaces,
                output_dir=Path(directory) / "pickles",
                layout_path=Path(directory) / "layout.csv",
            )
            self.assertEqual(len(rows), 2)
            layers = load_layout(
                Path(directory) / "layout.csv",
                up_dir=Path(directory) / "missing-up",
                shape_dirs=[Path(directory) / "pickles"],
            )
            self.assertEqual([layer.detector for layer in layers], ["TV", "FT"])
            self.assertTrue(layers[0].contains(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
