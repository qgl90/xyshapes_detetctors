"""Generate Mighty Tracker transverse acceptance shapes."""

import argparse

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .geometry_io import box_with_hole, dump_geometries


FT_MODULE_SIZE_X = 529.3
FT_HEIGHT = 4354.0
FT_HOLE_SIZE = 260.0
FT_MODULE_COUNTS = {
    "FT_8Modules": 8,
    "FT_10Modules": 10,
    "FT_12Modules": 12,
}
FT_LAYER_SEQUENCE = ("X1", "U", "V", "X2")
FT_LAYER_ANGLES = {
    "X1": 0.0,
    "U": -5.0,
    "V": 5.0,
    "X2": 0.0,
}
FT_LAYER_Z_OFFSETS = {
    "X1": -30.0,
    "U": -10.0,
    "V": 10.0,
    "X2": 30.0,
}
FT_BOX_DIMENSIONS = {
    name: (FT_MODULE_SIZE_X * modules, FT_HEIGHT)
    for name, modules in FT_MODULE_COUNTS.items()
}
FT_LOW_LAYOUT = {
    "T1": (7931.0, "FT_8Modules"),
    "T2": (8613.0, "FT_8Modules"),
    "T3": (9298.0, "FT_10Modules"),
}
FT_BASELINE_LAYOUT = {
    "T1": (7931.0, "FT_10Modules"),
    "T2": (8613.0, "FT_10Modules"),
    "T3": (9298.0, "FT_12Modules"),
}
FT_PLANE_LAYOUT = FT_BASELINE_LAYOUT


def _layer_angle(layer):
    try:
        return {"x": 0.0, "u": -5.0, "v": 5.0}[layer]
    except KeyError as error:
        raise ValueError("layer must be x, u, or v") from error


def mighty_tracker(flag="FullSciFi", layer="x", fibre_part=True):
    """Build one FT/SciFi layer, preserving the original geometry definition."""
    if flag not in {"Frugal", "Modest", "FTDR", "FullSciFi"}:
        raise ValueError("flag must be Frugal, Modest, FTDR, or FullSciFi")

    module_size = 528.0
    y_max = 2700.0
    beam_hole = Polygon(((-130, -130), (130, -130), (130, 130), (-130, 130)))
    angle = _layer_angle(layer)
    edges = np.asarray([module_size * i for i in range(-6, 7)])
    quadrant_coordinates = {
        "Modest": ((0, 0), (module_size * 3, 0), (module_size * 3, 200),
                   (module_size * 2, 200), (module_size * 2, 300),
                   (module_size, 300), (module_size, 500), (0, 500)),
        "Frugal": ((0, 0), (module_size * 3, 0), (module_size * 3, 100),
                   (module_size * 2, 100), (module_size * 2, 200),
                   (module_size, 200), (module_size, 300), (0, 300)),
        "FTDR": ((0, 0), (module_size * 4, 0), (module_size * 4, 200),
                 (module_size * 3, 200), (module_size * 3, 300),
                 (module_size * 2, 300), (module_size * 2, 400),
                 (module_size, 400), (module_size, 500), (0, 500)),
        "FullSciFi": ((-135, -120), (135, -120), (135, 120), (-135, 120)),
    }
    q1 = Polygon(quadrant_coordinates[flag])
    q2 = affinity.scale(q1, xfact=-1, yfact=1, origin=Point(0, 0))
    q3 = affinity.rotate(q1, 180, origin=(0, 0))
    q4 = affinity.scale(q3, xfact=-1, yfact=1, origin=Point(0, 0))
    pixel_region = unary_union((q1, q2, q3, q4))

    fibres, pixels = [], []
    for x_min, x_max in zip(edges[:-1], edges[1:]):
        centre = ((x_min + x_max) / 2, 0)
        for y_min, y_max_side in ((0, y_max), (-y_max, 0)):
            module = Polygon(((x_min, y_min), (x_max, y_min),
                              (x_max, y_max_side), (x_min, y_max_side)))
            if flag == "FullSciFi":
                fibre = module
            else:
                fibre = module.difference(pixel_region) if module.overlaps(pixel_region) else module
            pixel = module.difference(fibre).difference(beam_hole)
            fibres.append(affinity.rotate(fibre, angle, origin=centre))
            pixels.append(affinity.rotate(pixel, angle, origin=centre))

    fibre_geometry = unary_union(fibres)
    if flag == "FullSciFi":
        fibre_geometry = fibre_geometry.difference(
            affinity.rotate(beam_hole, angle, origin=(0, 0))
        )
    return fibre_geometry if fibre_part else unary_union(pixels)


def ft_geometries():
    return ft_box_geometries()


def ft_box_geometries():
    return {
        name: box_with_hole(
            width,
            height,
            hole_width=FT_HOLE_SIZE,
            hole_height=FT_HOLE_SIZE,
        )
        for name, (width, height) in FT_BOX_DIMENSIONS.items()
    }


def ft_layer_geometry(modules, layer):
    """Return an FT layer with module-count crop applied before stereo rotation."""
    if modules not in {8, 10, 12}:
        raise ValueError("modules must be one of 8, 10, or 12")
    if layer not in FT_LAYER_ANGLES:
        raise ValueError(f"layer must be one of {', '.join(FT_LAYER_ANGLES)}")

    shape = box_with_hole(
        FT_MODULE_SIZE_X * modules,
        FT_HEIGHT,
        hole_width=FT_HOLE_SIZE,
        hole_height=FT_HOLE_SIZE,
    )
    return affinity.rotate(shape, FT_LAYER_ANGLES[layer], origin=(0, 0))


def ft_layer_geometries():
    """Return reusable X/U/V FT layer shapes for 8, 10, and 12 modules."""
    geometries = {}
    for modules in (8, 10, 12):
        for layer in ("X", "U", "V"):
            key = "X1" if layer == "X" else layer
            geometries[f"FT_{modules}Modules_{layer}"] = ft_layer_geometry(modules, key)
    return geometries


def ft_station_layer_layout(station_modules, station_z=None):
    """Return a station/layer layout using X, U, V, X layer ordering.

    ``station_modules`` maps station names such as ``T1`` to 8, 10, or 12.
    ``station_z`` maps station names to their center z positions in millimetres;
    when omitted, the baseline station centers are used.
    """
    station_z = station_z or {
        station: z
        for station, (z, _) in FT_BASELINE_LAYOUT.items()
    }
    layout = {}
    for station, modules in station_modules.items():
        for layer in FT_LAYER_SEQUENCE:
            shape_layer = "X" if layer.startswith("X") else layer
            layout[f"{station}_{layer}"] = (
                station_z[station] + FT_LAYER_Z_OFFSETS[layer],
                f"FT_{modules}Modules_{shape_layer}",
            )
    return layout


FT_LOW_LAYER_LAYOUT = ft_station_layer_layout({
    "T1": 8,
    "T2": 8,
    "T3": 10,
})
FT_BASELINE_LAYER_LAYOUT = ft_station_layer_layout({
    "T1": 10,
    "T2": 10,
    "T3": 12,
})


MightyTracker = mighty_tracker
DownstreamTrackerGeometries = ft_geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()
    for path in dump_geometries(ft_geometries(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
