"""Generate SciFi and Mighty Tracker transverse acceptance shapes."""

import argparse

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .geometry_io import dump_geometries


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
    geometries = {}
    for design in ("Modest", "FTDR", "Frugal"):
        for layer in ("x", "u", "v"):
            geometries[f"{design}_Fibre_{layer}"] = mighty_tracker(design, layer, True)
        geometries[f"{design}_Pixel_x"] = mighty_tracker(design, "x", False)
    for layer in ("x", "u", "v"):
        geometries[f"SciFi_{layer}"] = mighty_tracker("FullSciFi", layer, True)
    return geometries


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

