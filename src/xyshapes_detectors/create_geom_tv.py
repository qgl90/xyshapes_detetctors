"""Generate TV detector transverse acceptance shapes."""

import argparse

from shapely import affinity
from shapely.geometry import box

from .geometry_io import dump_geometries


TV_OUTER_SIZE = 50.0
TV_OUTER_HALF_SIZE = TV_OUTER_SIZE / 2
TV_HOLE_SIZE = 7.2
TV_HOLE_HALF_SIZE = TV_HOLE_SIZE / 2
TV_ROTATION_DEGREES = 45.0
TV_Z_MIN = 0.0
TV_Z_MAX = 780.0
TV_STATIONS = 20


def tv_active_geometry():
    """Return the TV active area rotated into the global XY reference."""
    outer = box(
        -TV_OUTER_HALF_SIZE,
        -TV_OUTER_HALF_SIZE,
        TV_OUTER_HALF_SIZE,
        TV_OUTER_HALF_SIZE,
    )
    hole = box(
        -TV_HOLE_HALF_SIZE,
        -TV_HOLE_HALF_SIZE,
        TV_HOLE_HALF_SIZE,
        TV_HOLE_HALF_SIZE,
    )
    return affinity.rotate(outer.difference(hole), TV_ROTATION_DEGREES, origin=(0, 0))


def tv_z_positions():
    """Return provisional TV Z positions in millimetres, including the bounds."""
    if TV_STATIONS < 2:
        raise ValueError("TV_STATIONS must include at least the two bounds")
    step = (TV_Z_MAX - TV_Z_MIN) / (TV_STATIONS - 1)
    return [TV_Z_MIN + index * step for index in range(TV_STATIONS)]


def tv_geometries():
    return {"TV": tv_active_geometry()}


def tv_plane_layout():
    return {
        f"TV{index:02d}": (z, "TV")
        for index, z in enumerate(tv_z_positions())
    }


TVGeometries = tv_geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()
    for path in dump_geometries(tv_geometries(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
