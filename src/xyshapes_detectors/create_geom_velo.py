"""Generate the VELO transverse acceptance shapes."""

import argparse
from math import sqrt

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geometry_io import dump_geometries


def velo_geometries():
    inner = Polygon(((-4.7, -4.7), (-4.7, 4.7), (4.7, 4.7), (4.7, -4.7)))
    outer = Polygon(((-35.4, -35.4), (-35.4, 35.4), (35.4, 35.4), (35.4, -35.4)))
    closed = affinity.rotate(outer, 45, origin=(0, 0)).difference(
        affinity.rotate(inner, 45, origin=(0, 0))
    )

    left_inner = Polygon(((-4.7, -4.7), (-4.7, 4.7), (0, 4.7), (0, -4.7)))
    left_outer = Polygon(((-35.4, -35.4), (-35.4, 35.4), (0, 35.4), (0, -35.4)))
    left = affinity.rotate(left_outer, 45, origin=(0, 0)).difference(
        affinity.rotate(left_inner, 45, origin=(0, 0))
    )

    right_inner = Polygon(((0, -4.7), (0, 4.7), (4.7, 4.7), (4.7, -4.7)))
    right_outer = Polygon(((0, -35.4), (0, 35.4), (35.4, 35.4), (35.4, -35.4)))
    right = affinity.rotate(right_outer, 45, origin=(0, 0)).difference(
        affinity.rotate(right_inner, 45, origin=(0, 0))
    )

    offset = 4.9 / sqrt(2)
    left_open = affinity.translate(left, xoff=-offset, yoff=-offset)
    right_open = affinity.translate(right, xoff=offset, yoff=offset)
    return {
        "VeloClose": closed,
        "VeloLeft": left,
        "VeloRight": right,
        "VeloOpen": unary_union((left_open, right_open)),
        "VeloLeftOpen": left_open,
        "VeloRightOpen": right_open,
    }


VeloGeometries = velo_geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()
    for path in dump_geometries(velo_geometries(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

