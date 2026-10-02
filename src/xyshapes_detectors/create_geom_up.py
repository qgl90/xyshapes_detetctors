"""Generate the Upstream Tracker transverse acceptance shapes."""

import argparse

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .geometry_io import dump_geometries


def _ut_geometry(name, layer="x", hole_radius=66.8 / 2,
                 nstaves=16, stave_size=95.5, y_max=1338 / 2):
    if "UTU2" in name:
        width, height = 1672.0, 1355.0
        if name == "UTU2_BorderLess":
            width *= 10 / 12
            height *= 32 / 36
        outer = Polygon(((-width / 2, -height / 2), (width / 2, -height / 2),
                         (width / 2, height / 2), (-width / 2, height / 2)))
        beam_hole = Polygon(((-39, -37), (39, -37), (39, 37), (-39, 37)))
        return outer.difference(beam_hole)

    angles = {"x": 0.0, "u": 5.0, "v": -5.0}
    if layer not in angles:
        raise ValueError("layer must be x, u, or v")
    edges = np.asarray([stave_size * i for i in np.arange(-nstaves / 2, nstaves / 2 + 1)])
    staves = []
    for x_min, x_max in zip(edges[:-1], edges[1:]):
        centre = (x_min + x_max) / 2
        stave = Polygon(((x_min, -y_max), (x_max, -y_max),
                         (x_max, y_max), (x_min, y_max)))
        staves.append(affinity.rotate(stave, angles[layer], origin=(centre, 0)))
    return unary_union(staves).difference(Point(0, 0).buffer(hole_radius))


def upstream_geometries():
    return {
        "UTaX": _ut_geometry("UTaX", layer="x", nstaves=16),
        "UTaU": _ut_geometry("UTaU", layer="u", nstaves=16),
        # Preserve the +5 degree transverse shape from the original generator.
        "UTbV": _ut_geometry("UTbV", layer="u", nstaves=19),
        "UTbX": _ut_geometry("UTbX", layer="x", nstaves=19),
        "UT_U2": _ut_geometry("UTU2", nstaves=20),
        "UT_U2_BorderLess": _ut_geometry("UTU2_BorderLess", nstaves=20),
    }


UpstreamTrackerGeometries = upstream_geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()
    for path in dump_geometries(upstream_geometries(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
