"""Compare a box-with-circle-hole to a border defined by XY points."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.plotting import plot_polygon

from xyshapes_detectors import box_with_hole, dump_geometries, polygon_from_points


def build_examples():
    box_shape = box_with_hole(width=2000, height=1400, hole_radius=150)

    # Walk around the detector border in order. The last point is connected
    # back to the first automatically, so it does not need to be repeated.
    border_points = [
        (-1000, -500),
        (-650, -750),
        (500, -700),
        (1050, -300),
        (900, 550),
        (250, 750),
        (-700, 650),
        (-1100, 150),
    ]
    point_border_hole = Point(0, 0).buffer(175)
    point_shape = polygon_from_points(border_points, hole=point_border_hole)
    return box_shape, point_shape, border_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/geometry_examples.png"),
                        help="output sketch")
    parser.add_argument("--pickle-dir", type=Path, default=Path("examples/output"),
                        help="directory for example pickle files")
    args = parser.parse_args()

    box_shape, point_shape, border_points = build_examples()
    dump_geometries(
        {"BoxWithCircleHole": box_shape, "PointDefinedBorder": point_shape},
        args.pickle_dir,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    plot_polygon(box_shape, ax=axes[0], add_points=False,
                 facecolor="tab:blue", edgecolor="navy", alpha=0.45)
    axes[0].set_title("Box with circular hole")

    plot_polygon(point_shape, ax=axes[1], add_points=False,
                 facecolor="tab:orange", edgecolor="darkred", alpha=0.45)
    xs, ys = zip(*border_points)
    axes[1].scatter(xs, ys, color="darkred", zorder=3, label="border points")
    for index, point in enumerate(border_points):
        axes[1].annotate(str(index), point, xytext=(5, 5), textcoords="offset points")
    axes[1].set_title("Point-defined border minus a hole shape")
    axes[1].legend(loc="lower right")

    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")
    print(f"wrote pickles to {args.pickle_dir}")


if __name__ == "__main__":
    main()
