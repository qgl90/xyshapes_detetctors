"""Create a rectangular XY acceptance with an optional beam hole."""

import argparse

from .geometry_io import box_with_hole, dump_geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="shape name; creates <name>.pickle")
    parser.add_argument("--width", type=float, required=True, help="outer width [mm]")
    parser.add_argument("--height", type=float, required=True, help="outer height [mm]")
    holes = parser.add_mutually_exclusive_group()
    holes.add_argument("--hole-radius", type=float, help="circular-hole radius [mm]")
    holes.add_argument("--hole-size", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"),
                       help="rectangular-hole dimensions [mm]")
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()

    hole_width, hole_height = args.hole_size if args.hole_size else (None, None)
    shape = box_with_hole(
        args.width, args.height,
        hole_width=hole_width, hole_height=hole_height,
        hole_radius=args.hole_radius,
    )
    output = dump_geometries({args.name: shape}, args.output_dir)[0]
    print(output)
    print(f"type={shape.geom_type}, area={shape.area:g} mm^2, bounds={shape.bounds}")


if __name__ == "__main__":
    main()
