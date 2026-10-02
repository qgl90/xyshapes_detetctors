"""Generate all built-in detector XY acceptances."""

import argparse

from .create_geom_ft import ft_geometries
from .create_geom_up import upstream_geometries
from .create_geom_velo import velo_geometries
from .geometry_io import dump_geometries


def all_geometries():
    geometries = {}
    geometries.update(velo_geometries())
    geometries.update(upstream_geometries())
    geometries.update(ft_geometries())
    return geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()
    for path in dump_geometries(all_geometries(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
