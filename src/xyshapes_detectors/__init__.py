"""Detector-plane XY acceptance generators."""

from .geometry_io import (
    box_with_hole,
    dump_geometries,
    load_geometry,
    polygon_from_points,
    save_geometry_figure,
)

__all__ = [
    "box_with_hole",
    "polygon_from_points",
    "dump_geometries",
    "load_geometry",
    "save_geometry_figure",
]
