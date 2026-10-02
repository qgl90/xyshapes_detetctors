"""Creation and persistence helpers for transverse detector geometries."""

from pathlib import Path
import pickle

from shapely.geometry import Point, box


def box_with_hole(width, height, *, hole_width=None, hole_height=None,
                  hole_radius=None, centre=(0.0, 0.0)):
    """Return a centred box acceptance with an optional centred hole.

    All dimensions are in millimetres. Supply ``hole_radius`` for a circular
    hole, or both ``hole_width`` and ``hole_height`` for a rectangular hole.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if hole_radius is not None and (hole_width is not None or hole_height is not None):
        raise ValueError("choose either a circular or a rectangular hole")
    if (hole_width is None) != (hole_height is None):
        raise ValueError("hole_width and hole_height must be supplied together")

    cx, cy = centre
    shape = box(cx - width / 2, cy - height / 2,
                cx + width / 2, cy + height / 2)
    if hole_radius is not None:
        if hole_radius <= 0:
            raise ValueError("hole_radius must be positive")
        if 2 * hole_radius >= min(width, height):
            raise ValueError("the circular hole must fit inside the outer box")
        shape = shape.difference(Point(cx, cy).buffer(hole_radius))
    elif hole_width is not None:
        if hole_width <= 0 or hole_height <= 0:
            raise ValueError("hole dimensions must be positive")
        if hole_width >= width or hole_height >= height:
            raise ValueError("the rectangular hole must fit inside the outer box")
        hole = box(cx - hole_width / 2, cy - hole_height / 2,
                   cx + hole_width / 2, cy + hole_height / 2)
        shape = shape.difference(hole)

    if shape.is_empty or not shape.is_valid:
        raise ValueError("could not create a valid box geometry")
    return shape


def dump_geometries(geometries, output_dir="."):
    """Dump ``{name: geometry}`` to standard ``<name>.pickle`` files."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, shape in geometries.items():
        path = destination / f"{name}.pickle"
        with path.open("wb") as output:
            pickle.dump(shape, output, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
    return paths


def load_geometry(path):
    """Load a geometry written by :func:`dump_geometries`."""
    with Path(path).open("rb") as source:
        return pickle.load(source)

