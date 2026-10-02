"""Creation and persistence helpers for transverse detector geometries."""

from pathlib import Path
import pickle

from shapely.geometry import Point, Polygon, box


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


def polygon_from_points(border, *, hole=None, holes=None):
    """Create an XY acceptance from ordered border points.

    ``border`` is an iterable of at least three ``(x, y)`` points. Shapely
    closes the final edge automatically. ``hole`` may be any Shapely geometry
    to subtract. ``holes`` may contain point sequences describing excluded
    polygonal regions.
    """
    border = list(border)
    holes = [list(hole) for hole in holes] if holes is not None else None
    if len(border) < 3:
        raise ValueError("a border needs at least three points")
    if holes is not None and any(len(hole) < 3 for hole in holes):
        raise ValueError("each hole needs at least three points")

    shape = Polygon(border, holes=holes)
    if shape.is_empty or not shape.is_valid or shape.area <= 0:
        raise ValueError("the points do not define a valid polygon")
    if hole is not None:
        if hole.is_empty or not hole.is_valid or not shape.contains(hole):
            raise ValueError("the hole must be a valid geometry inside the border")
        shape = shape.difference(hole)
    return shape


def dump_geometries(geometries, output_dir="."):
    """Dump geometries as pickles and matching PNG figures.

    Each ``name`` produces both ``<name>.pickle`` and ``<name>.png`` in the
    output directory. The returned paths are the pickle paths.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, shape in geometries.items():
        path = destination / f"{name}.pickle"
        with path.open("wb") as output:
            pickle.dump(shape, output, protocol=pickle.HIGHEST_PROTOCOL)
        save_geometry_figure(shape, destination / f"{name}.png", title=name)
        paths.append(path)
    return paths


def save_geometry_figure(shape, path, *, title=None):
    """Save a headless PNG preview of a Shapely polygon or multipolygon."""
    import matplotlib

    # Generators must also work on batch nodes without a display server.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon

    figure, axis = plt.subplots(figsize=(6, 6))
    plot_polygon(
        shape,
        ax=axis,
        add_points=False,
        facecolor="tab:blue",
        edgecolor="navy",
        alpha=0.45,
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.grid(True, alpha=0.3)
    if title:
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return Path(path)


def load_geometry(path):
    """Load a geometry written by :func:`dump_geometries`."""
    with Path(path).open("rb") as source:
        return pickle.load(source)
