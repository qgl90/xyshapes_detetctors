"""Derived XY acceptance variants with unchanged detector placements."""

from dataclasses import replace
import hashlib

from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


def mp_hole_envelope(layers):
    """Unite MP left/right sides across the central gap near global x=0.

    Keep both sides' original square stair-step contours. In each y band,
    bridge only the nearest negative-x and positive-x edges across the centre.
    This closes the central aperture without interpolating the outer borders.
    No projection in z or rotation into an FT layer's local frame is applied.
    """
    masks = [layer.geometry for layer in layers if layer.detector == "MP"]
    if not masks:
        raise ValueError("FT_withHole needs at least one MP acceptance surface")
    if any(mask.is_empty or not mask.is_valid for mask in masks):
        raise ValueError("MP acceptance surfaces must be valid and nonempty")
    active = unary_union(masks)
    polygons = active.geoms if hasattr(active, "geoms") else (active,)
    y_levels = set()
    original_holes = []
    for polygon in polygons:
        original_holes.extend(Polygon(ring) for ring in polygon.interiors)
        for ring in (polygon.exterior, *polygon.interiors):
            points = list(ring.coords)
            y_levels.update(y for x, y in points)
            if any(abs(x1 - x0) > 1e-7 and abs(y1 - y0) > 1e-7
                   for (x0, y0), (x1, y1) in zip(points[:-1], points[1:])):
                raise ValueError("the stepped MP envelope needs axis-aligned XY borders")
    y_levels = sorted(y_levels)
    xmin, ymin, xmax, ymax = active.bounds
    left_side = active.intersection(box(xmin - 1, ymin - 1, 0, ymax + 1))
    right_side = active.intersection(box(0, ymin - 1, xmax + 1, ymax + 1))
    bridges = []
    for bottom, top in zip(y_levels[:-1], y_levels[1:]):
        middle = bottom + (top - bottom) / 2
        line = LineString(((xmin - 1, middle), (xmax + 1, middle)))
        left_section = left_side.intersection(line)
        right_section = right_side.intersection(line)
        if left_section.length > 0 and right_section.length > 0:
            left = left_section.bounds[2]
            right = right_section.bounds[0]
            if left < right:
                bridges.append(box(left, bottom, right, top))
    joined = unary_union([active, *bridges])
    parts = joined.geoms if hasattr(joined, "geoms") else (joined,)
    # Joining staggered inner sensor edges can enclose tiny central slivers.
    # Close those, while preserving any holes already present within either side.
    envelope = unary_union([Polygon(part.exterior) for part in parts])
    if original_holes:
        envelope = envelope.difference(unary_union(original_holes))
    envelope = envelope.simplify(0)
    if envelope.is_empty or envelope.area <= 0 or not envelope.is_valid:
        raise ValueError("the MP outer borders must enclose a valid XY area")
    return envelope


def ft_with_hole_layers(layers):
    """Subtract the MP sides joined across the central x gap from every FT layer.

    Preserve detector IDs, source plane names, logical labels, z positions,
    stereo outer borders, and all non-FT geometries. Derived shape keys include
    a fingerprint of the hole so different MP choices cannot alias a pickle.
    """
    layers = list(layers)
    if not any(layer.detector == "FT" for layer in layers):
        raise ValueError("FT_withHole needs at least one FT acceptance surface")
    hole = mp_hole_envelope(layers)
    fingerprint = hashlib.sha256(hole.normalize().wkb).hexdigest()[:12]
    result = []
    for layer in layers:
        if layer.detector != "FT":
            result.append(layer)
            continue
        geometry = layer.geometry.difference(hole)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"MP envelope leaves no valid FT acceptance in {layer.name}")
        result.append(replace(
            layer,
            xyshape=f"{layer.xyshape}_FT_withHole_{fingerprint}",
            geometry=geometry,
        ))
    return result
