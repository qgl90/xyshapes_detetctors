"""Convert ViewLHCb exported placements into XY acceptance shapes.

The ViewLHCb geometry source is usually a ROOT TGeoManager or GDML tree.  This
module deliberately starts from a small JSON interchange format so that the
ROOT/GDML extraction step can run in whichever environment has a working
geometry reader, while the repository keeps using ordinary Shapely pickles.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import pickle
from typing import Iterable

from shapely.geometry import Polygon, box
from shapely.ops import unary_union


@dataclass(frozen=True)
class AcceptanceSurface:
    """One detector acceptance surface placed at one z coordinate."""

    detector: str
    layer: str
    z: float
    shape: object
    source: str = ""


def _shape_from_record(record):
    if "points" in record:
        points = [(float(x), float(y)) for x, y in record["points"]]
        return Polygon(points)
    if "bounds" in record:
        xmin, ymin, xmax, ymax = [float(value) for value in record["bounds"]]
        return box(xmin, ymin, xmax, ymax)
    raise ValueError("each acceptance record needs either 'points' or 'bounds'")


def load_acceptance_json(path) -> list[AcceptanceSurface]:
    """Load acceptance surfaces from a JSON export.

    Expected schema::

        {
          "surfaces": [
            {
              "detector": "TV",
              "layer": "TV_000",
              "z": 12.3,
              "points": [[x0, y0], [x1, y1], ...]
            }
          ]
        }

    ``bounds: [xmin, ymin, xmax, ymax]`` may be used instead of ``points`` for
    axis-aligned boxes.  Multiple records with the same detector/layer/z are
    merged into a single Shapely union.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["surfaces"] if isinstance(payload, dict) else payload

    grouped = {}
    for record in records:
        detector = str(record["detector"])
        layer = str(record["layer"])
        z = float(record["z"])
        key = detector, layer, z
        grouped.setdefault(key, []).append(_shape_from_record(record))

    surfaces = []
    for (detector, layer, z), shapes in sorted(grouped.items()):
        shape = unary_union(shapes)
        if shape.is_empty or not shape.is_valid:
            raise ValueError(f"invalid extracted shape for {detector}/{layer} at z={z}")
        surfaces.append(
            AcceptanceSurface(
                detector=detector,
                layer=layer,
                z=z,
                shape=shape,
                source=str(path),
            )
        )
    return surfaces


def write_acceptance_outputs(
    surfaces: Iterable[AcceptanceSurface],
    *,
    output_dir,
    layout_path,
    detector_prefixes=None,
):
    """Write Shapely pickles and a layout CSV for exported acceptances."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_path = Path(layout_path)
    layout_path.parent.mkdir(parents=True, exist_ok=True)

    detector_prefixes = detector_prefixes or {}
    rows = []
    used_shape_names = {}
    for surface in surfaces:
        prefix = detector_prefixes.get(surface.detector, surface.detector)
        z_tag = f"{surface.z:.6f}".replace("-", "m").replace(".", "p")
        base_shape_name = f"{prefix}_{surface.layer}_z{z_tag}"
        count = used_shape_names.get(base_shape_name, 0)
        used_shape_names[base_shape_name] = count + 1
        shape_name = base_shape_name if count == 0 else f"{base_shape_name}_{count + 1}"
        pickle_path = output_dir / surface.detector / f"{shape_name}.pickle"
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with pickle_path.open("wb") as handle:
            pickle.dump(surface.shape, handle, protocol=pickle.HIGHEST_PROTOCOL)
        rows.append(
            {
                "Use": "1",
                "detector": surface.detector,
                "layer": surface.layer,
                "z": f"{surface.z:.9g}",
                "xyshape": shape_name,
            }
        )

    with layout_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Use", "detector", "layer", "z", "xyshape"])
        writer.writeheader()
        writer.writerows(rows)

    return rows
