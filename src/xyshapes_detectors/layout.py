"""Load detector-layer layouts and evaluate simple acceptance masks."""

from dataclasses import dataclass
import csv
from pathlib import Path

from shapely.geometry import Point

from .create_geom_ft import ft_box_geometries
from .create_geom_muon import muon_geometries
from .create_geom_tv import tv_geometries
from .geometry_io import load_geometry


@dataclass(frozen=True)
class DetectorLayer:
    detector: str
    name: str
    z: float
    xyshape: str
    geometry: object
    station: str | None = None

    def contains(self, x, y):
        return self.geometry.contains(Point(x, y))


def default_shape_registry(up_dir="UP", shape_dirs=None):
    """Return the built-in Run 5 shape registry."""
    registry = {}
    registry.update(tv_geometries())
    registry.update(ft_box_geometries())
    registry.update(muon_geometries())

    up_path = Path(up_dir)
    if up_path.is_dir():
        for path in sorted(up_path.glob("UP*.pickle")):
            registry[path.stem] = load_geometry(path)
    for shape_dir in shape_dirs or []:
        for path in sorted(Path(shape_dir).rglob("*.pickle")):
            registry[path.stem] = load_geometry(path)
    return registry


def load_layout(path, *, shape_registry=None, up_dir="UP", shape_dirs=None):
    """Load a CSV layout into placed detector layers.

    The CSV columns include ``Use,detector,layer,station,z,xyshape``. Rows with ``Use=0``
    are skipped. ``shape_registry`` maps ``xyshape`` names to Shapely
    geometries; when omitted, TV, MT, Muon, and visible UP pickles are loaded.
    Additional pickle directories from ``shape_dirs`` are scanned recursively,
    which is useful for ViewLHCb-derived acceptances.
    """
    registry = shape_registry if shape_registry is not None else default_shape_registry(
        up_dir, shape_dirs=shape_dirs)
    layers = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Use", "1").strip() in {"0", "false", "False"}:
                continue
            xyshape = row["xyshape"]
            if xyshape not in registry:
                raise KeyError(f"unknown xyshape {xyshape!r}")
            layers.append(DetectorLayer(
                detector=row["detector"],
                name=row["layer"],
                z=float(row["z"]),
                xyshape=xyshape,
                geometry=registry[xyshape],
                station=row.get("station") or None,
            ))
    return sorted(layers, key=lambda layer: layer.z)


def layers_by_detector(layers):
    grouped = {}
    for layer in layers:
        grouped.setdefault(layer.detector, []).append(layer)
    return grouped


def detector_masks(layers, track_xy_at_z, detectors=("TV", "UP", "MT")):
    """Return has<detector>_Run5 style masks.

    ``track_xy_at_z`` is a callable receiving a layer ``z`` in millimetres and
    returning ``(x, y)`` in millimetres at that z. ``detectors`` controls which
    detector groups are evaluated; pass ``None`` to evaluate every detector in
    the loaded layout.
    """
    grouped = layers_by_detector(layers)
    masks = {}
    for detector in sorted(grouped) if detectors is None else detectors:
        detector_layers = grouped.get(detector, [])
        masks[f"has{detector}_Run5"] = bool(detector_layers) and all(
            layer.contains(*track_xy_at_z(layer.z))
            for layer in detector_layers
        )
    return masks
