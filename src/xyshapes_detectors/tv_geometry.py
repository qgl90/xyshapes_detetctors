"""Sensor-resolved TV planes, grouped by physical module for hit counting."""

from collections import defaultdict
import json
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .layout import DetectorLayer


TV_SENSOR_SOURCE = Path(__file__).resolve().parents[2] / "detector_setups" / "tv_sensor_geometry.json"
TV_CONFIG = {
    "grouping": "physical_module",
    "surfaces": "coplanar_sensitive_silicon",
    "z_placement": "sensor_centres",
    "counting": "any_sensor_hit_per_module",
}


def tv_sensor_layers(source=TV_SENSOR_SOURCE):
    """Return one XY union per coplanar sensor pair, with explicit module IDs.

    Separate modules (including opposite sides) are never unioned together.
    Each module's two Z planes share a station label, so nTV counts it once.
    """
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if payload.get("schema") != "tv-sensitive-silicon-v1" or payload.get("units") != "mm":
        raise ValueError("expected a TV sensitive-silicon inventory in mm")
    modules = defaultdict(list)
    for sensor in payload["sensors"]:
        if sensor["material"] != "TV:SensorSilicon":
            raise ValueError("TV acceptance must contain sensitive silicon only")
        modules[sensor["module_id"]].append(sensor)
    if not modules:
        raise ValueError("TV sensor inventory is empty")
    ordered = sorted(modules.values(), key=lambda rows: sum(r["z_mm"] for r in rows) / len(rows))
    layers = []
    for index, sensors in enumerate(ordered, 1):
        sides = {r["side"] for r in sensors}
        if len(sensors) != 4 or len(sides) != 1 or not sides <= {"left", "right"}:
            raise ValueError("expected four sensors from one side per TV module")
        side = sides.pop()
        planes = defaultdict(list)
        for sensor in sensors:
            geometry = Polygon(sensor["points_xy"])
            if not geometry.is_valid or geometry.area <= 0:
                raise ValueError("invalid TV sensor polygon")
            planes[round(float(sensor["z_mm"]), 6)].append(geometry)
        if len(planes) != 2 or any(len(shapes) != 2 for shapes in planes.values()):
            raise ValueError("expected two pairs of coplanar sensors per TV module")
        station = f"TV_layer{index}"
        for plane, (z, shapes) in enumerate(sorted(planes.items()), 1):
            name = f"{station}_{side}_sensorplane{plane}"
            layers.append(DetectorLayer("TV", name, z, name, unary_union(shapes), station))
    return sorted(layers, key=lambda layer: layer.z)
