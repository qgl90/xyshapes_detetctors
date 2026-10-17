"""Named LHCb detector-mask configurations and reference-track evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import shapely

from .acceptance import AcceptanceModel
from .propagation import Track, propagate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPOSITORY_ROOT / "detector_setups" / "catalog.json"


def _read_catalog(path=None):
    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG
    if not catalog_path.exists():
        raise FileNotFoundError(f"configuration catalog not found: {catalog_path}")
    return catalog_path, json.loads(catalog_path.read_text(encoding="utf-8"))


def available_configurations(catalog=None):
    """Return the registered configurations, including resolved setup paths."""
    catalog_path, payload = _read_catalog(catalog)
    rows = []
    for entry in payload.get("configurations", []):
        row = dict(entry)
        row["path"] = (catalog_path.parent / entry["setup"]).resolve()
        row["available"] = row["path"].is_dir()
        rows.append(row)
    return rows


@dataclass
class LoadedMaskConfiguration:
    """A named self-contained setup with its acceptance model and inventory."""

    id: str
    label: str
    description: str
    path: Path
    config: dict
    model: AcceptanceModel

    def plane_inventory(self):
        """Return one row per stored XY surface, with its logical layer ID."""
        rows = []
        for station in self.model.stations:
            for index, (z, geometry) in enumerate(station.surfaces):
                xmin, ymin, xmax, ymax = geometry.bounds
                rows.append({
                    "detector": station.detector,
                    "plane": station.layers[index],
                    "fired_layer": station.name,
                    "z_mm": float(z),
                    "x_size_mm": float(xmax - xmin),
                    "y_size_mm": float(ymax - ymin),
                    "station_surface": index + 1,
                    "surfaces_in_layer": station.n_surfaces,
                })
        return sorted(rows, key=lambda row: (row["z_mm"], row["detector"], row["plane"]))

    def layer_inventory(self):
        return self.model.station_table()

    def evaluate_reference_track(self, track, field, **propagate_kwargs):
        """Return per-surface hits and per-detector surface/layer counts.

        Transport visits every stored surface z. A logical fired layer is true
        when any member surface contains the crossing point.
        """
        if not isinstance(track, Track):
            raise TypeError("track must be a xyshapes_detectors.Track")
        crossings = propagate([track], field, self.model.plane_z, **propagate_kwargs)
        fired_layers = self.model.fired(crossings)[0]
        column_of = {float(z): index for index, z in enumerate(self.model.plane_z)}
        plane_rows = []
        layer_index = 0

        for station in self.model.stations:
            station_hit = bool(fired_layers[layer_index])
            for surface_index, (z, geometry) in enumerate(station.surfaces):
                column = column_of[float(z)]
                crossed = bool(crossings.crossed[0, column])
                x = float(crossings.x[0, column]) if crossed else None
                y = float(crossings.y[0, column]) if crossed else None
                surface_hit = bool(shapely.contains_xy(geometry, x, y)) if crossed else False
                plane_rows.append({
                    "detector": station.detector,
                    "plane": station.layers[surface_index],
                    "fired_layer": station.name,
                    "z_mm": float(z),
                    "x_mm": x,
                    "y_mm": y,
                    "crossed": crossed,
                    "fired": surface_hit,
                    "layer_fired": station_hit,
                })
            layer_index += 1

        detector_counts = []
        for detector in self.model.detectors:
            detector_layers = [
                index for index, station in enumerate(self.model.stations)
                if station.detector == detector
            ]
            detector_surfaces = [row for row in plane_rows if row["detector"] == detector]
            detector_counts.append({
                "detector": detector,
                "layers": len(detector_layers),
                "fired_layers": sum(bool(fired_layers[index]) for index in detector_layers),
                "surfaces": len(detector_surfaces),
                "fired_surfaces": sum(row["fired"] for row in detector_surfaces),
            })

        return {
            "configuration": self.id,
            "track": {
                "charge": track.charge,
                "p_mev_c": track.p,
                "tx": track.tx,
                "ty": track.ty,
                "x0_mm": track.x,
                "y0_mm": track.y,
                "z0_mm": track.z,
            },
            "stop_reason": str(crossings.stop_reason[0]),
            "z_exit_mm": float(crossings.z_exit[0]),
            "counts": detector_counts,
            "planes": plane_rows,
        }


def load_mask_configuration(identifier, *, catalog=None, exact_surfaces=True):
    """Load a registered ID or a setup directory as a reusable mask model."""
    source = Path(identifier).expanduser()
    shape_dirs = []
    if source.exists():
        setup_path = source.resolve()
        entry = {
            "id": setup_path.name,
            "label": setup_path.name,
            "description": "Loaded directly from a setup path.",
        }
    else:
        catalog_path, payload = _read_catalog(catalog)
        entries = {item["id"]: item for item in payload.get("configurations", [])}
        if str(identifier) not in entries:
            choices = ", ".join(entries)
            raise KeyError(f"unknown detector configuration {identifier!r}; available: {choices}")
        entry = entries[str(identifier)]
        setup_path = (catalog_path.parent / entry["setup"]).resolve()
        shape_dirs = [
            (catalog_path.parent / relative_path).resolve()
            for relative_path in entry.get("shape_dirs", [])
        ]

    if not setup_path.is_dir():
        raise FileNotFoundError(f"detector setup directory not found: {setup_path}")
    config_path = setup_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if shape_dirs:
        model = AcceptanceModel.from_layout(
            setup_path / "layout.csv",
            shape_dirs=shape_dirs,
            up_dir=setup_path / "__no_up_fallback__",
            exact_surfaces=exact_surfaces,
        )
    else:
        model = AcceptanceModel.load(setup_path, exact_surfaces=exact_surfaces)
    return LoadedMaskConfiguration(
        id=entry["id"],
        label=entry["label"],
        description=entry["description"],
        path=setup_path,
        config=config,
        model=model,
    )
