"""Materialize FT_withHole setups with square MP borders and filled inner x gaps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle

from xyshapes_detectors.configuration import DEFAULT_CATALOG, load_mask_configuration
from xyshapes_detectors.layout import load_layout
from xyshapes_detectors.variants import ft_with_hole_layers, mp_hole_envelope


def build_variants(catalog=DEFAULT_CATALOG):
    catalog = Path(catalog).resolve()
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    root = catalog.parent
    shape_dir = root / "ft_with_hole_shapes"
    shape_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        entry for entry in payload["configurations"]
        if entry.get("ft_acceptance") != "FT_withHole"
        and not entry["id"].endswith("_FT_withHole")
    ]
    variants = []
    used_shape_keys = set()
    for entry in sources:
        source = load_mask_configuration(entry["id"], catalog=catalog)
        source_dirs = [
            (root / relative).resolve() for relative in entry.get("shape_dirs", [])
        ] or [source.path / "shapes"]
        layers = load_layout(
            source.path / "layout.csv", shape_dirs=source_dirs,
            up_dir=source.path / "__no_up_fallback__",
        )
        hole = mp_hole_envelope(layers)
        derived = ft_with_hole_layers(layers)
        identifier = f"{entry['id']}_FT_withHole"
        destination = root / identifier
        destination.mkdir(parents=True, exist_ok=True)
        dirs = [*source_dirs, shape_dir]
        config = dict(source.config)
        config.update({
            "name": identifier,
            "source_configuration": entry["id"],
            "ft_acceptance": "FT_withHole",
            "shape_dirs": [os.path.relpath(path, destination) for path in dirs],
            "ft_hole": {
                "operation": "FT.difference(join_MP_left_right_at_x0(union(all MP XY surfaces)))",
                "outer_border": "original axis-aligned MP staircase; no convex interpolation",
                "frame": "global_xy_mm; no z projection",
                "fills_central_x_gap": True,
                "bounds_mm": list(hole.bounds),
                "area_mm2": hole.area,
                "vertices_mm": [list(point) for point in hole.exterior.coords],
                "interior_rings_mm": [list(ring.coords) for ring in hole.interiors],
            },
        })
        (destination / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        with (destination / "layout.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("Use", "detector", "layer", "station", "z", "xyshape")
            )
            writer.writeheader()
            for layer in derived:
                writer.writerow({
                    "Use": 1, "detector": layer.detector, "layer": layer.name,
                    "station": layer.station or layer.name, "z": f"{layer.z:.6f}",
                    "xyshape": layer.xyshape,
                })
                if layer.detector == "FT":
                    used_shape_keys.add(layer.xyshape)
                    with (shape_dir / f"{layer.xyshape}.pickle").open("wb") as output:
                        pickle.dump(layer.geometry, output, protocol=pickle.HIGHEST_PROTOCOL)
        summary = []
        for before, after in zip(layers, derived):
            if before.detector == "FT":
                summary.append({
                    "plane": before.name, "z_mm": before.z,
                    "source_area_mm2": before.geometry.area,
                    "kept_area_mm2": after.geometry.area,
                    "removed_area_mm2": before.geometry.area - after.geometry.area,
                    "removed_percent": 100 * (1 - after.geometry.area / before.geometry.area),
                })
        (destination / "ft_area_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        variants.append({
            "id": identifier, "label": entry["label"] + " / FT_withHole",
            "description": "FT minus the original MP sides joined across their central x gap.",
            "setup": os.path.relpath(destination, root),
            "shape_dirs": [os.path.relpath(path, root) for path in dirs],
            "family": entry.get("family", "detector setups"),
            "source_configuration": entry["id"], "ft_acceptance": "FT_withHole",
        })
    variant_ids = {entry["id"] for entry in variants}
    payload["configurations"] = [
        entry for entry in payload["configurations"] if entry["id"] not in variant_ids
    ] + variants
    payload.setdefault("available_choices", {}).setdefault("FT", {})[
        "acceptance_variants"
    ] = ["nominal", "FT_withHole"]
    catalog.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for path in shape_dir.glob("*_FT_withHole_*.pickle"):
        if path.stem not in used_shape_keys:
            path.unlink()
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    for entry in build_variants(args.catalog):
        print(entry["id"])


if __name__ == "__main__":
    main()
