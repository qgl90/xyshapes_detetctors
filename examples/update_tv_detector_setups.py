"""Update saved detector setups to the canonical sensor-resolved TV geometry."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

from build_lhcb_xy_setup import (
    read_rows, summarize_layers, write_detector_setup_pickle, write_rows,
    write_shape, write_summary,
)
from xyshapes_detectors.configuration import load_mask_configuration
from xyshapes_detectors.layout import load_layout
from xyshapes_detectors.tv_geometry import TV_CONFIG, TV_SENSOR_SOURCE, tv_sensor_layers


ROOT = Path(__file__).resolve().parents[1]


def import_inventory(inventory, destination):
    source = json.loads(Path(inventory).read_text())
    sensors = []
    for row in source["nodes"]:
        if row.get("material") != "TV:SensorSilicon":
            continue
        sensors.append(dict(module_id=row["module_id"],
                            side="left" if "/pvTVLeft/" in row["path"] else "right",
                            z_mm=row["centre_xyz"][2], points_xy=row["points_xy"],
                            material=row["material"], path=row["path"]))
    if len(sensors) != 200:
        raise ValueError("expected 200 audited TV sensitive sensors")
    payload = dict(schema="tv-sensitive-silicon-v1", units="mm", source=source["source"],
                   selection="TV:SensorSilicon within lvTV", sensors=sensors)
    destination.write_text(json.dumps(payload, indent=2)+"\n")


def update_setups(root, *, refresh_views=False):
    root = Path(root).resolve()
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    registered = {(root / entry["setup"]).resolve(): entry for entry in catalog["configurations"]}
    layers = tv_sensor_layers()
    tv_rows = [dict(Use="1", detector="TV", layer=r.name, station=r.station,
                    z=f"{r.z:.6f}", xyshape=r.xyshape) for r in layers]
    for layer in layers:
        write_shape(root / "shared_shapes" / "TV" / f"{layer.xyshape}.pickle", layer.geometry)
    updated = []
    for layout in sorted(root.rglob("layout.csv")):
        rows = read_rows(layout)
        if not any(row["detector"] == "TV" for row in rows):
            continue
        directory = layout.parent
        before = [row for row in rows if row["detector"] != "TV"]
        rows = sorted([*before, *tv_rows], key=lambda row: (float(row["z"]), row["detector"], row["layer"]))
        write_rows(layout, rows)
        after = [row for row in read_rows(layout) if row["detector"] != "TV"]
        normalize = lambda data: sorted(data, key=lambda r: (r["detector"], r["layer"], float(r["z"])))
        # CSV writer fills a missing station with its layer name.
        expected = [{"station": row.get("station", row["layer"]), **row} for row in before]
        if normalize(expected) != normalize(after):
            raise AssertionError(f"non-TV rows changed: {layout}")
        for layer in layers:
            write_shape(directory / "shapes" / "TV" / f"{layer.xyshape}.pickle", layer.geometry)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {"name": directory.name}
        config["tv"] = dict(TV_CONFIG)
        entry = registered.get(directory.resolve())
        if entry and entry.get("shape_dirs"):
            config["shape_dirs"] = [os.path.relpath(root / relative, directory)
                                    for relative in entry["shape_dirs"]]
        config_path.write_text(json.dumps(config, indent=2)+"\n")
        loaded = load_mask_configuration(directory)
        shape_dirs = [directory / path for path in config.get("shape_dirs", [])] or [directory / "shapes"]
        # Catalog base entries may use shared shapes, while direct-path loading
        # uses their self-contained local copies. Validate both after migration.
        physical = load_layout(layout, shape_dirs=shape_dirs, up_dir=directory / "__no_up_fallback__")
        summary = summarize_layers(physical)
        write_summary(directory / "summary.csv", summary)
        write_detector_setup_pickle(directory / "detector_setup.pickle", config, physical, summary)
        stations = loaded.model.stations_of("TV")
        if len(stations) != 50 or sum(s.n_surfaces for s in stations) != 100:
            raise AssertionError(f"TV grouping was not preserved in {directory}")
        if refresh_views:
            html = directory / "experiment_3d.html"
            if html.exists():
                cmd = [sys.executable, str(ROOT / "examples" / "create_viewlhcb_experiment_3d.py"),
                       "--layout", str(layout), "-o", str(html)]
                for shape_dir in shape_dirs:
                    cmd.extend(["--shape-dir", str(shape_dir)])
                env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
                subprocess.run(cmd, check=True, env=env, capture_output=True)
                if html.with_suffix(".html.gz").exists():
                    html.with_suffix(".html.gz").write_bytes(gzip.compress(html.read_bytes(), mtime=0))
        updated.append(str(directory.relative_to(root)))
        print(f"Updated {directory.name}: 100 sensor-Z surfaces / 50 logical modules", flush=True)
    catalog["available_choices"]["TV"]["layer_definition"] = (
        "Sensitive silicon at 100 sensor-centre Z planes; separate left/right modules, "
        "50 logical layers, one hit per physical module."
    )
    catalog_path.write_text(json.dumps(catalog, indent=2)+"\n")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "detector_setups")
    parser.add_argument("--import-inventory", type=Path)
    parser.add_argument("--refresh-views", action="store_true")
    args = parser.parse_args()
    if args.import_inventory:
        import_inventory(args.import_inventory, TV_SENSOR_SOURCE)
    updated = update_setups(args.root, refresh_views=args.refresh_views)
    print(f"Updated {len(updated)} saved setups")


if __name__ == "__main__":
    main()
