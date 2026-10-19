"""Build a reusable configurable LHCb XY-surface detector setup."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
from xml.sax.saxutils import escape

from shapely.ops import unary_union

from xyshapes_detectors.create_geom_muon import muon_geometries, muon_plane_layout
from xyshapes_detectors.layout import DetectorLayer, load_layout
from xyshapes_detectors.tv_geometry import TV_CONFIG, TV_SENSOR_SOURCE, tv_sensor_layers

from create_viewlhcb_simplified_ft_layout import ft_layer_centers


ROOT = Path(__file__).resolve().parents[1]
BASE_LAYOUT = ROOT / "examples" / "viewlhcb_lower_layout_ftshrunk_picocal2_rich.csv"
BASE_SHAPE_DIRS = [
    ROOT / "ViewLHCb_acceptances" / "pickles_ft_module_shrunk",
    ROOT / "ViewLHCb_acceptances" / "pickles_picocal_two_layers",
    ROOT / "ViewLHCb_acceptances" / "pickles_rich_entry_exit",
]
UP_VARIANTS = {
    "gdml": ROOT / "ViewLHCb_acceptances" / "pickles_up_yisheng_variants" / "UP_TDRv0",
    "y8": ROOT / "ViewLHCb_acceptances" / "pickles_up_yisheng_variants" / "UP_TDRv0_Y8",
    "y8x6": ROOT / "ViewLHCb_acceptances" / "pickles_up_yisheng_variants" / "UP_TDRv0_Y8X6",
}
FT_SHAPE_DIR = ROOT / "FT_layers"
DETECTOR_ORDER = {
    "TV": 0,
    "RICH1": 1,
    "UP": 2,
    "MP": 3,
    "FT": 4,
    "RICH2": 5,
    "PicoCal": 6,
    "MUON": 7,
}
FT_LAYER_NAMES = ("X1", "U", "V", "X2")
FT_SHAPE_LAYER = {"X1": "X", "U": "U", "V": "V", "X2": "X"}
UP_FOUR_LAYER_NAMES = tuple(f"UP_layer{i}" for i in range(1, 5))


DEFAULT_CONFIGS = {
    "baseline": {
        "name": "lhcb_xy_baseline",
        "source": "ViewLHCb lower ROOT/GDML extracted surfaces",
        "include": ["TV", "RICH1", "UP", "MP", "FT", "RICH2", "PicoCal", "MUON"],
        "up": {"variant": "gdml", "plane_mode": "all16"},
        "tv": dict(TV_CONFIG),
        "ft": {"stations": {"T1": 10, "T2": 10, "T3": 12}},
    },
    "lower": {
        "name": "lhcb_xy_lower",
        "source": "ViewLHCb lower ROOT/GDML extracted surfaces",
        "include": ["TV", "RICH1", "UP", "MP", "FT", "RICH2", "PicoCal", "MUON"],
        "up": {"variant": "y8x6", "plane_mode": "all16"},
        "tv": dict(TV_CONFIG),
        "ft": {"stations": {"T1": 8, "T2": 8, "T3": 10}},
    },
}

DETECTOR_DEFAULT = list(DETECTOR_ORDER)
DETECTOR_DESCRIPTIONS = {
    "TV": "Sensitive TV silicon: separate left/right modules, 100 sensor-Z surfaces counted as 50 layers.",
    "RICH1": "Projective RICH1 radiator entry/exit surfaces from angular acceptance.",
    "UP": "Upstream Tracker surfaces; choose nominal GDML or Yisheng-inspired border crops.",
    "MP": "Five Mighty Pixel layers, each formed from the union of its extracted tiles.",
    "FT": "Mighty Tracker X,U,V,X stations; choose 8/10/12 module widths per T station.",
    "RICH2": "Projective RICH2 radiator entry/exit surfaces from angular acceptance.",
    "PicoCal": "Two front-region PicoCal surfaces, entry and exit, as a simple volume proxy.",
    "MUON": "Muon stations M2-M5, rectangular acceptances with central beam holes.",
}
UP_VARIANT_DESCRIPTIONS = {
    "gdml": "nominal ViewLHCb/GDML UP acceptance",
    "y8": "remove top/bottom border using Yisheng Y8 crop",
    "y8x6": "remove top/bottom and left/right borders using Yisheng Y8X6 crop",
}
UP_PLANE_DESCRIPTIONS = {
    "all16": "keep all 16 extracted surfaces, tagged as four fired layers",
    "four": "union each group of four surfaces into one fired layer",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row.get("Use", "1").strip() not in {"0", "false", "False"}
        ]


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Use", "detector", "layer", "station", "z", "xyshape"],
        )
        writer.writeheader()
        writer.writerows({"station": row.get("station", row["layer"]), **row} for row in rows)


def load_shape(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_shape(path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(shape, handle, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_prerequisites():
    missing = []
    for path in [BASE_LAYOUT, FT_SHAPE_DIR, TV_SENSOR_SOURCE, *BASE_SHAPE_DIRS, *UP_VARIANTS.values()]:
        if not path.exists():
            missing.append(path)
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"missing prerequisite generated geometry paths:\n{lines}")


def group_up_rows(up_rows):
    rows = sorted(up_rows, key=lambda row: float(row["z"]))
    groups = []
    current = [rows[0]]
    for row in rows[1:]:
        if float(row["z"]) - float(current[-1]["z"]) > 20.0:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
    groups.append(current)
    if len(groups) != 4:
        raise RuntimeError(f"expected four UP layer groups, found {len(groups)}")
    return groups


def group_rows_by_z(rows, tolerance):
    groups = []
    for row in sorted(rows, key=lambda item: float(item["z"])):
        if groups and float(row["z"]) - float(groups[-1][-1]["z"]) <= tolerance:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def build_union_layer_rows(detector, groups, layer_prefix, setup_shape_dir):
    rows = []
    for index, group in enumerate(groups, 1):
        shape_names = [row["xyshape"] for row in group]
        shapes = [load_shape(find_shape_file(detector, name)) for name in shape_names]
        geometry = unary_union(shapes)
        layer_name = f"{layer_prefix}_layer{index}"
        write_shape(setup_shape_dir / detector / f"{layer_name}.pickle", geometry)
        z_values = [float(row["z"]) for row in group]
        rows.append({
            "Use": "1",
            "detector": detector,
            "layer": layer_name,
            "station": layer_name,
            "z": f"{sum(z_values) / len(z_values):.6g}",
            "xyshape": layer_name,
        })
    return rows


def find_shape_file(detector, shape_name):
    matches = [
        path for directory in [ROOT / "ViewLHCb_acceptances" / "pickles", *BASE_SHAPE_DIRS]
        for path in directory.rglob(f"{shape_name}.pickle")
        if detector in path.parts
    ]
    if not matches:
        raise RuntimeError(f"could not find shape {shape_name!r} for {detector}")
    return matches[0]


def build_tv_rows(source_rows, config, setup_shape_dir):
    config["tv"] = dict(TV_CONFIG)
    rows = []
    for layer in tv_sensor_layers():
        write_shape(setup_shape_dir / "TV" / f"{layer.xyshape}.pickle", layer.geometry)
        rows.append(dict(Use="1", detector="TV", layer=layer.name,
                         station=layer.station, z=f"{layer.z:.6f}", xyshape=layer.xyshape))
    return rows


def build_mp_rows(source_rows, setup_shape_dir):
    mp_rows = [row for row in source_rows if row["detector"] == "MP"]
    groups = group_rows_by_z(mp_rows, 10.0)
    if len(groups) not in {5, 6}:
        raise RuntimeError(f"expected five or six MP layers, found {len(groups)}")
    return build_union_layer_rows("MP", groups, "MP", setup_shape_dir)


def build_up_rows(config, source_rows, shape_dir, setup_shape_dir):
    variant = config["up"]["variant"]
    plane_mode = config["up"].get("plane_mode", "all16")
    up_source_dir = UP_VARIANTS[variant] / "UP"
    up_rows = [row for row in source_rows if row["detector"] == "UP"]

    if plane_mode == "all16":
        for row in up_rows:
            source = up_source_dir / f"{row['xyshape']}.pickle"
            target = setup_shape_dir / "UP" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rows = []
        for group_index, group in enumerate(group_up_rows(up_rows), 1):
            for surface_index, row in enumerate(group, 1):
                source = up_source_dir / f"{row['xyshape']}.pickle"
                target = setup_shape_dir / "UP" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                rows.append({
                    **row,
                    "layer": f"UP_layer{group_index}_surface{surface_index}",
                    "station": f"UP_layer{group_index}",
                })
        return rows

    if plane_mode != "four":
        raise ValueError("UP plane_mode must be 'all16' or 'four'")

    rows = []
    for name, group in zip(UP_FOUR_LAYER_NAMES, group_up_rows(up_rows)):
        shapes = [load_shape(up_source_dir / f"{row['xyshape']}.pickle") for row in group]
        union = unary_union(shapes)
        shape_name = f"{name}_{variant}"
        write_shape(setup_shape_dir / "UP" / f"{shape_name}.pickle", union)
        rows.append({
            "Use": "1",
            "detector": "UP",
            "layer": name,
            "station": name,
            "z": f"{sum(float(row['z']) for row in group) / len(group):.6g}",
            "xyshape": shape_name,
        })
    return rows


def build_ft_rows(config, setup_shape_dir):
    centers = ft_layer_centers()
    rows = []
    for station in ("T1", "T2", "T3"):
        modules = int(config["ft"]["stations"][station])
        if modules not in {8, 10, 12}:
            raise ValueError(f"FT {station} module count must be one of 8, 10, or 12")
        for layer in FT_LAYER_NAMES:
            shape_layer = FT_SHAPE_LAYER[layer]
            shape_name = f"FT_{modules}Modules_{shape_layer}"
            (setup_shape_dir / "FT").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                FT_SHAPE_DIR / f"{shape_name}.pickle",
                setup_shape_dir / "FT" / f"{shape_name}.pickle",
            )
            rows.append({
                "Use": "1",
                "detector": "FT",
                "layer": f"scifi_layer{len(rows) + 1}_{station}_{layer}",
                "station": f"scifi_layer{len(rows) + 1}",
                "z": f"{centers[station][layer]:.3f}",
                "xyshape": shape_name,
            })
    return rows


def build_muon_rows(setup_shape_dir):
    rows = []
    geometries = muon_geometries()
    layout = muon_plane_layout()
    for station, (z, shape_name) in layout.items():
        write_shape(setup_shape_dir / "MUON" / f"{shape_name}.pickle", geometries[shape_name])
        rows.append({
            "Use": "1",
            "detector": "MUON",
            "layer": f"muon_{station.lower()}",
            "station": f"muon_{station.lower()}",
            "z": f"{z:.9g}",
            "xyshape": shape_name,
        })
    return rows


def copy_nonconfigurable_shapes(rows, setup_shape_dir):
    search_dirs = [ROOT / "ViewLHCb_acceptances" / "pickles", *BASE_SHAPE_DIRS]
    for row in rows:
        name = row["xyshape"]
        detector = row["detector"]
        target = setup_shape_dir / detector / f"{name}.pickle"
        if target.exists():
            continue
        matches = []
        for directory in search_dirs:
            matches.extend(directory.rglob(f"{name}.pickle"))
        if not matches:
            raise RuntimeError(f"could not find shape {name!r} for {detector}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matches[0], target)


def build_layout(config, output_dir):
    include = set(config.get("include", DETECTOR_ORDER))
    setup_shape_dir = output_dir / "shapes"
    if setup_shape_dir.exists():
        shutil.rmtree(setup_shape_dir)
    setup_shape_dir.mkdir(parents=True)

    source_rows = read_rows(BASE_LAYOUT)
    rows = [
        row for row in source_rows
        if row["detector"] in include and row["detector"] not in {"TV", "UP", "MP", "FT", "MUON"}
    ]
    for row in rows:
        if row["detector"] in {"RICH1", "RICH2", "PicoCal"}:
            face = "Entry" if row["layer"].lower().endswith("entry") else "Exit"
            detector_name = row["detector"].lower()
            row["layer"] = f"{detector_name}_layer{face}"
            row["station"] = row["layer"]
        else:
            row["station"] = row["layer"]
    rows.extend(build_tv_rows(source_rows, config, setup_shape_dir) if "TV" in include else [])
    rows.extend(build_mp_rows(source_rows, setup_shape_dir) if "MP" in include else [])
    rows.extend(build_up_rows(config, source_rows, BASE_SHAPE_DIRS[0], setup_shape_dir) if "UP" in include else [])
    rows.extend(build_ft_rows(config, setup_shape_dir) if "FT" in include else [])
    rows.extend(build_muon_rows(setup_shape_dir) if "MUON" in include else [])
    rows.sort(key=lambda row: (float(row["z"]), DETECTOR_ORDER.get(row["detector"], 99), row["layer"]))
    copy_nonconfigurable_shapes(rows, setup_shape_dir)
    return rows, setup_shape_dir


def summarize_layers(layers):
    summary = {}
    for layer in layers:
        item = summary.setdefault(layer.detector, {
            "detector": layer.detector,
            "planes": 0,
            "_stations": set(),
            "z_min": layer.z,
            "z_max": layer.z,
            "x_min": None,
            "y_min": None,
            "x_max": None,
            "y_max": None,
        })
        xmin, ymin, xmax, ymax = layer.geometry.bounds
        item["planes"] += 1
        item["_stations"].add(layer.station or layer.name)
        item["z_min"] = min(item["z_min"], layer.z)
        item["z_max"] = max(item["z_max"], layer.z)
        item["x_min"] = xmin if item["x_min"] is None else min(item["x_min"], xmin)
        item["y_min"] = ymin if item["y_min"] is None else min(item["y_min"], ymin)
        item["x_max"] = xmax if item["x_max"] is None else max(item["x_max"], xmax)
        item["y_max"] = ymax if item["y_max"] is None else max(item["y_max"], ymax)
    rows = [summary[key] for key in sorted(summary, key=lambda detector: DETECTOR_ORDER.get(detector, 99))]
    for row in rows:
        row["fired_layers"] = len(row.pop("_stations"))
    return rows


def write_summary(path, summary):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["detector", "planes", "fired_layers", "z_min", "z_max", "x_min", "y_min", "x_max", "y_max", "x_size", "y_size"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                **row,
                "x_size": row["x_max"] - row["x_min"],
                "y_size": row["y_max"] - row["y_min"],
            })


def write_detector_setup_pickle(path, config, layers, summary):
    payload = {
        "config": config,
        "summary": summary,
        "layers": layers,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def svg_text(x, y, value, *, size=14, fill="#17202a", weight="400", anchor="start"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def svg_rect(x, y, width, height, **attrs):
    attr_text = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" {attr_text}/>'


def shape_to_svg_path(shape, bounds, panel_x, panel_y, panel_w, panel_h, margin=24):
    xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        return ""
    scale = min((panel_w - 2 * margin) / (xmax - xmin), (panel_h - 2 * margin) / (ymax - ymin))

    def project(x, y):
        return (
            panel_x + margin + (x - xmin) * scale,
            panel_y + panel_h - margin - (y - ymin) * scale,
        )

    geoms = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    pieces = []
    for geom in geoms:
        if geom.is_empty or geom.geom_type != "Polygon":
            continue
        for ring in [geom.exterior, *geom.interiors]:
            coords = list(ring.coords)
            if not coords:
                continue
            first = project(*coords[0])
            pieces.append(f"M {first[0]:.2f} {first[1]:.2f}")
            for x, y in coords[1:]:
                sx, sy = project(x, y)
                pieces.append(f"L {sx:.2f} {sy:.2f}")
            pieces.append("Z")
    return " ".join(pieces)


def layer_union(layers):
    return unary_union([layer.geometry for layer in layers])


def write_svg_page(path, body, *, width=1280, height=720):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def convert_svg(path):
    outputs = []
    for suffix, fmt in (("png", "png"), ("pdf", "pdf")):
        output = path.with_suffix(f".{suffix}")
        subprocess.run(["rsvg-convert", "-f", fmt, "-o", str(output), str(path)], check=True)
        outputs.append(output)
    return outputs


def z_strip(elements, detector_layers, x, y, width, label):
    zmin = min(layer.z for layer in detector_layers)
    zmax = max(layer.z for layer in detector_layers)
    if zmax == zmin:
        zmax = zmin + 1.0
    elements.append(svg_text(x, y - 8, label, size=12, fill="#475569"))
    elements.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + width:.2f}" y2="{y:.2f}" stroke="#64748b" stroke-width="1.2"/>')
    for layer in detector_layers:
        tx = x + width * (layer.z - zmin) / (zmax - zmin)
        elements.append(f'<line x1="{tx:.2f}" y1="{y - 6:.2f}" x2="{tx:.2f}" y2="{y + 6:.2f}" stroke="#0f172a" stroke-width="0.8"/>')
    elements.append(svg_text(x, y + 24, f"{zmin:.1f} mm", size=11, fill="#475569"))
    elements.append(svg_text(x + width, y + 24, f"{zmax:.1f} mm", size=11, fill="#475569", anchor="end"))


def grouped_layers(layers):
    grouped = {}
    for layer in layers:
        grouped.setdefault(layer.detector, []).append(layer)
    return {
        detector: sorted(values, key=lambda layer: layer.z)
        for detector, values in grouped.items()
    }


def write_selected_overview(presentation_dir, config, layers, summary):
    path = presentation_dir / "slide_01_selected_setup.svg"
    body = [
        svg_text(52, 52, f"LHCb XY surface setup: {config['name']}", size=28, weight="700"),
        svg_text(52, 80, "Selected reusable detector planes for transport and acceptance masks.", size=15, fill="#475569"),
    ]
    y = 122
    body.append(svg_text(58, y, "detector", weight="700"))
    body.append(svg_text(170, y, "surfaces", weight="700"))
    body.append(svg_text(235, y, "fired layers", weight="700"))
    body.append(svg_text(340, y, "z range [mm]", weight="700"))
    body.append(svg_text(525, y, "XY envelope [mm]", weight="700"))
    body.append(svg_text(730, y, "description", weight="700"))
    y += 26
    for row in summary:
        body.append(svg_text(58, y, row["detector"], fill="#334155", weight="700"))
        body.append(svg_text(185, y, row["planes"], fill="#334155"))
        body.append(svg_text(255, y, row["fired_layers"], fill="#334155"))
        body.append(svg_text(340, y, f"{row['z_min']:.1f} to {row['z_max']:.1f}", fill="#334155"))
        body.append(svg_text(525, y, f"{row['x_max'] - row['x_min']:.1f} x {row['y_max'] - row['y_min']:.1f}", fill="#334155"))
        body.append(svg_text(730, y, DETECTOR_DESCRIPTIONS.get(row["detector"], ""), fill="#334155", size=12))
        y += 25
    body.append(svg_rect(52, 610, 1176, 58, fill="#ffffff", stroke="#cbd5e1", rx="5"))
    body.append(svg_text(72, 635, "Configuration", size=14, weight="700"))
    body.append(svg_text(72, 657, json.dumps(config, sort_keys=True), size=11, fill="#475569"))
    write_svg_page(path, body)
    convert_svg(path)
    return path


def write_detector_pages(presentation_dir, layers):
    outputs = []
    for detector, detector_layers in grouped_layers(layers).items():
        union = layer_union(detector_layers)
        bounds = union.bounds
        xmin, ymin, xmax, ymax = bounds
        path = presentation_dir / f"slide_detector_{detector.lower()}.svg"
        body = [
            svg_text(46, 46, f"{detector} XY surfaces", size=26, weight="700"),
            svg_text(
                46,
                74,
                f"{len(detector_layers)} planes, z = {min(layer.z for layer in detector_layers):.3f} to {max(layer.z for layer in detector_layers):.3f} mm, "
                f"XY envelope = {xmax - xmin:.1f} x {ymax - ymin:.1f} mm",
                size=14,
                fill="#475569",
            ),
            svg_rect(46, 104, 640, 500, fill="#ffffff", stroke="#d0d7e2"),
        ]
        path_data = shape_to_svg_path(union, bounds, 46, 104, 640, 500)
        body.append(f'<path d="{path_data}" fill="#dbeafe" fill-opacity="0.72" stroke="#2563eb" stroke-width="1.4" fill-rule="evenodd"/>')
        body.append(svg_text(58, 628, "x-y view; holes are shown by white cutouts where present", size=12, fill="#475569"))
        z_strip(body, detector_layers, 740, 148, 430, "z placement of all planes")
        body.append(svg_text(740, 220, "Layers", size=15, weight="700"))
        for index, layer in enumerate(detector_layers[:16]):
            body.append(svg_text(740, 246 + 20 * index, f"{layer.name}: z={layer.z:.3f}", size=11, fill="#334155"))
        if len(detector_layers) > 16:
            body.append(svg_text(740, 246 + 20 * 16, f"... plus {len(detector_layers) - 16} more", size=11, fill="#64748b"))
        write_svg_page(path, body)
        convert_svg(path)
        outputs.append(path)
    return outputs


def write_variant_inventory(presentation_dir):
    path = presentation_dir / "slide_variants_up_ft.svg"
    body = [
        svg_text(46, 46, "Available UP and FT XY variants", size=26, weight="700"),
        svg_text(46, 74, "Use this page to compare choices before building the final setup.", size=14, fill="#475569"),
    ]
    up_items = []
    for variant, directory in UP_VARIANTS.items():
        shapes = [load_shape(path) for path in sorted((directory / "UP").glob("*.pickle"))]
        up_items.append((variant, unary_union(shapes), UP_VARIANT_DESCRIPTIONS[variant]))
    ft_items = []
    for modules in (8, 10, 12):
        for layer in ("X", "U", "V"):
            ft_items.append((f"FT {modules} {layer}", load_shape(FT_SHAPE_DIR / f"FT_{modules}Modules_{layer}.pickle"), "module-count crop before stereo rotation"))
    all_shapes = [shape for _, shape, _ in [*up_items, *ft_items]]
    bounds = unary_union(all_shapes).bounds
    panels = [
        *[(item, 46 + i * 390, 104, 350, 220) for i, item in enumerate(up_items)],
        *[(item, 46 + (i % 3) * 390, 364 + (i // 3) * 110, 350, 94) for i, item in enumerate(ft_items)],
    ]
    for (name, shape, description), x, y, w, h in panels:
        body.append(svg_rect(x, y, w, h, fill="#ffffff", stroke="#d0d7e2"))
        body.append(svg_text(x + 12, y + 20, name, size=14, weight="700"))
        local_bounds = shape.bounds if name.startswith("UP") else bounds
        path_data = shape_to_svg_path(shape, local_bounds, x + 8, y + 28, w - 16, h - 48, margin=8)
        color = "#15803d" if name.startswith("UP") else "#d97706"
        body.append(f'<path d="{path_data}" fill="{color}" fill-opacity="0.24" stroke="{color}" stroke-width="1.2" fill-rule="evenodd"/>')
        bx = shape.bounds
        body.append(svg_text(x + 12, y + h - 12, f"{bx[2] - bx[0]:.1f} x {bx[3] - bx[1]:.1f} mm", size=11, fill="#475569"))
    write_svg_page(path, body)
    convert_svg(path)
    return path


def write_presentation(output_dir, config, layers, summary):
    presentation_dir = output_dir / "presentation"
    if presentation_dir.exists():
        shutil.rmtree(presentation_dir)
    presentation_dir.mkdir(parents=True)
    slides = [
        write_selected_overview(presentation_dir, config, layers, summary),
        *write_detector_pages(presentation_dir, layers),
        write_variant_inventory(presentation_dir),
    ]
    pdfs = [slide.with_suffix(".pdf") for slide in slides]
    combined = presentation_dir / "geometry_summary.pdf"
    subprocess.run(["pdfunite", *map(str, pdfs), str(combined)], check=True)
    return combined


def generate_3d(layout, shape_dir, output):
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run([
        "python",
        str(ROOT / "examples" / "create_viewlhcb_experiment_3d.py"),
        "--layout",
        str(layout),
        "--shape-dir",
        str(shape_dir),
        "-o",
        str(output),
    ], cwd=ROOT, env=env, check=True)


def prompt_value(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def prompt_choice(prompt, choices, default):
    choice_text = "/".join(choices)
    while True:
        value = prompt_value(f"{prompt} ({choice_text})", default)
        if value in choices:
            return value
        print(f"Please choose one of: {choice_text}")


def prompt_int_choice(prompt, choices, default):
    choices = tuple(int(choice) for choice in choices)
    while True:
        value = prompt_value(prompt, str(default))
        try:
            parsed = int(value)
        except ValueError:
            print("Please enter a number.")
            continue
        if parsed in choices:
            return parsed
        print("Please choose one of: " + ", ".join(str(choice) for choice in choices))


def prompt_yes_no(prompt, default=True):
    default_text = "y" if default else "n"
    while True:
        value = prompt_value(f"{prompt} (y/n)", default_text).lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_detector_list(default):
    print("Detector groups:", ", ".join(DETECTOR_DEFAULT))
    for detector in DETECTOR_DEFAULT:
        print(f"  {detector:7s} - {DETECTOR_DESCRIPTIONS[detector]}")
    print("Enter comma-separated names, or 'all'.")
    while True:
        value = prompt_value("Include detectors", ",".join(default))
        if value.lower() == "all":
            return DETECTOR_DEFAULT[:]
        detectors = [item.strip() for item in value.split(",") if item.strip()]
        invalid = [item for item in detectors if item not in DETECTOR_ORDER]
        if not invalid:
            return detectors
        print("Unknown detector(s): " + ", ".join(invalid))


def interactive_config(args):
    print("Interactive LHCb XY setup builder")
    preset = prompt_choice("Start from preset", sorted(DEFAULT_CONFIGS), args.preset)
    config = json.loads(json.dumps(DEFAULT_CONFIGS[preset]))
    config["name"] = prompt_value("Setup name", args.name or config["name"])
    config["include"] = prompt_detector_list(config.get("include", DETECTOR_DEFAULT))

    if "UP" in config["include"]:
        current_up = config.get("up", {})
        print("\nUP variants:")
        for name, description in UP_VARIANT_DESCRIPTIONS.items():
            print(f"  {name:5s} - {description}")
        print("UP plane modes:")
        for name, description in UP_PLANE_DESCRIPTIONS.items():
            print(f"  {name:5s} - {description}")
        config["up"] = {
            "variant": prompt_choice(
                "UP variant",
                sorted(UP_VARIANTS),
                args.up_variant or current_up.get("variant", "gdml"),
            ),
            "plane_mode": prompt_choice(
                "UP plane mode",
                ("all16", "four"),
                args.up_planes or current_up.get("plane_mode", "all16"),
            ),
        }

    if "FT" in config["include"]:
        current_ft = config.get("ft", {}).get("stations", {})
        print("\nFT station modules: choose 8, 10, or 12 modules for each station.")
        print("Each station is generated as X,U,V,X; U=-5 deg and V=+5 deg.")
        config["ft"] = {"stations": {}}
        for station, arg_value in (("T1", args.ft_t1), ("T2", args.ft_t2), ("T3", args.ft_t3)):
            config["ft"]["stations"][station] = prompt_int_choice(
                f"FT {station} modules",
                (8, 10, 12),
                arg_value or current_ft.get(station, 10),
            )

    skip_3d = not prompt_yes_no("Generate 3D HTML view", not args.skip_3d)
    print("\nSelected configuration:")
    print(json.dumps(config, indent=2, sort_keys=True))
    if not prompt_yes_no("Build this setup", True):
        raise SystemExit("Canceled.")
    return config, skip_3d


def load_config(args):
    if args.interactive:
        config, skip_3d = interactive_config(args)
        args.skip_3d = skip_3d
        return config
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    else:
        config = json.loads(json.dumps(DEFAULT_CONFIGS[args.preset]))
    config["name"] = args.name or config["name"]
    if args.up_variant:
        config.setdefault("up", {})["variant"] = args.up_variant
    if args.up_planes:
        config.setdefault("up", {})["plane_mode"] = args.up_planes
    for station, value in (("T1", args.ft_t1), ("T2", args.ft_t2), ("T3", args.ft_t3)):
        if value is not None:
            config.setdefault("ft", {}).setdefault("stations", {})[station] = value
    return config


def build_setup(config, *, output_root=None, skip_3d=False):
    """Build one setup directory and return its generated artifact paths."""
    ensure_prerequisites()
    output_root = Path(output_root) if output_root is not None else ROOT / "detector_setups"
    output_dir = output_root / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, shape_dir = build_layout(config, output_dir)
    layout_path = output_dir / "layout.csv"
    write_rows(layout_path, rows)
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    layers = load_layout(layout_path, shape_dirs=[shape_dir], up_dir=ROOT / "__nonexistent_up__")
    summary = summarize_layers(layers)
    summary_path = output_dir / "summary.csv"
    write_summary(summary_path, summary)
    setup_pickle = output_dir / "detector_setup.pickle"
    write_detector_setup_pickle(setup_pickle, config, layers, summary)
    presentation = write_presentation(output_dir, config, layers, summary)

    html = output_dir / "experiment_3d.html"
    if not skip_3d:
        generate_3d(layout_path, shape_dir, html)

    return {
        "output_dir": output_dir,
        "layout": layout_path,
        "shape_dir": shape_dir,
        "config": config_path,
        "summary": summary_path,
        "setup_pickle": setup_pickle,
        "presentation": presentation,
        "html": None if skip_3d else html,
        "layers": layers,
        "summary_rows": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(DEFAULT_CONFIGS), default="lower")
    parser.add_argument("--config", type=Path, help="JSON config to use instead of a preset")
    parser.add_argument("--name", help="setup name and output directory stem")
    parser.add_argument("--up-variant", choices=sorted(UP_VARIANTS))
    parser.add_argument("--up-planes", choices=("all16", "four"))
    parser.add_argument("--ft-t1", type=int, choices=(8, 10, 12))
    parser.add_argument("--ft-t2", type=int, choices=(8, 10, 12))
    parser.add_argument("--ft-t3", type=int, choices=(8, 10, 12))
    parser.add_argument("--output-root", type=Path, default=ROOT / "detector_setups")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true", help="prompt for setup choices")
    args = parser.parse_args()

    config = load_config(args)
    artifacts = build_setup(config, output_root=args.output_root, skip_3d=args.skip_3d)

    print(artifacts["output_dir"])
    print(artifacts["layout"])
    print(artifacts["shape_dir"])
    print(artifacts["config"])
    print(artifacts["summary"])
    print(artifacts["setup_pickle"])
    print(artifacts["presentation"])
    if artifacts["html"] is not None:
        print(artifacts["html"])
    for row in artifacts["summary_rows"]:
        print(
            f"{row['detector']:7s} surfaces={row['planes']:3d} "
            f"fired_layers={row['fired_layers']:2d} "
            f"z=[{row['z_min']:.3f},{row['z_max']:.3f}] "
            f"xy={row['x_max'] - row['x_min']:.1f}x{row['y_max'] - row['y_min']:.1f}"
        )


if __name__ == "__main__":
    main()
