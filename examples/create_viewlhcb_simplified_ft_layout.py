"""Create ViewLHCb-derived layouts with simplified FT X,U,V,X layers."""

from __future__ import annotations

import csv
from pathlib import Path

from xyshapes_detectors.layout import layers_by_detector, load_layout


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LAYOUT = ROOT / "examples" / "viewlhcb_lower_layout.csv"
SOURCE_PICKLES = ROOT / "ViewLHCb_acceptances" / "pickles"
OUTPUT_BASELINE = ROOT / "examples" / "viewlhcb_simplified_ft_baseline_layout.csv"
OUTPUT_LOW = ROOT / "examples" / "viewlhcb_simplified_ft_low_layout.csv"

LAYER_NAMES = ("X1", "U", "V", "X2")
SHAPE_LAYER = {"X1": "X", "U": "U", "V": "V", "X2": "X"}


def ft_layer_centers():
    """Derive station/layer z centers from the extracted FT ROOT masks."""
    ft_layers = layers_by_detector(
        load_layout(SOURCE_LAYOUT, shape_dirs=[SOURCE_PICKLES], up_dir=ROOT / "UP")
    )["FT"]
    zs = sorted({round(layer.z, 3) for layer in ft_layers})

    stations = []
    current = [zs[0]]
    for z in zs[1:]:
        if z - current[-1] > 100.0:
            stations.append(current)
            current = [z]
        else:
            current.append(z)
    stations.append(current)

    result = {}
    for station_index, station_zs in enumerate(stations, start=1):
        groups = []
        current = [station_zs[0]]
        for z in station_zs[1:]:
            if z - current[-1] > 20.0:
                groups.append(current)
                current = [z]
            else:
                current.append(z)
        groups.append(current)
        if len(groups) != 4:
            raise ValueError(f"expected 4 FT layer groups in T{station_index}, found {len(groups)}")
        station = f"T{station_index}"
        result[station] = {
            layer: 0.5 * (min(group) + max(group))
            for layer, group in zip(LAYER_NAMES, groups)
        }
    return result


def non_ft_rows():
    with SOURCE_LAYOUT.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row.get("Use", "1").strip() not in {"0", "false", "False"}
            and row["detector"] != "FT"
        ]


def write_layout(path, station_modules, centers):
    rows = non_ft_rows()
    for station in ("T1", "T2", "T3"):
        modules = station_modules[station]
        for layer in LAYER_NAMES:
            rows.append({
                "Use": "1",
                "detector": "FT",
                "layer": f"{station}_{layer}",
                "z": f"{centers[station][layer]:.3f}",
                "xyshape": f"FT_{modules}Modules_{SHAPE_LAYER[layer]}",
            })
    rows.sort(key=lambda row: float(row["z"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Use", "detector", "layer", "z", "xyshape"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    centers = ft_layer_centers()
    print(write_layout(OUTPUT_BASELINE, {"T1": 10, "T2": 10, "T3": 12}, centers))
    print(write_layout(OUTPUT_LOW, {"T1": 8, "T2": 8, "T3": 10}, centers))
    for station, layers in centers.items():
        print(station, ", ".join(f"{layer}={z:.3f}" for layer, z in layers.items()))


if __name__ == "__main__":
    main()
