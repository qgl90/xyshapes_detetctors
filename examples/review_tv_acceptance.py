"""Build sensor-resolved TV masks and review plots from the ROOT silicon audit."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from xyshapes_detectors.configuration import load_mask_configuration
from xyshapes_detectors.acceptance import AcceptanceModel


def draw(ax, geometry, color, alpha=0.25):
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for part in parts:
        ax.add_patch(Patch(np.asarray(part.exterior.coords), facecolor=color,
                           edgecolor=color, alpha=alpha))
        for ring in part.interiors:
            ax.add_patch(Patch(np.asarray(ring.coords), facecolor="white",
                               edgecolor=color))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path,
                        default=Path("docs/TV/review/root_tv_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/TV/review"))
    args = parser.parse_args()
    source = json.loads(args.inventory.read_text())
    sensors = [r for r in source["nodes"] if r.get("points_xy")]
    if len(sensors) != 200:
        raise ValueError(f"expected 200 sensitive sensors, found {len(sensors)}")
    modules = {}
    for sensor in sensors:
        modules.setdefault(sensor["module_id"], []).append(sensor)
    modules = sorted(modules.items(), key=lambda item: np.mean(
        [r["centre_xyz"][2] for r in item[1]]))
    cfg = load_mask_configuration("lhcb_lower_up_y8x6_ft_8_8_10")
    stored = cfg.model.stations_of("TV")
    if len(stored) != len(modules):
        raise ValueError("stored TV module count differs from the ROOT hierarchy")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, exact_layers, unions, layout = [], [], [], []
    shape_dir = args.output / "sensor_setup" / "shapes" / "TV"
    shape_dir.mkdir(parents=True, exist_ok=True)
    for i, ((module_id, members), existing) in enumerate(zip(modules, stored), 1):
        if len(members) != 4:
            raise ValueError(f"module {module_id} does not contain four sensors")
        surfaces = [(float(r["centre_xyz"][2]), Polygon(r["points_xy"]))
                    for r in members]
        union = unary_union([shape for _, shape in surfaces])
        z = float(np.mean([z for z, _ in surfaces]))
        difference = union.symmetric_difference(existing.geometry).area
        if difference > 1e-7 or abs(z - existing.z) > 1e-6:
            raise ValueError(f"stored layer {i} differs from ROOT silicon")
        unions.append(union)
        rows.append(dict(layer=f"TV_layer{i}", module_id=module_id,
                         side="left" if module_id % 2 else "right",
                         z_mean_mm=z, z_min_mm=min(z for z, _ in surfaces),
                         z_max_mm=max(z for z, _ in surfaces), sensors=len(members),
                         area_mm2=union.area, stored_difference_mm2=difference))
        exact_layers.append(dict(name=f"TV_layer{i}", detector="TV",
                                 module_id=module_id, z_mean_mm=z,
                                 surfaces=surfaces,
                                 sensor_paths=[r["path"] for r in members]))
        # Union only coplanar sensors; retain both physical sensor-centre Zs.
        planes = {}
        for sensor_z, shape in surfaces:
            planes.setdefault(round(sensor_z, 6), []).append(shape)
        for j, (sensor_z, shapes) in enumerate(sorted(planes.items()), 1):
            name = f"TV_layer{i}_sensorplane{j}"
            with (shape_dir / f"{name}.pickle").open("wb") as stream:
                pickle.dump(unary_union(shapes), stream)
            layout.append(dict(Use=1, detector="TV", layer=name,
                               station=f"TV_layer{i}", z=sensor_z, xyshape=name))
    layout_path = args.output / "sensor_setup" / "layout.csv"
    with layout_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layout[0]))
        writer.writeheader()
        writer.writerows(layout)
    reloaded = AcceptanceModel.from_setup(layout_path.parent, exact_surfaces=True)
    if len(reloaded.stations_of("TV")) != 50 or len(reloaded.plane_z) != 100:
        raise AssertionError("sensor setup did not preserve 50 modules / 100 Z planes")
    with (args.output / "tv_sensor_masks.pickle").open("wb") as stream:
        pickle.dump(dict(schema="tv-silicon-surfaces-v1", units="mm",
                         source=source["source"],
                         counting="one hit per module if any member sensor is hit",
                         layers=exact_layers), stream)
    with (args.output / "tv_layers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    colors = ["#167ca6", "#b34f70", "#478c48", "#bc8418"]
    for ax, index in zip(axes[:2], [0, 1]):
        for j, (z, shape) in enumerate(exact_layers[index]["surfaces"]):
            draw(ax, shape, colors[j], 0.35)
            p = shape.centroid
            ax.text(p.x, p.y, f"s{j+1}\nz={z:.3f}", ha="center", va="center", fontsize=8)
        bounds = unions[index].bounds
        ax.set_title(f"TV_layer{index+1}: module {rows[index]['module_id']}\n"
                     f"XY envelope {bounds[2]-bounds[0]:.2f} x {bounds[3]-bounds[1]:.2f} mm")
    draw(axes[2], unions[0], colors[0])
    draw(axes[2], unions[1], colors[1])
    axes[2].set_title("Opposite modules overlaid in XY\nDifferent Z: not a single physical plane")
    for ax in axes:
        ax.set(xlim=(-52, 52), ylim=(-52, 52), xlabel="x [mm]", ylabel="y [mm]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle("TV sensitive silicon only: four 38.73 x 12.8 x 0.23 mm sensors per module")
    fig.savefig(args.output / "tv_sensor_xy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    for side, color in [("left", colors[0]), ("right", colors[1])]:
        selected = [r for r in rows if r["side"] == side]
        y = 1 if side == "left" else 0
        ax.scatter([r["z_mean_mm"] for r in selected], [y]*len(selected),
                   color=color, label=f"{side}: 25 modules", zorder=3)
        for r in selected:
            ax.text(r["z_mean_mm"], y+0.08, r["layer"].replace("TV_layer", ""),
                    fontsize=6, ha="center", rotation=90)
    ax.axvline(0, color="black", linestyle="--", label="origin z=0")
    ax.axvspan(253.85, 440.15, color="#ddc48b", alpha=0.3, label="186.3 mm module-centre gap")
    ax.set(xlabel="global z [mm]", yticks=[0, 1], yticklabels=["right", "left"],
           ylim=(-0.3, 1.6), xlim=(-310, 800),
           title="TV positions from the source hierarchy; sensor centres end at 558.146 mm")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(args.output / "tv_sensor_z.png", dpi=180)
    plt.close(fig)
    summary = dict(sensors=len(sensors), modules=len(rows),
                   centre_z_range_mm=source["selected_sensor_z_range"],
                   upstream_modules_at_origin=sum(r["z_mean_mm"] < 0 for r in rows),
                   max_stored_xy_difference_mm2=max(r["stored_difference_mm2"] for r in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
