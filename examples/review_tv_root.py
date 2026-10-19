"""Audit TV placements directly in the source ROOT hierarchy (requires PyROOT)."""

import argparse
from collections import Counter
import json
import re
from pathlib import Path

import ROOT

from extract_viewlhcb_root_acceptances import (
    bbox_corners, convex_hull_xy, find_volume, global_points, multiply, node_children,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT.TFile.Open(str(args.root_file))
    if not source or source.IsZombie():
        raise RuntimeError(f"cannot open {args.root_file}")
    geometry = source.GetListOfKeys().At(0).ReadObj()
    top = geometry.GetTopVolume() or geometry.GetMasterVolume()
    found = find_volume(top, "lvTV")
    if found is None:
        raise RuntimeError("lvTV is missing")
    branch, transform = found
    nodes = []

    def walk(volume, matrix, path, depth):
        children = node_children(volume)
        material = volume.GetMaterial()
        shape = volume.GetShape()
        row = dict(path=path, depth=depth, volume=volume.GetName(),
                   material=material.GetName() if material else None,
                   shape_class=shape.ClassName() if shape else None,
                   children=len(children))
        corners = bbox_corners(shape) if shape else None
        if corners:
            points = global_points(matrix, corners)
            row.update(
                centre_xyz=[sum(point[axis] for point in points) / len(points)
                            for axis in range(3)],
                bounds_xyz=[min(point[axis] for point in points) for axis in range(3)]
                           + [max(point[axis] for point in points) for axis in range(3)],
                local_half_sizes=[shape.GetDX(), shape.GetDY(), shape.GetDZ()],
            )
            if row["material"] == "TV:SensorSilicon":
                if shape.ClassName() != "TGeoBBox":
                    raise RuntimeError(f"non-box silicon needs exact extraction: {path}")
                row["points_xy"] = convex_hull_xy(points)
                match = re.search(r"/pvModule(\d+)WithSupport(?:/|$)", path)
                if not match:
                    raise RuntimeError(f"cannot identify sensor module: {path}")
                row["module_id"] = int(match.group(1))
        nodes.append(row)
        for child in children:
            walk(child.GetVolume(), multiply(matrix, child.GetMatrix()),
                 path + "/" + child.GetName(), depth + 1)

    walk(branch, transform, "lvTV", 0)
    sensors = [row for row in nodes if row["material"] == "TV:SensorSilicon"]
    if not sensors:
        raise RuntimeError("no sensitive silicon was found in lvTV")
    counts = dict(Counter(row["volume"] for row in nodes))
    summary = {
        "source": str(args.root_file), "root_version": ROOT.gROOT.GetVersion(),
        "node_count": len(nodes), "volume_counts": counts,
        "selected_sensor_count": len(sensors),
        "selected_sensor_z_range": [min(row["centre_xyz"][2] for row in sensors),
                                    max(row["centre_xyz"][2] for row in sensors)],
        "selection": "material == TV:SensorSilicon within lvTV; no supports, chips or guard rings",
        "nodes": [row for row in nodes if row["depth"] <= 2
                  or row["volume"] == "lvModule"
                  or row["material"] == "TV:SensorSilicon"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"nodes", "volume_counts"}}, indent=2))


if __name__ == "__main__":
    main()
