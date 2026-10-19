"""Extract mask surfaces from a ViewLHCb ROOT TGeo geometry.

Run with a clean ROOT environment when multiple ROOT installs exist, e.g.::

    env -i PATH=/Users/renato/miniconda3/envs/pyCombiner/bin:/usr/bin:/bin \
      HOME=/private/tmp TMPDIR=/private/tmp \
      /Users/renato/miniconda3/envs/pyCombiner/bin/python \
      examples/extract_viewlhcb_root_acceptances.py \
      View_LHCb_run5/LHCb_lower-scenario-Sep2026.root \
      -o ViewLHCb_acceptances/lower_acceptances.json
"""

from __future__ import annotations

import argparse
from array import array
import json
import re
from pathlib import Path

import ROOT


DETECTORS = {
    "TV": {
        "root": "lvTV",
        "leaf": re.compile(r"^lvDet$"),
        "group": "z",
    },
    "MP": {
        "root": "lvMP",
        "leaf": re.compile(r"^lvSensor$"),
        "group": "z",
    },
    "FT": {
        "root": "lvFT",
        "leaf": re.compile(r"^lvFTFibre(?:Short)?Mat"),
        "group": "z",
    },
    "UP": {
        "root": "lvUP",
        "leaf": re.compile(r"^lvTDRv0ActiveSensor$"),
        "group": "z",
    },
    "PicoCal": {
        "root": "lvPicoCal",
        # The front-cover shapes are the nearest robust front-face proxy in the
        # current geometry and avoid expanding the calorimeter's deep fibre tree.
        "leaf": re.compile(r"^lvPicoCalFrontCover_"),
        "group": "single",
    },
}


def node_children(volume):
    nodes = volume.GetNodes()
    return [nodes.At(index) for index in range(nodes.GetEntries())] if nodes else []


def find_volume(volume, name, matrix=None):
    if matrix is None:
        matrix = ROOT.TGeoHMatrix()
    if volume.GetName() == name:
        return volume, matrix
    for node in node_children(volume):
        found = find_volume(node.GetVolume(), name, multiply(matrix, node.GetMatrix()))
        if found:
            return found
    return None


def multiply(parent, local):
    matrix = ROOT.TGeoHMatrix(parent)
    matrix.Multiply(local)
    return matrix


def bbox_corners(shape):
    if not hasattr(shape, "GetDX"):
        return None
    dx, dy, dz = shape.GetDX(), shape.GetDY(), shape.GetDZ()
    return [
        (-dx, -dy, -dz),
        (dx, -dy, -dz),
        (dx, dy, -dz),
        (-dx, dy, -dz),
        (-dx, -dy, dz),
        (dx, -dy, dz),
        (dx, dy, dz),
        (-dx, dy, dz),
    ]


def global_points(matrix, corners):
    points = []
    for corner in corners:
        local = array("d", corner)
        master = array("d", [0.0, 0.0, 0.0])
        matrix.LocalToMaster(local, master)
        points.append(tuple(master))
    return points


def convex_hull_xy(points):
    """Return the convex hull of projected XY points as ordered vertices."""
    xy_points = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(xy_points) <= 1:
        return xy_points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in xy_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(xy_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def record_from_bbox(detector, layer, z, points, source_path, volume_name):
    return {
        "detector": detector,
        "layer": layer,
        "z": z,
        "points": convex_hull_xy(points),
        "source": str(source_path),
        "volume": volume_name,
    }


def extract_detector(root_volume, detector, config, source_path, z_round=3):
    found = find_volume(root_volume, config["root"])
    if found is None:
        raise RuntimeError(f"could not find {config['root']}")
    branch, branch_matrix = found

    raw = []

    def walk(volume, matrix, depth):
        name = volume.GetName()
        shape = volume.GetShape()
        if config["leaf"].search(name) and shape:
            corners = bbox_corners(shape)
            if corners:
                points = global_points(matrix, corners)
                z_values = [point[2] for point in points]
                z_center = sum(z_values) / len(z_values)
                raw.append((name, z_center, points))
                if config["group"] == "single":
                    return
        # PicoCal is intentionally shallow: the front face is enough for masks.
        if detector == "PicoCal" and depth >= 4:
            return
        for node in node_children(volume):
            walk(node.GetVolume(), multiply(matrix, node.GetMatrix()), depth + 1)

    walk(branch, branch_matrix, 0)

    records = []
    for index, (name, z_center, points) in enumerate(raw):
        if config["group"] == "single":
            layer = f"{detector}_front"
            z = min(point[2] for point in points)
        else:
            z = round(z_center, z_round)
            layer = f"{detector}_{z:.{z_round}f}".replace("-", "m").replace(".", "p")
        records.append(record_from_bbox(detector, layer, z, points, source_path, name))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=list(DETECTORS),
        choices=list(DETECTORS),
        help="detectors to extract",
    )
    args = parser.parse_args()

    root_file = ROOT.TFile.Open(str(args.root_file))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"could not open {args.root_file}")
    key = root_file.GetListOfKeys().At(0)
    geometry = key.ReadObj()
    root_volume = geometry.GetTopVolume() or geometry.GetMasterVolume()

    surfaces = []
    summary = {}
    for detector in args.detectors:
        records = extract_detector(root_volume, detector, DETECTORS[detector], args.root_file)
        surfaces.extend(records)
        summary[detector] = len(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source": str(args.root_file),
                "detectors": args.detectors,
                "summary": summary,
                "surfaces": surfaces,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
