"""Create a 3D view of ViewLHCb-derived masks plus Muon stations."""

from __future__ import annotations

import json
from pathlib import Path
import argparse

from xyshapes_detectors.create_geom_muon import muon_geometries, muon_plane_layout
from xyshapes_detectors.layout import DetectorLayer, load_layout


ROOT = Path(__file__).resolve().parents[1]
PICKLES = ROOT / "ViewLHCb_acceptances" / "pickles"

COLORS = {
    "TV": "#2864b4",
    "UP": "#2b8a3e",
    "MP": "#7c3aed",
    "FT": "#d97706",
    "PicoCal": "#c026d3",
    "RICH1": "#0891b2",
    "RICH2": "#0f766e",
    "MUON": "#64748b",
}


def polygon_parts(shape):
    geoms = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    parts = []
    for polygon in geoms:
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        parts.append({
            "outer": [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]],
            "holes": [
                [[float(x), float(y)] for x, y in ring.coords[:-1]]
                for ring in polygon.interiors
            ],
        })
    return parts


def layer_payload(layer):
    xmin, ymin, xmax, ymax = layer.geometry.bounds
    return {
        "detector": layer.detector,
        "name": layer.name,
        "z": float(layer.z),
        "bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "parts": polygon_parts(layer.geometry),
        "color": COLORS.get(layer.detector, "#64748b"),
    }


def muon_layers():
    geometries = muon_geometries()
    layout = muon_plane_layout()
    return [
        DetectorLayer(
            detector="MUON",
            name=station,
            z=z,
            xyshape=shape_name,
            geometry=geometries[shape_name],
        )
        for station, (z, shape_name) in layout.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        type=Path,
        default=ROOT / "examples" / "viewlhcb_lower_layout.csv",
        help="layout CSV to visualize",
    )
    parser.add_argument(
        "--shape-dir",
        type=Path,
        action="append",
        default=[PICKLES],
        help="additional pickle directory; may be repeated",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "docs" / "ViewLHCb" / "viewlhcb_lower_experiment_3d.html",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    layers = load_layout(args.layout, shape_dirs=args.shape_dir, up_dir=ROOT / "UP")
    if not any(layer.detector == "MUON" for layer in layers):
        layers = [*layers, *muon_layers()]
    layers = sorted(layers, key=lambda layer: layer.z)
    payload = [layer_payload(layer) for layer in layers]
    z_min = min(layer["z"] for layer in payload)
    z_max = max(layer["z"] for layer in payload)
    z_center = 0.5 * (z_min + z_max)
    summary = {}
    for layer in payload:
        summary[layer["detector"]] = summary.get(layer["detector"], 0) + 1

    args.output.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ViewLHCb lower experiment masks</title>
  <style>
    body {{
      margin: 0;
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #17202a;
    }}
    header {{
      padding: 14px 18px 10px;
      border-bottom: 1px solid #d9dee7;
      background: #fff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      color: #475569;
      font-size: 13px;
    }}
    main {{
      position: relative;
      min-height: 0;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      background: linear-gradient(#fbfdff, #e8eef6);
    }}
    .panel {{
      position: absolute;
      right: 16px;
      top: 16px;
      width: 270px;
      padding: 12px 14px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.94);
      box-shadow: 0 12px 26px rgba(15, 23, 42, 0.12);
      font-size: 13px;
      line-height: 1.55;
    }}
    .key {{
      display: inline-block;
      width: 11px;
      height: 11px;
      margin-right: 7px;
      border: 1px solid rgba(0, 0, 0, 0.25);
      vertical-align: -1px;
    }}
    label {{
      display: block;
      margin-top: 8px;
      color: #334155;
    }}
  </style>
</head>
<body>
  <header>
    <h1>ViewLHCb lower geometry: extracted XY masks at global z</h1>
    <div class="meta">
      <span>ViewLHCb masks: TV, UP, MP, FT, PicoCal</span>
      <span>Muon: M2-M5 from station mask builder</span>
      <span>Drag to rotate. Wheel to zoom.</span>
    </div>
  </header>
  <main>
    <canvas id="scene"></canvas>
    <div class="panel">
      <strong>Layers</strong>
      <div id="legend"></div>
      <label><input id="labels" type="checkbox" checked> labels</label>
      <label><input id="holes" type="checkbox" checked> draw holes</label>
      <div style="margin-top:8px">z range: {z_min:.1f} to {z_max:.1f} mm</div>
    </div>
  </main>
  <script>
    const surfaces = {json.dumps(payload)};
    const summary = {json.dumps(summary, sort_keys=True)};
    const colors = {json.dumps(COLORS)};
    const zCenter = {z_center};
    const zMin = {z_min};
    const zMax = {z_max};
    const canvas = document.getElementById("scene");
    const context = canvas.getContext("2d");
    const labels = document.getElementById("labels");
    const holes = document.getElementById("holes");
    let yaw = -0.54;
    let pitch = 0.34;
    let zoom = 0.095;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    document.getElementById("legend").innerHTML = Object.entries(summary)
      .map(([name, count]) => `<div><span class="key" style="background:${{colors[name] || "#64748b"}}"></span>${{name}}: ${{count}}</div>`)
      .join("");

    function resize() {{
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * ratio);
      canvas.height = Math.floor(canvas.clientHeight * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }}

    function rotate(point) {{
      const x = point.x;
      const y = point.y;
      const z = point.z - zCenter;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);
      const x1 = x * cy - z * sy;
      const z1 = x * sy + z * cy;
      return {{ x: x1, y: y * cp - z1 * sp, z: y * sp + z1 * cp }};
    }}

    function project(point) {{
      const rotated = rotate(point);
      const distance = 12000;
      const scale = zoom * distance / (distance + rotated.z);
      return {{
        x: canvas.clientWidth / 2 + rotated.x * scale,
        y: canvas.clientHeight / 2 - rotated.y * scale,
        depth: rotated.z,
      }};
    }}

    function pathFor(points, z) {{
      context.beginPath();
      points.forEach(([x, y], index) => {{
        const p = project({{ x, y, z }});
        if (index === 0) context.moveTo(p.x, p.y);
        else context.lineTo(p.x, p.y);
      }});
      context.closePath();
    }}

    function drawPart(part, surface) {{
      pathFor(part.outer, surface.z);
      const opacity = surface.detector === "TV" ? 0.30 :
        surface.detector === "UP" ? 0.40 :
        surface.detector === "MP" ? 0.18 :
        surface.detector === "FT" ? 0.20 :
        surface.detector === "PicoCal" ? 0.32 : 0.18;
      context.fillStyle = surface.color + Math.round(opacity * 255).toString(16).padStart(2, "0");
      context.strokeStyle = surface.color;
      context.lineWidth = surface.detector === "MUON" ? 1.2 : 0.7;
      context.fill();
      context.stroke();
      if (holes.checked) {{
        for (const hole of part.holes) {{
          pathFor(hole, surface.z);
          context.fillStyle = "#f8fafc";
          context.fill();
          context.strokeStyle = "#8b1e1e";
          context.lineWidth = 0.8;
          context.stroke();
        }}
      }}
    }}

    function shouldLabel(surface) {{
      if (!labels.checked) return false;
      if (surface.detector === "TV") return surface.name.endsWith("m287p996") || surface.name.endsWith("558p146");
      if (surface.detector === "MP" || surface.detector === "FT") return surface.name.endsWith("070") || surface.name.endsWith("625");
      return surface.detector !== "UP" || surface.name.includes("2314") || surface.name.includes("2655");
    }}

    function drawSurface(surface) {{
      for (const part of surface.parts) drawPart(part, surface);
      if (shouldLabel(surface)) {{
        const [xmin, , , ymax] = surface.bounds;
        const p = project({{ x: xmin, y: ymax + 90, z: surface.z }});
        context.fillStyle = "#17202a";
        context.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
        context.fillText(`${{surface.detector}} ${{surface.name}} z=${{surface.z.toFixed(1)}}`, p.x, p.y);
      }}
    }}

    function drawAxis() {{
      const x = -6500, y = -5400;
      const start = project({{ x, y, z: zMin }});
      const end = project({{ x, y, z: zMax }});
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.strokeStyle = "#334155";
      context.lineWidth = 2.5;
      context.stroke();
      context.fillStyle = "#334155";
      context.font = "14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      context.fillText("z", end.x + 8, end.y);
    }}

    function draw() {{
      context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawAxis();
      surfaces
        .map(surface => ({{ surface, depth: rotate({{ x: 0, y: 0, z: surface.z }}).z }}))
        .sort((a, b) => b.depth - a.depth)
        .forEach(item => drawSurface(item.surface));
    }}

    canvas.addEventListener("pointerdown", event => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener("pointermove", event => {{
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.006;
      pitch += (event.clientY - lastY) * 0.006;
      pitch = Math.max(-1.2, Math.min(1.2, pitch));
      lastX = event.clientX;
      lastY = event.clientY;
      draw();
    }});
    canvas.addEventListener("pointerup", () => {{ dragging = false; }});
    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      zoom *= event.deltaY > 0 ? 0.92 : 1.08;
      zoom = Math.max(0.02, Math.min(0.6, zoom));
      draw();
    }}, {{ passive: false }});
    labels.addEventListener("change", draw);
    holes.addEventListener("change", draw);
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
""", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
