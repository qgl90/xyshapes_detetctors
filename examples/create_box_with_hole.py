"""Generate and reload a simple UP-like XY acceptance."""

from pathlib import Path

from xyshapes_detectors import box_with_hole, dump_geometries, load_geometry


output_dir = Path(__file__).parent / "output" / "UP"
shape = box_with_hole(2000, 1500, hole_width=80, hole_height=60)
output = dump_geometries({"MyUPShape": shape}, output_dir)[0]

# Demonstrate that the resulting file is an ordinary, readable Shapely pickle.
loaded = load_geometry(output)
print(f"wrote {output}")
print(f"type={loaded.geom_type}, area={loaded.area:g} mm^2, bounds={loaded.bounds}")

