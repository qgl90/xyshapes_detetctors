# TV sensitive-silicon acceptance review

## Source and selection

The source is `View_LHCb_run5/LHCb_lower-scenario-Sep2026.root`, a ROOT
TGeo detector description. These masks are extracted directly from that
hierarchy, not from the earlier hand-made 20-plane TV mockup.

Within `lvTV`, select material **`TV:SensorSilicon`**. In this source its
volume is `lvDet`, a `TGeoBBox` measuring 38.73 x 12.8 x 0.23 mm.
Do not select every material containing the word silicon: that would also
include chip, substrate and guard-ring volumes, which are not active sensors.
The extractor rejects non-box sensitive shapes rather than silently using
their bounding box as acceptance.

The complete parent-to-global transform is applied to the eight corners of
each sensor box. The projected XY hull is exact for these boxes. No envelope
rectangle, artificial central hole or convex hull across separate sensors is
used. Real gaps remain inactive.

## Layers and positions

- 200 sensitive sensors belong to 50 physical modules (25 on each side).
- Each module has four sensors, in two coplanar pairs separated by 0.992 mm.
- Preserve those 100 distinct sensor-centre Z planes for hit evaluation.
- Count at most one hit per module: `TV_layer1` through `TV_layer50`, sorted
  by increasing global module-centre Z. The original ROOT module ID and
  sensor paths are retained separately for provenance.
- Sensor centres span -287.996 to 558.146 mm. Module means span -287.5 to
  557.65 mm. The source does not place active silicon near 700 or 780 mm.
- The 186.3 mm gap between module means 253.85 and 440.15 mm is present in
  the input hierarchy; it is not created by layer grouping.
- A forward trajectory born at Z=0 cannot fire the 21 upstream modules.

The previous saved 50-layer masks already match the union of the four actual
sensor outlines in each module. However, they put that union at the module
mean Z. The review export below preserves the two actual sensor-centre Zs,
without changing the logical counting convention. Calling the two Z planes
entry/exit sensor faces is incorrect: they contain different sensors.

## Reusable Outputs

- `sensor_setup/layout.csv` and `sensor_setup/shapes/TV/*.pickle`: standard
  loader-compatible layout; 100 surfaces, explicitly grouped into 50 layers.
- `tv_sensor_masks.pickle`: sensor-resolved dictionary, including 200 Shapely
  polygons with their centre Z and source paths. Load only trusted pickles.
- `tv_layers.csv`: module IDs, side, Z range, area and comparison to saved masks.
- `root_tv_inventory.json`: compact source hierarchy and exact sensor outlines.
- `tv_sensor_xy.png`: separate sensors and opposite-side XY overlay.
- `tv_sensor_z.png`: global module placement and downstream gap.

```python
from xyshapes_detectors.acceptance import AcceptanceModel

tv = AcceptanceModel.from_setup(
    "docs/TV/review/sensor_setup", exact_surfaces=True
)
# Supply XY at each value of tv.plane_z using the external trajectory service.
# tv.fired(crossings) returns one boolean per module, not one per sensor.
```

The same sensor geometry is now used in all registered full detector
configurations, through `detector_setups/tv_sensor_geometry.json` and the
canonical `tv_sensor_layers()` builder. Upstream masks are treated as not crossed for a forward
track born downstream; do not extend the track backwards to increase nTV.

## Reproduction

Run `examples/review_tv_root.py SOURCE.root --output
docs/TV/review/root_tv_inventory.json` with a consistent PyROOT environment.
Then run `PYTHONPATH=src python examples/review_tv_acceptance.py` in the project environment.
The second script verifies the stored mask outlines and Z means against the
sensor extraction, writes both mask formats, and reloads the standard layout
to check that 50 logical modules and 100 Z planes are preserved.
