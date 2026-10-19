# ViewLHCb Acceptance Extraction

The current checkout contains ViewLHCb ROOT geometry files, not standalone GDML:

- `View_LHCb_run5/LHCb_lower-scenario-Sep2026.root`
- `View_LHCb_run5/LHCb_middle-scenario-July2026.root`

The intended pipeline is:

1. Export detector placements from ROOT or GDML to a compact JSON table.
2. Convert that JSON into Shapely pickles and a Run 5-style layout CSV.
3. Load the layout with `xyshapes_detectors.layout.load_layout(..., shape_dirs=[...])`.

## JSON Schema

```json
{
  "surfaces": [
    {
      "detector": "TV",
      "layer": "TV_000",
      "z": 12.3,
      "points": [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    }
  ]
}
```

For axis-aligned boxes, `bounds` may be used instead:

```json
{"detector": "FT", "layer": "T1", "z": 7931.0, "bounds": [-2646.5, -2177.0, 2646.5, 2177.0]}
```

Records with identical `detector`, `layer`, and `z` are unioned. This is useful
for two detector halves or many sensors in one station.

## GDML Path

If the lower-geometry GDML is available, run for each detector branch. Examples:

```bash
PYTHONPATH=src python examples/extract_gdml_acceptances.py lower.gdml \
  --detector TV \
  --root-pattern 'lvTV' \
  --leaf-pattern 'lvModule|Sensor|Pixel' \
  -o ViewLHCb_acceptances/tv_lower.json

PYTHONPATH=src python examples/extract_gdml_acceptances.py lower.gdml \
  --detector FT \
  --root-pattern 'lvFT' \
  --leaf-pattern 'Module|Mat|Fibre|SciFi' \
  -o ViewLHCb_acceptances/ft_lower.json
```

Then convert:

```bash
PYTHONPATH=src python examples/export_viewlhcb_acceptances.py \
  ViewLHCb_acceptances/tv_lower.json \
  -o ViewLHCb_acceptances/pickles \
  --layout examples/viewlhcb_tv_layout.csv
```

## ROOT Path

The ViewLHCb web viewer identifies the relevant roots as:

- TV / VELO: `lvTV`
- UP: `lvUP`
- MP: `lvMP`
- FT: `lvFT`
- PicoCal: `lvPicoCal`

On this machine the local ROOT installation currently fails in cling before it
can open the ROOT file, so the extraction could not be run live here. Once ROOT
is working, the ROOT exporter should emit the same JSON schema above by walking
the `TGeoManager`, accumulating node matrices, and projecting selected box-like
active volumes to global XY.

The Python side is ready for those JSON exports:

```python
from xyshapes_detectors.layout import load_layout

layers = load_layout(
    "examples/viewlhcb_tv_layout.csv",
    shape_dirs=["ViewLHCb_acceptances/pickles"],
)
```
