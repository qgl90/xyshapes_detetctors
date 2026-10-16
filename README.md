# LHCb Detector Masks

A small catalog of reusable XY detector surfaces, a loader/viewer, and track
acceptance labels. Each setup stores detector plane Z positions, logical layer
names, and Shapely XY surfaces. Track projection to those planes is provided by
the transport API; external projectors can instead submit their XY crossings
to the same masks.

## Install and View

```bash
python -m pip install -e ".[app,field]"
PYTHONPATH=src streamlit run examples/streamlit_lhcb_mask_loader.py
```

Select one of six registered UP/SciFi geometries in the sidebar, inspect its
3D placement and plane inventory, and optionally compare a reference track
against another configuration. The optional field map is `magnetic_field_v5r0.root`;
without a field map, the app uses straight-line transport.

Configurations and their choices are described in
[`detector_setups/catalog.json`](detector_setups/catalog.json). Each setup
directory contains `config.json`, `layout.csv`, a compact summary, and a
compressed `experiment_3d.html.gz` view. Pickles are stored once in the shared
shape directories referenced by the catalog. `layout.csv` is authoritative
for detector, surface, logical layer/station, and Z placement.

Current catalog choices:

| UP border | SciFi modules T1-T2-T3 |
| --- | --- |
| GDML | 8-8-10 or 10-10-12 |
| Y8 | 8-8-10 or 10-10-12 |
| Y8X6 | 8-8-10 or 10-10-12 |

Each setup includes TV, UP, RICH1, MP, FT, RICH2, PicoCal, and Muon. Logical
counts are TV 50, UP 4 (backed by 16 source planes), MP 5, FT 12, RICH entry
and exit faces, PicoCal entry and exit faces, and Muon M2-M5. PicoCal acceptance
is `hasPicoCal`, true when either configured PicoCal layer is hit.

## Mask Labels

Load one or more configurations and append distinct suffixes to the same track
table. For example:

```python
from xyshapes_detectors import load_mask_configuration, Track, zero_field

track = Track.from_slopes(+1, 50_000, 0.02, -0.01)
field = zero_field()  # Or MagneticField.load("magnetic_field_v5r0.root")

lower = load_mask_configuration("lhcb_lower_up_y8x6_ft_8_8_10")
baseline = load_mask_configuration("lhcb_lower_up_gdml_ft_10_10_12")
lower_flags = lower.model.evaluate([track], field, label_suffix="lower")
baseline_flags = baseline.model.evaluate([track], field, label_suffix="baseline")
```

Counts are per logical layer (not raw surfaces). Each geometry suffix produces
`nTV`, `nUP`, `nMP`, `nFT`, and `nMuon` plus detector-specific rich flags:
`hasRichEntry`, `hasRichExit`, `hasRich1Entry`, `hasRich1Exit`,
`hasRich2Entry`, `hasRich2Exit`, and `hasPicoCal`. Existing aliases such as
`nMUON`, `RICH1_entry`, `inPicoCal`, and `isMuon` remain available. Pass
`label_suffix="lower"` to produce names such as `nTV_lower` and
`hasRich2Entry_lower`.

If another transport package already calculated XY intersections, use
`model.label_crossings(...)` with columns named
`x_<detector>_<station>` / `y_<detector>_<station>`. The helper applies the
station-level mask without repeating projection. For exact per-surface Z
crossings, use `model.label_surface_crossings(...)` with columns named
`x_<detector>_<plane>` / `y_<detector>_<plane>`; this preserves all 16 UP
surface crossings while counting four logical UP layers.

The catalog-backed Python API is `available_configurations()` and
`load_mask_configuration(name)`. To list setups and plane inventories:

```bash
PYTHONPATH=src python examples/inspect_lhcb_masks.py --list
PYTHONPATH=src python examples/inspect_lhcb_masks.py lhcb_lower_up_y8x6_ft_8_8_10
```

ROOT field-map support is optional (`uproot`). Surface pickle files must only
be loaded from trusted sources.
