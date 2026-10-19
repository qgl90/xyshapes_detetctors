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

Select one of twelve registered UP/SciFi geometries in the sidebar, inspect its
3D placement and plane inventory, and optionally compare a reference track
against another configuration. The optional field map is
`MagField/magnetic_field_v5r0.root`;
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
counts are TV 50 (backed by 100 sensor-Z surfaces), UP 4 (backed by 16 source planes), MP 5, FT 12, RICH entry
and exit faces, PicoCal entry and exit faces, and Muon M2-M5. PicoCal acceptance
is `hasPicoCal`, true when either configured PicoCal layer is hit.

TV uses the real left/right sensitive-silicon acceptances, retaining both
sensor-centre Z planes in each module. A module counts once if any sensor is
hit. From the origin, 21 upstream modules are not crossed; 29 are reachable.
The canonical source is `detector_setups/tv_sensor_geometry.json`.

### Annotate Tracks Across All Six Geometries

```python
from xyshapes_detectors import label_tracks

flags = label_tracks(
    dict(x=0., y=0., z=0., tx=tx_array, ty=ty_array, q_over_p=qop_array),
    field_path="MagField/magnetic_field_v5r0.root",  # mm, MeV, Tesla
    polarity=1,
)
tracks.update(flags)  # dict of same-length track columns
```

This propagates each track once and applies all three UP variants (`gdml`,
`y8`, `y8x6`) with both FT layouts (`8_8_10`, `10_10_12`), using **FT_withHole**.
Columns include
`hasTV_3Hits_up_y8x6_ft_8_8_10_withHole`, `hasTV_4Hits_up_y8x6_ft_8_8_10_withHole`,
`nUP_up_y8x6_ft_8_8_10_withHole` and `nFT_up_y8x6_ft_8_8_10_withHole`, plus the corresponding
fields for each other configuration. The TV flags mean >=3 and >=4 logical
module hits. `hasUP`/`hasFT` mean at least one hit; require `nUP == 4` or
`nFT == 12` for all logical layers. All detector flags are geometry-suffixed.
See [the comparison API](docs/geometry_comparison.md) for batching, custom
geometry labels, field polarity and externally projected crossings.

### Both Magnet Polarities

```python
from xyshapes_detectors import label_tracks_both_polarities

tracks.update(label_tracks_both_polarities(
    states, field_path="MagField/magnetic_field_v5r0.root",
    map_polarity="magdown",  # polarity of the supplied map
))
```

This returns both `_magup` and `_magdown` columns for each geometry, for example
`hasMT_up_y8x6_ft_8_8_10_withHole_magup`. Transport runs twice per track, once
per polarity, not twelve times. For large samples reuse `PolarityComparison`.
The default `hasMT` rule is `(nMP >= 3) | (nFT >= 10) | ((nMP >= 1) & (nFT >= 6))`.
Its four thresholds are configurable with `MTSelection`, globally or per
geometry. These flags describe hypothetical acceptance under each polarity;
they do not determine the polarity of the input data.

See [clone and decay-study quickstart](docs/decay_study_quickstart.md) and
[the geometry, TV and validation slides](docs/geometry_documentation/acceptance_study_summary.pdf).
The [tool user-guide presentation](docs/geometry_documentation/tool_user_guide_slides.pdf)
explains the workflow, default geometries and every flag family; the
[runtime output schema](docs/geometry_documentation/tool_output_schema.json)
lists all 492 default column names and dtypes.

Each of the six original setups also has an `_FT_withHole` variant. Its FT
acceptance excludes the **MP left/right sides joined near X=0**, filling their
central gap while preserving the original square outer steps. It spans
3288.4 x 672.6 mm; it is subtracted in global XY from every FT plane after the
stereo rotation. Placements, logical labels, and other detector masks stay the
same. Rebuild the six variants with:

```bash
PYTHONPATH=src python examples/build_ft_with_hole_variants.py
```

Load, for example, `lhcb_lower_up_y8x6_ft_8_8_10_FT_withHole`. The updated
[slide summary](docs/geometry_documentation/transport_ft_with_hole_slides.pdf)
compares the outer borders, enlarged holes, and track acceptance.
The [FT layer comparison](docs/geometry_documentation/ft_layer_variants_comparison.pdf)
shows original and holed X/U/V masks for 8, 10, and 12 modules.

## Transport and Mask Labels

`DetectorTransport` is the recommended end-to-end interface. Load a detector
setup and field map, propagate one state, then update the original track row
with the returned scalar fields:

```python
from xyshapes_detectors import (
    DetectorTransport,
    TrackState,
)

lower = DetectorTransport.from_files(
    "lhcb_lower_up_y8x6_ft_8_8_10",
    "MagField/magnetic_field_v5r0.root",
    polarity=+1,
)
state = TrackState.from_charge_momentum(
    charge=1, momentum_mev=50_000,
    x=0, y=0, tx=0.02, ty=-0.01, z=-300,
)
result = lower.propagate(state)
track_row.update(result.as_record())
```

`result.layer_hits` identifies individual fired logical layers, while
`result.detector_counts` gives fired-layer counts. `as_record()` contains
`nTV`, `nUP`, `nMP`, `nFT`, `nMuon`, RICH entry/exit flags, `hasPicoCal`, and
the per-station `fired_<detector>_<station>` fields. For side-by-side variants,
use a distinct suffix with each transport call:

```python
baseline = DetectorTransport.from_files(
    "lhcb_lower_up_gdml_ft_10_10_12",
    "MagField/magnetic_field_v5r0.root",
    polarity=-1,
)
track_row.update(lower.propagate(state, label_suffix="lower").as_record())
track_row.update(baseline.propagate(state, label_suffix="baseline").as_record())
```

For large track samples, use `transport.propagate_batch(columns)` with arrays
of `x, y, z, tx, ty, q_over_p`. Reuse the returned `batch.crossings` through
`other_transport.labels_from_crossings(...)` to compare XY variants at the same
Z positions without repeating field propagation. A process-pool example is:

```bash
PYTHONPATH=src python examples/benchmark_transport.py \
  --tracks 8192 --batch-size 1024 --workers 4 --step-mm 10
```

This run measured about 0.45 ms per track including mask labels on the task
machine, corresponding to approximately 30 minutes for four million tracks.
See the [transport guide](docs/detector_transport.md#batches-and-parallel-workers)
for the benchmark scope, step-size checks, and crossing-reuse constraints.

The lower-level batch API remains available for existing path-length studies:

```python
from xyshapes_detectors import (
    MagneticField,
    Track,
    load_mask_configuration,
    zero_field,
)

track = Track.from_slopes(+1, 50_000, 0.02, -0.01)
field = zero_field()  # Or MagneticField.load("MagField/magnetic_field_v5r0.root")

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
be loaded from trusted sources. See [detector_transport.md](docs/detector_transport.md)
for the geometry-editing and complete transport workflow.

## Field and Track Transport

The field service reads the `Bx`, `By`, and `Bz` TH3 histograms from the ROOT
map (the supplied file uses TH3D; TH3F is also supported). Field components
are in tesla, positions and z are in mm, and momentum is in MeV/c. On the map
axis, the integral from z=0 to 10000 mm is about -4.19 T m in By (the sign is
the map polarity).

The high-level `DetectorTransport` accepts the state `(x, y, dx/dz, dy/dz,
q/p)` at a starting z, transports it with RK4, and returns fired layer counts
and scalar mask flags. `polarity=-1` reverses all three components of the
loaded field map, including fringe fields. No material effects are included;
a track that becomes nearly transverse to z raises `TrajectoryError` rather
than returning incomplete crossings.

```python
from xyshapes_detectors import (
    DetectorTransport,
    TrackState,
)

transport = DetectorTransport.from_files(
    "lhcb_lower_up_y8x6_ft_8_8_10",
    "MagField/magnetic_field_v5r0.root",
    polarity=+1,
)
state = TrackState.from_charge_momentum(
    charge=1, momentum_mev=50_000,
    x=0, y=0, tx=0.01, ty=-0.005, z=-300,
)
result = transport.propagate(state)
track_properties = result.as_record()  # includes nTV, nUP, nFT, hasRichEntry, ...
```

See [detector_transport.md](docs/detector_transport.md) for editing a setup,
loading variants, polarity validation, and writing labels back to a track row.
