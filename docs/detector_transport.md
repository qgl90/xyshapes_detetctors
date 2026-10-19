# Detector Geometry and Track Transport

The transport workflow has three inputs: a detector setup, a field map, and a
track state. `DetectorTransport` connects them and returns fired-layer labels
for the track.

For all three UP variants and both FT module layouts in a single propagation,
use `label_tracks` or the reusable `GeometryComparison` service described in
[the multi-geometry API](geometry_comparison.md). Every returned mask has a
geometry suffix, including `hasTV_3Hits` and `hasTV_4Hits`. All catalog setups
now retain the TV left/right sensor masks at 100 actual sensor-centre Zs,
grouped into 50 physical-module counting layers. Upstream planes do not fire
for tracks born at the origin.

## Edit a Detector

Use `examples/build_lhcb_xy_setup.py` to generate a setup from the available
TV, UP, RICH1, MP, FT, RICH2, PicoCal, and Muon surfaces. For a new variant,
edit the builder inputs or its source layout, then write a distinct setup
directory. Do not edit a generated `layout.csv` without also ensuring its
referenced XY pickle exists.

Each setup has:

- `layout.csv`: `Use`, `detector`, `layer`, `station`, `z`, and `xyshape`.
- `shapes/` or catalog `shape_dirs`: trusted Shapely pickle geometries keyed
  by `xyshape`.
- `config.json`: the choices used to create the setup.

`layer` identifies a physical surface. Several surfaces can share a `station`
to count as one logical fired layer. The setup's Z values and XY shapes must
use mm and the same global coordinate frame as the field map.

Available catalog configurations are listed by:

```python
from xyshapes_detectors import available_configurations

for setup in available_configurations():
    print(setup["id"], setup["label"], setup["available"])
```

## FT_withHole Variants

Append `_FT_withHole` to any of the six original setup IDs to select the
corresponding enlarged-hole FT geometry. The operation is:

```python
hole = join_left_right_at_x0(union_of_all_five_MP_XY_masks)
ft_with_hole = original_FT_XY_mask.difference(hole)
```

For every horizontal band, join the nearest left and right MP edges across
X=0 with a solid rectangle. The two sides otherwise keep their original shapes
and square outer steps. There are no diagonal interpolations across outer steps.
An FT point inside that envelope is rejected even if it misses the active MP
patches. The same global XY envelope is
subtracted after each FT layer's stereo rotation, without projection between Z
positions. A removed FT crossing therefore does not imply an MP hit at its
different Z coordinate.

Run `PYTHONPATH=src python examples/build_ft_with_hole_variants.py` after
changing MP or FT source masks. The builder saves nine shared derived pickles,
one layout/configuration per source setup, and per-plane active-area summaries.
`config.json` records the stepped vertices, dimensions, and construction. Shared
`shape_dirs` in a setup config are relative to that setup directory, so a variant
loads both by catalog ID and by directory path.

## Load the Field

```python
from xyshapes_detectors import MagneticField

field = MagneticField.load("MagField/magnetic_field_v5r0.root")
reversed_field = MagneticField.load(
    "MagField/magnetic_field_v5r0.root", polarity=-1
)
```

The ROOT reader loads the `Bx`, `By`, and `Bz` TH3 maps. Reversing a dipole
polarity multiplies the full field vector by -1, including fringe components;
it is not a By-only change. `polarity` accepts `+1` or `-1` and does not write a
new map. `scale` can separately adjust the field strength. This is a polarity
sign reversal, not a substitute for a different measured field map if the
magnet configuration changes its spatial shape.

## Propagate and Label

The input state is `(x, y, tx, ty, q/p)` at its stated `z`; positions are mm,
slopes are `dx/dz` and `dy/dz`, momentum is MeV/c, and `q/p` is in inverse
MeV/c. The start state must be upstream of every surface in the selected
setup.

```python
from xyshapes_detectors import DetectorTransport, TrackState

transport = DetectorTransport.from_files(
    "lhcb_lower_up_y8x6_ft_8_8_10",
    "MagField/magnetic_field_v5r0.root",
    polarity=+1,
    step_mm=10,
)
state = TrackState.from_charge_momentum(
    charge=1, momentum_mev=50_000,
    x=0, y=0, tx=0.01, ty=-0.005, z=-300,
)
result = transport.propagate(state)
track_record = {"event_id": 42, **result.as_record()}
```

`result.trajectory` contains the integrated path and XY intercepts. `layer_hits`
maps names such as `FT_scifi_layer1` and `MUON_muon_m2` to booleans;
`detector_counts` gives fired logical layers by detector. `as_record()` returns
scalar fields such as `nTV`, `nUP`, `nFT`, `nMuon`, `hasRichEntry`,
`hasRichExit`, `hasRich1Entry`, `hasRich2Entry`, `hasPicoCal`, and `isMuon`.
The existing track record can be updated with these values.

To compare setups, create one `DetectorTransport` per setup and give each
result a distinct suffix:

```python
lower = lower_transport.propagate(state, label_suffix="lower").as_record()
baseline = baseline_transport.propagate(state, label_suffix="baseline").as_record()
track_record.update(lower)
track_record.update(baseline)
```

The integrator uses RK4 in z and interpolates the field trilinearly. It models
vacuum transport only: no material, energy loss, or scattering. A path that
turns or becomes nearly transverse to z raises `TrajectoryError`; no partial
acceptance labels are returned. The older batch `AcceptanceModel.evaluate`
API remains available for existing path-length transport workflows.

Run the field-map toy validation with:

```bash
PYTHONPATH=src python examples/validate_toy_trajectories.py
```

## Batches and Parallel Workers

For ntuples, use `propagate_batch` instead of calling the single-state spline
API once per track. It evaluates array-valued field queries using the existing
vectorized path-length RK4 integrator and returns plane crossings plus mask
columns. It accepts a list of `TrackState` objects or broadcastable columns:

```python
columns = {
    "x": x_array, "y": y_array, "z": z_array,
    "tx": tx_array, "ty": ty_array, "q_over_p": qop_array,
}
batch = transport.propagate_batch(columns, label_suffix="nominal")
track_table.update(batch.as_columns())

# Same trajectories, different XY masks: no further field integration.
other = DetectorTransport(other_configuration, transport.field)
track_table.update(other.labels_from_crossings(
    batch.crossings, label_suffix="withHole",
))
```

The crossing Z arrays must match exactly for reuse. The twelve current catalog
setups have the same physical planes. Different field polarities require
different trajectories. The batch API raises `TrajectoryError` with failed row
indices when a track curls, escapes, runs out of steps, or misses requested
planes; failed transport does not become an ordinary acceptance miss.

Use batches of approximately 1024--4096 tracks and separate processes for
parallel execution. Each worker should load its geometry/field once. The
reusable example uses `ProcessPoolExecutor` with a worker initializer:

```bash
PYTHONPATH=src python examples/benchmark_transport.py \
  --tracks 8192 --batch-size 1024 --workers 4 --step-mm 10
```

Measured on the task machine on 18 October 2026, including mask labels:

| Method | Time per track | Estimate for 4 million |
| --- | ---: | ---: |
| Individual Z spline, 10 mm steps | 314 ms | 14.6 days |
| Vectorized batches of 1024, one process, 10 mm | 1.25 ms | 1.4 hours |
| Same batches, four processes, 10 mm | 0.45 ms | 30 minutes |

Batch timings used 8192 toy tracks at 10--100 GeV/c; the four-process timing
includes worker startup. These are local throughput extrapolations, not measured
four-million-track runs. Parallel and serial hit-count checksums agreed.

The single-state Z path has about 2005 nodes for -300 to 18870 mm, including
93 sensor Z values, and about 8000 RK4 field queries. The batch integrator uses
10 mm path-length steps within the field map and 100 mm outside it: roughly
1500 main steps for these tracks, plus plane-intersection substeps. Its queries
are vectorized across tracks.

Increasing the step is an additional option, separate from batching. In a
4096-track check, 50 mm versus 10 mm changed sensor XY by at most 0.0272 mm
and changed no logical-layer hit flags. A 12-track check of batch 10 mm versus
the Z integrator differed by at most 0.00012 mm with identical flags. Keep the
default 10 mm when establishing the reference, and validate larger steps on the
intended track sample and near acceptance boundaries before adopting them.
