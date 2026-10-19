# Track Labels Across Detector Geometries

## One Call

```python
import numpy as np
from xyshapes_detectors import label_tracks

tracks = dict(
    x=np.zeros(2), y=np.zeros(2), z=np.zeros(2),  # mm
    tx=np.array([0.04, 0.02]), ty=np.array([0.01, -0.01]),
    q_over_p=np.array([1/20000., -1/50000.]),     # 1/MeV
)
tracks.update(label_tracks(
    tracks, field_path="MagField/magnetic_field_v5r0.root", polarity=1,
))
```

Provide the field ROOT file separately (or an equivalent supported NPZ map).
For reverse polarity use `polarity=-1`, which reverses all three components,
not only By. There is no material scattering or energy loss.

Default geometry labels and their masks:

| Output suffix | UP | FT modules T1 / T2 / T3 |
|---|---|---|
| up_gdml_ft_8_8_10 | nominal GDML | 8 / 8 / 10 |
| up_gdml_ft_10_10_12 | nominal GDML | 10 / 10 / 12 |
| up_y8_ft_8_8_10 | top/bottom cutoff | 8 / 8 / 10 |
| up_y8_ft_10_10_12 | top/bottom cutoff | 10 / 10 / 12 |
| up_y8x6_ft_8_8_10 | top/bottom and left/right cutoff | 8 / 8 / 10 |
| up_y8x6_ft_10_10_12 | top/bottom and left/right cutoff | 10 / 10 / 12 |

Each setup includes TV, UP, RICH1, MP, FT, RICH2, PicoCal and Muon. TV is
identical across them: 200 sensitive silicon sensors represented by 100
coplanar-pair XY masks at their actual Z, explicitly grouped into 50 physical
modules. Left/right acceptances remain separate. TV sensors upstream of each
track's starting Z do not fire. This is now the main configuration geometry,
not a review-only alternative.

## Returned Columns

All arrays are in input-track order, with one scalar per track. For each
suffix, for example `up_y8x6_ft_8_8_10`, the outputs include:

- `nTV_<suffix>`: number of fired modules, counting each module once.
- `hasTV_3Hits_<suffix>`: `nTV >= 3`.
- `hasTV_4Hits_<suffix>`: `nTV >= 4`.
- `nUP_<suffix>`: 0-4 logical UP layers, backed by 16 physical surfaces.
- `nFT_<suffix>`: 0-12 logical X/U/V/X layers, four per station.
- `nMP_<suffix>`, `nMuon_<suffix>` and detector-presence flags.
- `hasRich1Entry_<suffix>`, `hasRich1Exit_<suffix>` and equivalent RICH2 flags.
- `hasRichEntry_<suffix>` and `hasRichExit_<suffix>`: either RICH detector.
- `hasPicoCal_<suffix>`: at least one configured PicoCal surface hit.

`hasUP` and `hasFT` mean **at least one** layer, not full tracking acceptance.
For full logical-layer coverage use `nUP == 4` and `nFT == 12`. The FT module
numbers change each layer's XY width, not the number of logical layers.
Set `include_layer_hits=True` for additional per-layer `fired_*` booleans.

No unsuffixed geometry-dependent flags are returned by the comparison API,
avoiding collisions when adding the six configurations to a track table.

## Reuse for Large Samples

```python
from xyshapes_detectors import GeometryComparison

comparison = GeometryComparison.from_files(
    "MagField/magnetic_field_v5r0.root", polarity=-1,
)
for chunk in input_chunks:
    chunk.update(comparison.label_tracks(chunk))
```

Load once and process bounded chunks (for example 1024-8192 tracks). Momentum
is in MeV and all positions are mm. Inputs accept broadcastable arrays or a
sequence of `TrackState` objects. A zero q/p is a neutral straight ray.
Do not pass a frame row count as an input state: provide the six named state
columns `x,y,z,tx,ty,q_over_p`.

The implementation propagates through the union of requested Z planes once,
then projects the crossings onto each configuration's plane inventory. It
does not repeat the magnetic-field integration six times. Different requested
Z inventories are supported. Failed, curling/backward or step-limited
transport raises `TrajectoryError`; it is not silently treated as acceptance
failure. Planes upstream of the starting state are legitimately not crossed.

## Custom Labels and FT_withHole

```python
comparison = GeometryComparison.from_files(
    "MagField/magnetic_field_v5r0.root",
    configurations={
        "lower": "lhcb_lower_up_y8x6_ft_8_8_10",
        "baselineFT": "lhcb_lower_up_y8x6_ft_10_10_12",
        "lowerHoled": "lhcb_lower_up_y8x6_ft_8_8_10_FT_withHole",
    },
)
```

Suffixes must start with a letter and contain only letters, digits and
underscores. All 12 catalog configurations use the same updated TV masks.
The default six compare nominal FT surfaces; select `_FT_withHole` explicitly
to subtract the filled-central-gap MP acceptance from FT.

## External Projection Service

To supply transport separately, project onto `comparison.plane_z` and create a
`PlaneCrossings` object. Then call:

```python
flags = comparison.labels_from_crossings(crossings, initial_z=track_start_z)
```

Crossing Zs must match that union inventory exactly. Supply the real starting
Z array so upstream TV planes are exempted from the completeness requirement.
Downstream transport must still be complete and valid.

## Geometry Maintenance

The canonical TV inventory is `detector_setups/tv_sensor_geometry.json`.
`tv_sensor_layers()` constructs the same masks for future builders. To update
saved setups after an audited source change:

```bash
PYTHONPATH=src python examples/update_tv_detector_setups.py --refresh-views
```

This updates TV rows/configuration, trusted shape pickles, summaries, saved
setup pickles and existing 3D views. It validates that non-TV layout rows are
unchanged. Raw historical source dumps and the explicitly named square TV
mockup generator are retained as provenance, not used by the main catalog.
