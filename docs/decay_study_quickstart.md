# Clone and Annotate Tracks for Decay Studies

```bash
git clone --branch feature/detector-mask-loader https://github.com/qgl90/xyshapes_detetctors.git
cd xyshapes_detetctors
python -m venv .venv
source .venv/bin/activate
pip install -e '.[field]'
python -m unittest discover -s tests
```

The catalog geometry masks and `MagField/magnetic_field_v5r0.root` are included.
Loading masks needs no ROOT installation; the histogram field reader uses uproot.
Raw ROOT/GDML conversion and regenerating figures/slides have separate tooling
requirements, but are not needed to consume the saved detector setups.

## Track Arrays

```python
import numpy as np
from xyshapes_detectors import MTSelection, PolarityComparison

service = PolarityComparison.from_files(
    "MagField/magnetic_field_v5r0.root",
    map_polarity="magdown",
    mt_selection=MTSelection(min_mp=3, min_ft=10, mixed_min_mp=1, mixed_min_ft=6),
)

states = dict(
    x=np.array([0., 0.]), y=np.array([0., 0.]), z=np.array([0., 0.]),  # mm
    tx=np.array([.04, .02]), ty=np.array([.01, -.01]),
    q_over_p=np.array([1/20000., -1/50000.]),  # 1/MeV
)
flags = service.label_tracks(states)
assert all(column.shape == (2,) for column in flags.values())
```

Each state is `(x,y,dx/dz,dy/dz,q/p)` at its supplied Z. Use the production
vertex of each charged daughter, not necessarily the collision origin. No
back-propagation is performed and planes upstream of the state do not fire.
Use mm for positions and MeV/c for momentum. With momentum components,
`tx=px/pz`, `ty=py/pz`, and `q_over_p=charge/sqrt(px**2+py**2+pz**2)` for pz>0.

The default is all three UP variants x both FT layouts, with **FT_withHole**,
evaluated separately for magup and magdown. Other sub-detectors are included.
Example names are:

```text
nFT_up_y8x6_ft_8_8_10_withHole_magup
nMP_up_y8x6_ft_8_8_10_withHole_magup
nUP_up_y8x6_ft_8_8_10_withHole_magup
hasMT_up_y8x6_ft_8_8_10_withHole_magup
hasTV_3Hits_up_y8x6_ft_8_8_10_withHole_magup
hasTV_4Hits_up_y8x6_ft_8_8_10_withHole_magup
```

The corresponding magdown columns are also returned. `hasMT` implements the
OR of MP-only, FT-only and mixed requirements; TV and UP cuts are separate.
Keep nMP/nFT/nUP columns to change thresholds offline later. MP geometry is
unchanged across the FT variants, so nMP should match between the two FT
layouts for identical tracks, field and starting states.

## Add to Existing Ntuples Without Losing Decay Alignment

Extract one same-length state-array dictionary per daughter and process chunks:

```python
output_columns = dict(original_columns)
for daughter, daughter_states in daughter_state_columns.items():
    daughter_flags = service.label_tracks(daughter_states)
    output_columns.update({f"{daughter}_{name}": values
                           for name, values in daughter_flags.items()})
```

No event filtering or row reordering is performed. Prefix labels by daughter
so each candidate retains independent daughter flags. For jagged multi-track
collections, flatten the states and explicitly restore the original offsets
after labelling; do not lose candidate/track associations. For fixed-daughter
trees, each daughter array must have one entry per original candidate.

If a decay-level requirement needs all daughters to pass, combine the relevant
daughter booleans with AND, choosing the correct geometry and actual data
polarity. Save to a new file or friend tree rather than overwriting the input.
I/O and physics selections belong to the analysis code, not the propagator.

Use bounded chunks (start around 1024-8192 tracks) and reuse this service.
Failing/curling transport raises `TrajectoryError`; do not silently interpret
failed propagation as an acceptance miss. Both-polarity flags are hypothetical
predictions, not a measurement of the polarity of an event.

See [the API documentation](geometry_comparison.md) and
[the slides](geometry_documentation/acceptance_study_summary.pdf).
