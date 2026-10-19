# Configurable LHCb XY Surface Setups

Use `examples/build_lhcb_xy_setup.py` to produce one self-contained detector
setup directory for transport and acceptance-mask studies.

Each setup contains:

- `config.json`: the choices used to build the setup.
- `layout.csv`: ordered detector planes with `detector`, `layer`, `station`,
  `z`, and `xyshape`. `station` is the stable logical fired-layer label.
- `shapes/`: every Shapely pickle referenced by `layout.csv`.
- `detector_setup.pickle`: reusable payload with config, summary, and loaded
  `DetectorLayer` objects.
- `summary.csv`: source surface counts, fired-layer counts, z ranges, and XY
  envelopes by detector.
- `experiment_3d.html`: inspection view.
- `presentation/geometry_summary.pdf`: simplified visual presentation of the
  selected setup, including one page per sub-detector, XY surfaces with holes,
  z placement ticks, and UP/FT variant inventory.

## Presets

```bash
PYTHONPATH=src python examples/build_lhcb_xy_setup.py \
  --preset baseline \
  --name lhcb_xy_baseline

PYTHONPATH=src python examples/build_lhcb_xy_setup.py \
  --preset lower \
  --name lhcb_xy_lower
```

The baseline preset uses:

- UP: `gdml`, `all16` extracted planes.
- FT: T1/T2/T3 = `10/10/12` modules with `X,U,V,X` layers.
- Muon: M2-M5 included by default.
- MP: five logical layers, each made from the XY union of its extracted tiles.
- TV: 50 physical-module counting layers backed by 100 sensor-centre Z planes.
  Left/right acceptances stay separate; only coplanar sensor pairs are unioned.

The lower preset uses:

- UP: `y8x6`, `all16` extracted planes.
- FT: T1/T2/T3 = `8/8/10` modules with `X,U,V,X` layers.
- Muon: M2-M5 included by default.
- MP and TV use the same logical-layer reduction as baseline.

## Custom Setup

For a guided terminal prompt:

```bash
PYTHONPATH=src python examples/build_lhcb_xy_setup.py --interactive
```

For the Streamlit UI:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src streamlit run examples/streamlit_lhcb_builder.py
```

For a fully scripted custom setup:

```bash
PYTHONPATH=src python examples/build_lhcb_xy_setup.py \
  --name my_setup \
  --up-variant y8 \
  --up-planes four \
  --ft-t1 8 \
  --ft-t2 8 \
  --ft-t3 12
```

Allowed UP variants:

- `gdml`: nominal ViewLHCb/GDML UP acceptance.
- `y8`: Yisheng top/bottom crop applied to ViewLHCb/GDML UP.
- `y8x6`: Yisheng top/bottom plus left/right crop applied to
  ViewLHCb/GDML UP.

Allowed UP plane modes:

- `all16`: keep all 16 extracted UP surfaces, tagged into four fired layers.
- `four`: union each group of four surfaces into one logical plane.

The default `all16` mode keeps the 16 UP surfaces in the layout and tags each
set of four consecutive z surfaces as `UP_layer1` through `UP_layer4`. The
acceptance model therefore reports four UP fired layers. FT is represented by
12 separately fired `scifi_layer1` through `scifi_layer12` layers. MP has five
unioned layers. TV retains actual sensitive-silicon XY and Z coordinates,
grouping the two sensor planes within each module into one counting layer.
RICH1, RICH2, and PicoCal retain named entry and exit planes; Muon stations are
labeled `muon_m2` through `muon_m5`.

Allowed FT module counts for each station are `8`, `10`, and `12`.

## Acceptance Labels

Use a setup directory directly to augment propagated or propagatable tracks.
For a CSV with `charge`, `p`, and either `tx`/`ty` or `px`/`py`/`pz`:

```bash
PYTHONPATH=src python examples/augment_tracks_with_setup.py \
  detector_setups/lhcb_xy_lower \
  tracks.csv \
  tracks_with_acceptance.csv \
  --field magnetic_field_v5r0.root \
  --label-suffix Run5
```

For straight-line checks:

```bash
PYTHONPATH=src python examples/augment_tracks_with_setup.py \
  detector_setups/lhcb_xy_lower \
  tracks.csv \
  tracks_with_acceptance.csv \
  --zero-field
```

The compact output includes columns such as `nUP_Run5`, `hasUP_Run5`,
`nFT_Run5`, `hasFT_Run5`, `nMUON_Run5`, `hasMUON_Run5`, `isMuon`, and volume
entry/exit labels such as `RICH2_entry_exit_Run5`. Add `--station-flags` only
when per-station fired columns are needed; it is intentionally verbose.
