# TV straight-ray acceptance from the origin

## Definition

These are geometric toy results, not reconstruction efficiencies. Rays start
at `(x,y,z)=(0,0,0)` and travel in positive Z, with no magnetic field, material,
sensor inefficiency or interaction-point spread.

For each pseudorapidity eta, use theta = 2 atan(exp(-eta)) and uniformly sample
azimuth phi. Slopes are `tx = cos(phi)/sinh(eta)` and
`ty = sin(phi)/sinh(eta)`, so `x(z)=tx*z` and `y(z)=ty*z` in mm.

The masks come from `docs/TV/review/tv_sensor_masks.pickle`: actual active
silicon polygons from the lower-scenario ROOT geometry. Each sensor is
tested at its own centre Z, with gaps preserved and boundaries included.
`nTV` counts a physical module once if any of its four sensors is hit, not
each sensor separately. Sensors at negative Z are not crossed: 29 of the
50 modules are reachable from this origin.

At every eta, the mean includes all azimuth directions, including zero hits.
The >=3 and >=4 acceptances are the fractions of azimuth directions passing
those hit requirements. There is no production-spectrum weighting in eta.

## Scan and Outputs

601 eta values from 1 to 7 (spacing 0.01), with 1440 midpoint azimuth samples
at each eta: 865,440 rays. An independent scan with 2880 azimuth samples
checks angular discretization; the maximum mean-hit difference is 0.00521,
and the largest acceptance differences are 0.208 and 0.382 percentage points
for >=3 and >=4. These are numerical quadrature checks, not detector
systematic uncertainties or statistical efficiency error bars.

- `tv_eta_acceptance.png`: mean nTV, the 10-90% azimuthal range, and the two
  requested threshold acceptances. The band is not an uncertainty on the mean.
- `tv_eta_phi_hits.png`: number of hits versus eta and phi, showing gaps.
- `tv_eta_acceptance.csv`: numerical curves, azimuth standard deviation and
  minimum/maximum hits.
- `tv_eta_phi_hits.npz`: eta grid, phi grid in radians and the full hit array.
- `summary.json`: source, assumptions, convergence and representative values.

The mean peaks at approximately 9.303 hits at eta=3.57. At eta=5 it is 1.732,
with 38.19% acceptance for >=3 hits and 0% for >=4. Tracks at sufficiently
high eta stay inside the central inactive region out to the last sensors.
These conclusions apply to the extracted geometry whose last sensor centre
is at Z=558.146 mm; they do not assume hypothetical stations at Z=700 mm.

## Reproduce

```bash
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/xyshapes-mpl \
  python examples/toy_tv_eta_acceptance.py
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/xyshapes-mpl \
  python -m unittest tests.test_tv_toy
```

CLI options include `--masks`, `--eta-min`, `--eta-max`, `--n-eta`, `--n-phi`
and `--output`. Three unit tests check upstream rejection, single counting
per module, per-sensor Z evaluation, gap preservation and eta-to-slope mapping.
