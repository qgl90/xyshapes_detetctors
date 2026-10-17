# Detector geometry slides

`detector_geometry_slides.tex` is the editable Beamer source for the slide
summary. `assets/` contains the generated geometry summary and six-configuration
comparison used as figure sources in the deck.

Build from this directory with:

```bash
pdflatex -halt-on-error -interaction=nonstopmode \
  -output-directory=../../output/pdf detector_geometry_slides.tex
cp ../../output/pdf/detector_geometry_slides.pdf .
```

The reusable source of truth for selected plane names, XY shapes, and exact
global z positions remains in `detector_setups/catalog.json` and each setup's
`layout.csv` / `shapes/` payload.
