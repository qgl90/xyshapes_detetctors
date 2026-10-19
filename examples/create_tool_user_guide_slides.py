"""Generate the tool user-guide deck and an exact runtime output-schema example."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from xyshapes_detectors import MTSelection, PolarityComparison


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "geometry_documentation"
ASSETS = DOCS / "assets" / "tool_user_guide"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    states = dict(x=[0., 0.], y=[0., 0.], z=[0., 0.], tx=[.04, .02], ty=[.01, -.01],
                  q_over_p=[1/20000., -1/50000.])
    service = PolarityComparison.from_files(ROOT / "MagField" / "magnetic_field_v5r0.root")
    columns = service.label_tracks(states)
    geometry = "up_y8x6_ft_8_8_10_withHole"
    suffix = geometry + "_magup"
    properties = {name[:-len(suffix)-1]: str(values.dtype)
                  for name, values in columns.items() if name.endswith("_"+suffix)}
    for name, dtype in properties.items():
        for label in service.configurations:
            for polarity in ("magup", "magdown"):
                array = columns[f"{name}_{label}_{polarity}"]
                if array.shape != (2,) or str(array.dtype) != dtype:
                    raise AssertionError("inconsistent output schema")
    for label in service.configurations:
        for polarity in ("magup", "magdown"):
            tail = f"{label}_{polarity}"
            np.testing.assert_array_equal(columns[f"hasMT_{tail}"], MTSelection().evaluate(
                columns[f"nMP_{tail}"], columns[f"nFT_{tail}"]))
    metadata = dict(
        api="PolarityComparison.label_tracks", sample_states=states, units="mm, MeV/c, tesla",
        map_polarity=service.map_polarity, polarity_factors=service.polarity_factors,
        geometries={label: config.id for label, config in service.configurations.items()},
        mt_selection=asdict(MTSelection()), include_layer_hits=False,
        properties_per_prediction=len(properties), predictions=len(service.configurations)*2,
        total_columns=len(columns), properties=properties,
        columns={name: dict(dtype=str(values.dtype), example_values=values.tolist())
                 for name, values in columns.items()},
    )
    (DOCS / "tool_output_schema.json").write_text(json.dumps(metadata, indent=2)+"\n")

    rows = []
    config = service.configurations[geometry]
    for detector in config.model.detectors:
        stations = config.model.stations_of(detector)
        zs = [z for station in stations for z, _ in station.surfaces]
        rows.append(f"{detector} & {sum(s.n_surfaces for s in stations)} & {len(stations)} & "
                    f"{min(zs):.1f}--{max(zs):.1f} " + r"\\")
    (ASSETS / "inventory.tex").write_text("\\newcommand{\\InventoryRows}{\n"+"\n".join(rows)+"\n}\n")
    examples = []
    for row in range(2):
        values = [str(row+1)]
        for name in ("nTV", "nUP", "nMP", "nFT", "hasMT", "hasPicoCal"):
            value = columns[f"{name}_{suffix}"][row]
            values.append(("true" if value else "false") if isinstance(value, np.bool_) else str(value))
        examples.append(" & ".join(values)+r" \\")
    (ASSETS / "example_rows.tex").write_text("\\newcommand{\\ExampleRows}{\n"+"\n".join(examples)+"\n}\n")
    (ASSETS / "values.tex").write_text(
        f"\\newcommand{{\\PropertyCount}}{{{len(properties)}}}\n"
        f"\\newcommand{{\\ColumnCount}}{{{len(columns)}}}\n")
    output = ROOT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-output-directory={output}", str(DOCS / "tool_user_guide_slides.tex")],
                       cwd=DOCS, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pdf = output / "tool_user_guide_slides.pdf"
    shutil.copyfile(pdf, DOCS / pdf.name)
    print(f"Verified {len(properties)} properties x 12 geometry/polarity predictions = {len(columns)} columns")
    print(pdf)


if __name__ == "__main__":
    main()
