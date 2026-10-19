"""Build the geometry-comparison slides and a small reproducible field toy."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from xyshapes_detectors import MTSelection, PolarityComparison, load_mask_configuration


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "geometry_documentation"
ASSETS = DOCS / "assets" / "acceptance_study"


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "pdf.fonttype": 42})
    selection = MTSelection()
    mp, ft = np.meshgrid(np.arange(6), np.arange(13), indexing="ij")
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.pcolormesh(np.arange(-.5, 13), np.arange(-.5, 6), selection.evaluate(mp, ft),
                  cmap=ListedColormap(["#f0f1f2", "#32866c"]), vmin=0, vmax=1,
                  edgecolors="white", linewidth=.5)
    ax.set(xlabel="nFT", ylabel="nMP", xticks=range(13), yticks=range(6),
           title="hasMT: green cells pass the proposed hit-count rule")
    save(fig, "mt_selection")

    config = load_mask_configuration("lhcb_lower_up_y8x6_ft_8_8_10_FT_withHole")
    rows = []
    for detector in config.model.detectors:
        stations = config.model.stations_of(detector)
        zs = [z for station in stations for z, _ in station.surfaces]
        rows.append(f"{detector} & {sum(s.n_surfaces for s in stations)} & {len(stations)} & "
                    f"{min(zs):.1f}--{max(zs):.1f} " + r"\\")
    (ASSETS / "inventory.tex").write_text("\\newcommand{\\InventoryRows}{\n"+"\n".join(rows)+"\n}\n")

    eta = np.linspace(1.5, 5.5, 41)
    phi = (np.arange(32)+.5) * 2*np.pi/32
    ee, pp, qq = np.meshgrid(eta, phi, [-1., 1.], indexing="ij")
    states = dict(x=0., y=0., z=0., tx=(np.cos(pp)/np.sinh(ee)).ravel(),
                  ty=(np.sin(pp)/np.sinh(ee)).ravel(), q_over_p=qq.ravel()/20000.)
    service = PolarityComparison.from_files(ROOT / "MagField" / "magnetic_field_v5r0.root")
    labels = service.label_tracks(states)
    kept = {key: value for key, value in labels.items()
            if key.startswith(("nTV_", "nUP_", "nMP_", "nFT_", "hasMT_"))}
    np.savez_compressed(ASSETS / "toy_track_counts.npz", eta=ee.ravel(), phi=pp.ravel(),
                        charge=qq.ravel(), **kept)
    invariants = {}
    for polarity in ("magup", "magdown"):
        low = f"up_gdml_ft_8_8_10_withHole_{polarity}"
        high = f"up_gdml_ft_10_10_12_withHole_{polarity}"
        invariants[f"identical_MP_{polarity}"] = bool(np.array_equal(labels[f"nMP_{low}"], labels[f"nMP_{high}"]))
        invariants[f"baseline_FT_contains_lower_{polarity}"] = bool(np.all(labels[f"nFT_{high}"] >= labels[f"nFT_{low}"]))
        invariants[f"UP_crop_order_{polarity}"] = bool(np.all(
            (labels[f"nUP_{low}"] >= labels[f"nUP_up_y8_ft_8_8_10_withHole_{polarity}"])
            & (labels[f"nUP_up_y8_ft_8_8_10_withHole_{polarity}"]
               >= labels[f"nUP_up_y8x6_ft_8_8_10_withHole_{polarity}"])))
    if not all(invariants.values()):
        raise AssertionError(invariants)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    mean = lambda key: labels[key].reshape(41, 32, 2).mean(axis=(1, 2))
    for ft_name, color in [("8_8_10", "#287ca6"), ("10_10_12", "#bc6c35")]:
        suffix = f"up_gdml_ft_{ft_name}_withHole_magdown"
        axes[0].plot(eta, mean(f"nFT_{suffix}"), color=color, label=ft_name.replace("_", "/"))
        axes[1].plot(eta, mean(f"nMP_{suffix}"), color=color, linestyle="--" if ft_name=="10_10_12" else "-",
                     label=ft_name.replace("_", "/"))
    for up, color in [("gdml", "#287ca6"), ("y8", "#32866c"), ("y8x6", "#bc4f55")]:
        axes[2].plot(eta, mean(f"nUP_up_{up}_ft_8_8_10_withHole_magdown"), color=color, label=up)
    for ax, detector, maximum in zip(axes, ["FT", "MP", "UP"], [12, 5, 4]):
        ax.set(xlabel=r"$\eta$", ylabel=f"Mean n{detector}", ylim=(-.1, maximum+.3), xlim=(1.5, 5.5))
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    fig.suptitle("Field toy: origin, p=20 GeV, both charges, uniform azimuth; FT_withHole; MagDown")
    save(fig, "toy_counts")

    field = service.comparisons["magdown"].services[next(iter(service.configurations))].field
    integral = field.integral_bdl(0, 10000)[1]/1000
    summary = dict(tracks=int(ee.size), eta_range=[1.5, 5.5], phi_samples=32,
                   eta_samples=41, momentum_MeV=20000, charges=[-1, 1],
                   origin_mm=[0, 0, 0], map_polarity="magdown", field_integral_By_Tm=float(integral),
                   mt_selection=asdict(selection), invariants=invariants,
                   weighting="uniform eta grid and azimuth; equal charges; illustrative toy, not physics efficiency")
    (ASSETS / "checks.json").write_text(json.dumps(summary, indent=2)+"\n")
    (ASSETS / "values.tex").write_text(
        f"\\newcommand{{\\ToyTracks}}{{{ee.size:,}}}\n"
        f"\\newcommand{{\\FieldIntegral}}{{{integral:.4f}}}\n")

    output = ROOT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    tex = DOCS / "acceptance_study_summary.tex"
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-output-directory={output}", str(tex)], cwd=DOCS,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pdf = output / "acceptance_study_summary.pdf"
    shutil.copyfile(pdf, DOCS / pdf.name)
    print(json.dumps(summary, indent=2))
    print(pdf)


if __name__ == "__main__":
    main()
