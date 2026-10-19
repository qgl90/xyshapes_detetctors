"""Straight origin-ray TV acceptance, averaged uniformly over azimuth at fixed eta."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.ops import unary_union


def count_tv_hits(layers, tx, ty, *, origin_z=0.0):
    """Count modules, testing each sensor at its own downstream centre Z.

    tx and ty are dx/dz and dy/dz. A module fires if any member sensor
    contains the ray. XY starts at (0, 0) at origin_z. Boundaries count as hits.
    """
    tx, ty = np.broadcast_arrays(np.asarray(tx, dtype=float), np.asarray(ty, dtype=float))
    if not np.all(np.isfinite(tx)) or not np.all(np.isfinite(ty)):
        raise ValueError("slopes must be finite")
    counts = np.zeros(tx.shape, dtype=np.int32)
    for layer in layers:
        fired = np.zeros(tx.shape, dtype=bool)
        planes = {}
        for z, geometry in layer["surfaces"]:
            if z >= origin_z:
                planes.setdefault(float(z), []).append(geometry)
        for z, geometries in planes.items():
            geometry = unary_union(geometries)
            dz = z - origin_z
            fired |= shapely.intersects_xy(geometry, tx * dz, ty * dz)
        counts += fired
    return counts


def eta_scan(layers, eta, n_phi):
    eta = np.asarray(eta, dtype=float)
    if np.any(eta <= 0) or n_phi < 4:
        raise ValueError("use positive eta and at least four azimuth samples")
    # Midpoint quadrature avoids duplicating the azimuth seam.
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    slope = 1.0 / np.sinh(eta[:, None])
    counts = count_tv_hits(layers, slope * np.cos(phi), slope * np.sin(phi))
    return phi, counts


def mean_timing_resolution(counts, min_hits=1):
    """Mean of 50/sqrt(nTV) over passing directions; undefined if none pass."""
    if min_hits < 1:
        raise ValueError("timing requires at least one hit")
    counts = np.asarray(counts)
    selected = counts >= min_hits
    sigma = np.zeros(counts.shape, dtype=float)
    np.divide(50.0, np.sqrt(np.maximum(counts, 1)), out=sigma, where=selected)
    denominator = selected.sum(axis=1)
    return np.divide(sigma.sum(axis=1), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path,
                        default=Path("docs/TV/review/tv_sensor_masks.pickle"))
    parser.add_argument("--output", type=Path, default=Path("docs/TV/toy_origin"))
    parser.add_argument("--eta-min", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=7.0)
    parser.add_argument("--n-eta", type=int, default=601)
    parser.add_argument("--n-phi", type=int, default=1440)
    args = parser.parse_args()
    if not 0 < args.eta_min < args.eta_max or args.n_eta < 2:
        parser.error("need 0 < eta-min < eta-max and n-eta >= 2")
    with args.masks.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema") != "tv-silicon-surfaces-v1" or payload.get("units") != "mm":
        raise ValueError("expected sensor-resolved TV masks in mm")
    layers = payload["layers"]
    eta = np.linspace(args.eta_min, args.eta_max, args.n_eta)
    phi, counts = eta_scan(layers, eta, args.n_phi)
    mean = counts.mean(axis=1)
    std = counts.std(axis=1)
    a3, a4 = (counts >= 3).mean(axis=1), (counts >= 4).mean(axis=1)
    timing = mean_timing_resolution(counts)
    timing3 = mean_timing_resolution(counts, 3)
    timing4 = mean_timing_resolution(counts, 4)
    def eta_average(values):
        return float(np.sum(np.diff(eta) * (values[:-1] + values[1:]) / 2)
                     / (eta[-1] - eta[0]))

    integrated = dict(weighting="uniform in eta and azimuth; trapezoidal eta integration",
                      mean_nTV=eta_average(mean),
                      efficiency_ge3=eta_average(a3), efficiency_ge4=eta_average(a4))
    # Independent doubled-azimuth quadrature checks sharp sensor-edge effects.
    _, refined = eta_scan(layers, eta, 2 * args.n_phi)
    convergence = dict(
        max_mean_hit_difference=float(np.max(np.abs(mean - refined.mean(axis=1)))),
        max_acceptance3_difference=float(np.max(np.abs(a3 - (refined >= 3).mean(axis=1)))),
        max_acceptance4_difference=float(np.max(np.abs(a4 - (refined >= 4).mean(axis=1)))),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tv_eta_acceptance.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eta", "mean_nTV", "azimuth_std_nTV", "min_nTV", "max_nTV",
                         "acceptance_nTV_ge3", "acceptance_nTV_ge4",
                         "mean_sigma_t_ps_ge1", "mean_sigma_t_ps_ge3", "mean_sigma_t_ps_ge4"])
        writer.writerows(zip(eta, mean, std, counts.min(axis=1), counts.max(axis=1),
                             a3, a4, timing, timing3, timing4))
    np.savez_compressed(args.output / "tv_eta_phi_hits.npz", eta=eta, phi_rad=phi, nTV=counts)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    hit_line, = ax.plot(eta, mean, color="#147a91", linewidth=2.5, label="Mean nTV (all tracks)")
    ax.set(xlabel=r"Pseudorapidity $\eta$", ylabel="Average number of TV hits",
           title="TV: straight tracks from the origin",
           xlim=(args.eta_min, args.eta_max), ylim=(0, float(mean.max()) + 1))
    ax.text(0.98, 0.96, "Uniform azimuth average; one hit per module",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    for reference_eta, color in zip([4.4, 4.6, 4.8, 5.0],
                                    ["#477a42", "#b08024", "#b94e45", "#705694"]):
        ax.axvline(reference_eta, color=color, linestyle="--", linewidth=1.3)
        ax.text(reference_eta, 0.83, f"{reference_eta:.1f}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=color, fontsize=10, rotation=90)
    ax.grid(alpha=0.25)
    fig.savefig(args.output / "tv_mean_hits_vs_eta.png", dpi=180)
    timing_axis = ax.twinx()
    time_line, = timing_axis.plot(eta, timing, color="#b55147", linestyle="-.",
                                 linewidth=2, label=r"Mean $\sigma_t$ (nTV >= 1)")
    timing_axis.set(ylabel=r"Mean timing resolution $\sigma_t$ [ps]", ylim=(0, 55))
    timing_axis.tick_params(axis="y", colors="#b55147")
    timing_axis.yaxis.label.set_color("#b55147")
    ax.text(0.03, 0.04, r"$\sigma_t = 50\,\mathrm{ps}/\sqrt{nTV}$; zero hits: no timing measurement",
            transform=ax.transAxes, fontsize=9)
    ax.legend(handles=[hit_line, time_line], loc="upper left", fontsize=9)
    fig.savefig(args.output / "tv_mean_hits_and_timing_vs_eta.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    low, high = np.quantile(counts, [0.1, 0.9], axis=1)
    axes[0].fill_between(eta, low, high, color="#d4e8ee", label="10-90% range over azimuth")
    axes[0].plot(eta, mean, color="#147a91", linewidth=2, label="Mean nTV (including zero-hit tracks)")
    axes[0].set(ylabel="Fired TV modules", ylim=(0, max(3, float(high.max())+1)))
    axes[0].legend(fontsize=9)
    axes[1].plot(eta, a3 * 100, color="#287f4e", linewidth=2, label="nTV >= 3")
    axes[1].plot(eta, a4 * 100, color="#b55147", linewidth=2, label="nTV >= 4")
    axes[1].set(xlabel="Pseudorapidity eta", ylabel="Azimuthal acceptance [%]", ylim=(-2, 102))
    axes[1].legend(fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlim(args.eta_min, args.eta_max)
    fig.suptitle("TV geometric acceptance: straight rays from (0, 0, 0)\n"
                 "Actual silicon XY at sensor Z; uniform azimuth; one hit per module", fontsize=13)
    fig.savefig(args.output / "tv_eta_acceptance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(eta, a3 * 100, color="#287f4e", linewidth=2, label="nTV >= 3")
    ax.plot(eta, a4 * 100, color="#b55147", linewidth=2, label="nTV >= 4")
    for reference_eta in [4.4, 4.6, 4.8, 5.0]:
        if eta[0] <= reference_eta <= eta[-1]:
            ax.axvline(reference_eta, color="#999999", linestyle="--", linewidth=0.8)
    ax.set(xlabel=r"Pseudorapidity $\eta$", ylabel="Geometric efficiency [%]",
           title="TV hit requirements: straight tracks from the origin",
           xlim=(eta[0], eta[-1]), ylim=(-2, 103))
    ax.text(0.03, 0.08,
            f"Uniform eta / azimuth average over {eta[0]:.1f} <= eta <= {eta[-1]:.1f}:\n"
            f"nTV >= 3: {100*integrated['efficiency_ge3']:.2f}%\n"
            f"nTV >= 4: {100*integrated['efficiency_ge4']:.2f}%",
            transform=ax.transAxes, fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(args.output / "tv_hit_cut_efficiency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plot = ax.pcolormesh(eta, np.degrees(phi), counts.T, shading="nearest", cmap="viridis",
                         vmin=0, vmax=max(1, int(counts.max())), rasterized=True)
    fig.colorbar(plot, ax=ax, label="Fired TV modules (nTV)")
    ax.set(xlabel="Pseudorapidity eta", ylabel="Azimuth phi [degrees]",
           title="TV origin-ray hit map: actual sensor gaps retained")
    fig.savefig(args.output / "tv_eta_phi_hits.png", dpi=180)
    plt.close(fig)

    samples = []
    for target in [1.5, 2, 2.5, 3, 3.5, 4, 4.4, 4.5, 4.6, 4.8, 5, 5.5, 6]:
        if eta[0] <= target <= eta[-1]:
            i = int(np.argmin(np.abs(eta - target)))
            samples.append(dict(eta=float(eta[i]), mean_nTV=float(mean[i]),
                                acceptance_ge3=float(a3[i]), acceptance_ge4=float(a4[i]),
                                mean_sigma_t_ps_ge1=float(timing[i]) if np.isfinite(timing[i]) else None))
    summary = dict(source=payload["source"], masks=str(args.masks),
                   origin_mm=[0, 0, 0], field="none", material="none",
                   azimuth_weighting="uniform at each eta; midpoint quadrature",
                   logical_counting="any sensor hit per module, at actual sensor centre Z",
                   layers=len(layers),
                   reachable_layers=sum(any(z >= 0 for z, _ in r["surfaces"]) for r in layers),
                   eta_range=[float(eta[0]), float(eta[-1])], n_eta=len(eta),
                   n_phi=args.n_phi, rays=int(counts.size),
                   maximum_nTV=int(counts.max()), integrated=integrated,
                   timing_model="50 ps / sqrt(nTV); average per track conditional on nTV >= 1",
                   convergence=convergence, samples=samples)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
