"""List, inspect, and evaluate named LHCb detector-mask configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xyshapes_detectors import Track, available_configurations, load_mask_configuration
from xyshapes_detectors.magnetic_field import MagneticField, zero_field


def print_catalog(catalog):
    for item in available_configurations(catalog):
        state = "ready" if item["available"] else "missing setup files"
        print(f"{item['id']:45s} {item['label']} [{state}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", help="registered configuration ID or setup directory")
    parser.add_argument("--catalog", type=Path, help="configuration catalog JSON")
    parser.add_argument("--list", action="store_true", help="list all registered configurations")
    parser.add_argument(
        "--reference-track", nargs=7, type=float, metavar=("Q", "P", "TX", "TY", "X0", "Y0", "Z0"),
        help="evaluate one reference track; momentum is MeV/c and positions are mm",
    )
    parser.add_argument("--field", type=Path, help="ROOT or NPZ magnetic-field map")
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=10.0, help="transport step in mm")
    parser.add_argument("--output-json", type=Path, help="write the reference-track report as JSON")
    args = parser.parse_args()

    if args.list:
        print_catalog(args.catalog)
        return
    if not args.configuration:
        parser.error("provide a configuration ID, or use --list")

    configuration = load_mask_configuration(args.configuration, catalog=args.catalog)
    print(configuration.model.describe())
    print("\nStored XY surfaces grouped by fired layer:")
    for row in configuration.plane_inventory():
        print(
            f"{row['detector']:8s} {row['plane']:32s} "
            f"{row['fired_layer']:24s} z={row['z_mm']:9.3f} mm "
            f"XY={row['x_size_mm']:.1f} x {row['y_size_mm']:.1f} mm"
        )

    if args.reference_track is None:
        return

    if args.field:
        field = MagneticField.load(args.field, scale=args.field_scale)
        print(f"\nField: {field}")
    else:
        field = zero_field()
        print("\nField: zero (straight-line reference)")
    charge, momentum, tx, ty, x0, y0, z0 = args.reference_track
    track = Track.from_slopes(charge, momentum, tx, ty, (x0, y0, z0))
    report = configuration.evaluate_reference_track(track, field, step=args.step)

    print(f"\nReference trajectory stop: {report['stop_reason']}; z_exit={report['z_exit_mm']:.3f} mm")
    print("Fired layers and source surfaces:")
    for row in report["counts"]:
        print(
            f"  {row['detector']:8s} layers {row['fired_layers']:2d}/{row['layers']:2d}; "
            f"surfaces {row['fired_surfaces']:2d}/{row['surfaces']:2d}"
        )
    print("Fired plane rows:")
    for row in report["planes"]:
        if row["fired"]:
            print(f"  {row['detector']:8s} {row['plane']:32s} z={row['z_mm']:.3f} mm")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
