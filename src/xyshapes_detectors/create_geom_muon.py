"""Generate Muon detector transverse acceptance shapes."""

import argparse

from .geometry_io import box_with_hole, dump_geometries


MUON_STATIONS = {
    "M2": {
        "xmin": 304.2,
        "ymin": 247.539444,
        "xmax": 4909.3,
        "ymax": 4076.538810,
        "z": 15270.0,
    },
    "M3": {
        "xmin": 329.1,
        "ymin": 266.444197,
        "xmax": 5292.5,
        "ymax": 4394.831870,
        "z": 16470.0,
    },
    "M4": {
        "xmin": 355.3,
        "ymin": 286.069997,
        "xmax": 5676.9,
        "ymax": 4712.603883,
        "z": 17670.0,
    },
    "M5": {
        "xmin": 379.6,
        "ymin": 307.740688,
        "xmax": 6059.6,
        "ymax": 5028.654769,
        "z": 18870.0,
    },
}


def muon_station_geometry(station):
    """Return one Muon station acceptance in the transverse XY plane."""
    try:
        dimensions = MUON_STATIONS[station]
    except KeyError as error:
        raise ValueError("station must be one of M2, M3, M4, or M5") from error

    return box_with_hole(
        2 * dimensions["xmax"],
        2 * dimensions["ymax"],
        hole_width=2 * dimensions["xmin"],
        hole_height=2 * dimensions["ymin"],
    )


def muon_geometries():
    return {
        station: muon_station_geometry(station)
        for station in MUON_STATIONS
    }


def muon_plane_layout():
    return {
        station: (dimensions["z"], station)
        for station, dimensions in MUON_STATIONS.items()
    }


MuonGeometries = muon_geometries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default=".", help="pickle destination")
    args = parser.parse_args()
    for path in dump_geometries(muon_geometries(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
