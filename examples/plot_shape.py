"""Display a pickled detector XY shape."""

import argparse

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

from xyshapes_detectors import load_geometry


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pickle", help="path to a generated geometry pickle")
args = parser.parse_args()

shape = load_geometry(args.pickle)
figure, axis = plt.subplots()
plot_polygon(shape, ax=axis, add_points=False, facecolor="tab:blue", alpha=0.4)
axis.set_aspect("equal")
axis.set_xlabel("x [mm]")
axis.set_ylabel("y [mm]")
axis.set_title(args.pickle)
axis.grid(True)
plt.show()

