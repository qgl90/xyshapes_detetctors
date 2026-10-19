"""Create the UP_v1_Borders geometry variant from visible UP pickles."""

from pathlib import Path
import pickle

from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "UP"
OUTPUT_DIR = ROOT / "UP_v1_Borders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_borders(shape):
    minx, miny, maxx, maxy = shape.bounds
    width = maxx - minx
    height = maxy - miny
    keep_x = width * 5 / 6
    keep_y = height * 3 / 4
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    crop = box(
        cx - keep_x / 2,
        cy - keep_y / 2,
        cx + keep_x / 2,
        cy + keep_y / 2,
    )
    return shape.intersection(crop)


def main():
    for source in sorted(SOURCE_DIR.glob("UP*.pickle")):
        with source.open("rb") as handle:
            shape = pickle.load(handle)
        cropped = crop_borders(shape)
        output = OUTPUT_DIR / source.name
        with output.open("wb") as handle:
            pickle.dump(cropped, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(output)


if __name__ == "__main__":
    main()
