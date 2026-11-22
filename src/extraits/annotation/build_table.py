"""`extraits annotate build-table`: merge every annotation source into one table.

Merge every annotation source into the single table the training commands read:
``<workspace>/annotations/annotation_data.csv``, one row per image.

Three CSV sources, joined on the image file name:

    pose           the Label Studio exports, ALREADY converted to CSV by
                   `extraits annotate to-csv` -- that conversion is a separate step,
                   run it first.
                   -> image size, group, bbox, remarks, <keypoint>_x/_y/_v
    scale          hand-written CSV, see annotations/scale/README.md
                   -> scale_type, scale_px_per_mm, scale-bar box and text,
                      ruler direction and line range
    measurements   measurement_validation exports (one CSV per batch)
                   -> <measure>_status, <measure>_px

A cell is left empty when the source has nothing for that image: every image
known to any source gets a row, and no source is required.

Usage
-----
    extraits annotate to-csv          # 1. Label Studio exports -> pose CSV
    extraits annotate build-table     # 2. merge of the three CSVs
    extraits annotate build-table --pose pose.csv --scale my_scale.csv \\
        --measurements dir_or_file ... --output annotation_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = "image_name"

# Columns of the hand-written scale CSV (see its README). Any of them may be
# empty; only the image name is required.
SCALE_COLUMNS = [
    "scale_type",            # "ruler" | "scale_bar"
    "scale_px_per_mm",
    "scale_bar_bbox_x",      # scale bar only
    "scale_bar_bbox_y",      # scale bar only
    "scale_bar_text",        # scale bar only
    "ruler_direction",       # ruler only : "horizontal" | "vertical"
    "ruler_line_min",        # ruler only : line/column range the ruler spans
    "ruler_line_max",        # ruler only
]


def _basename(value) -> str:
    """File name of a path written with any separator, so a Windows path read on
    Linux still splits ('C:\\images\\a.png' -> 'a.png')."""
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def load_pose(source: Path) -> pd.DataFrame:
    """Pose annotations, as a CSV already converted by `extraits annotate to-csv`.

    The conversion is a step of its own, on purpose: this script merges CSVs, it
    does not read Label Studio exports.
    """
    if source.suffix.lower() != ".csv" or not source.is_file():
        raise SystemExit(
            f"The pose annotations must be an already converted CSV: {source}\n"
            f"Produce it first with: extraits annotate to-csv"
        )

    frame = pd.read_csv(source)
    if KEY not in frame.columns:
        raise KeyError(f"Column '{KEY}' missing from the pose annotations ({source})")
    return frame


def load_scale(source: Path) -> pd.DataFrame:
    """Hand-written scale annotations. Missing file -> no scale columns filled."""
    if not source.is_file():
        print(f"/!\\ no scale CSV at {source}: scale columns left empty")
        return pd.DataFrame(columns=[KEY, *SCALE_COLUMNS])

    frame = pd.read_csv(source)
    if KEY not in frame.columns:
        raise KeyError(f"Column '{KEY}' missing from the scale CSV ({source})")
    unknown = [c for c in frame.columns if c not in (KEY, *SCALE_COLUMNS)]
    if unknown:
        print(f"/!\\ unknown columns in {source.name}, kept as is: {unknown}")
    return frame


def load_measurements(sources) -> pd.DataFrame:
    """measurement_validation exports: '<measure>_status' / '<measure>_px' per image."""
    files: list[Path] = []
    for raw in sources:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"/!\\ skipped (not found): {path}")
    if not files:
        print("/!\\ no measurement-validation export: measurement columns left empty")
        return pd.DataFrame(columns=[KEY])

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        # The app writes the image as a full local path, under the name 'image'.
        column = "image" if "image" in frame.columns else KEY
        if column not in frame.columns:
            print(f"/!\\ {path.name}: neither 'image' nor '{KEY}', file skipped")
            continue
        frame[KEY] = frame[column].map(_basename)
        frame = frame.drop(columns=[c for c in (column,) if c != KEY])
        frame["measurements_source"] = path.name
        frames.append(frame)
        print(f"  {len(frame):5} rows  {path.name}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[KEY])


def deduplicate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """One row per image: a second annotation of the same image is dropped, loudly."""
    if frame.empty:
        return frame
    frame = frame.copy()
    frame[KEY] = frame[KEY].astype(str).map(_basename)
    duplicated = frame[KEY].duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, KEY].unique()[:3]
        print(f"/!\\ {label}: {int(duplicated.sum())} duplicate image(s), "
              f"only the first row is kept (e.g. {list(examples)})")
        frame = frame[~duplicated]
    return frame


def merge(pose: pd.DataFrame, scale: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    """Outer join of the three sources on the image name."""
    merged = pose
    for other, label in ((scale, "scale"), (measurements, "measurements")):
        if other.empty:
            continue
        overlap = [c for c in other.columns if c != KEY and c in merged.columns]
        if overlap:
            print(f"/!\\ columns duplicated between pose and {label}, "
                  f"those of {label} get the suffix '_{label}': {overlap}")
            other = other.rename(columns={c: f"{c}_{label}" for c in overlap})
        merged = merged.merge(other, on=KEY, how="outer")
    return merged.sort_values(KEY).reset_index(drop=True)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    from extraits import annotation as paths
    from extraits import workspace

    parser = argparse.ArgumentParser(
        prog=prog, description="Gather pose, scale and measurement validation into a single CSV.")
    parser.add_argument("--pose", default=str(paths.pose_csv()),
                        help="Pose CSV, produced by `extraits annotate to-csv` "
                             "(default: <workspace>/annotations/pose/pose_annotations.csv).")
    parser.add_argument("--scale", default=str(paths.scale_csv()),
                        help="Scale CSV (default: <workspace>/annotations/scale/scale_annotations.csv).")
    parser.add_argument("--measurements", nargs="*", default=[str(paths.measurements_dir())],
                        help="Measurement-validation CSVs, and/or folders "
                             "(default: <workspace>/annotations/meas_classifier).")
    parser.add_argument("--output", default=str(workspace.annotation_table()),
                        help="Combined CSV (default: <workspace>/annotations/annotation_data.csv).")
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)

    print("pose:")
    pose = deduplicate(load_pose(Path(args.pose)), "pose")
    print(f"  {len(pose):5} images")
    print("scale:")
    scale = deduplicate(load_scale(Path(args.scale)), "scale")
    print(f"  {len(scale):5} images")
    print("measurements:")
    measurements = deduplicate(load_measurements(args.measurements), "measurements")
    print(f"  {len(measurements):5} images")

    merged = merge(pose, scale, measurements)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    filled = merged.notna().sum()
    print(f"\n{len(merged)} images, {len(merged.columns)} columns -> {output}")
    for label, column in (("pose", "task_id"), ("scale", "scale_px_per_mm"),
                          ("measures", "measurements_source")):
        if column in merged.columns:
            print(f"  {label:8}: {int(filled[column]):5} image(s) filled in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
