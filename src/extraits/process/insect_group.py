"""
insect_group.py
================

The insect order (coleoptera / diptera / hymenoptera / lepidoptera, see
`insect_groups` in kp_infos.yaml) of every processed image. It is an input of
the measurement-validity classifiers and fills the '<group>_one_hot' CSV columns.

Sources, by priority:
    1. a table (--groups-csv): columns `image_name` and `group`;
    2. one order for every image (--group);
    3. the image path: the first folder of the path, below the input folder,
       whose name is an insect order, e.g. <input>/coleoptera/batch1/img.jpg.

An image matched by none of them gets no order: its one-hot columns are all 0
and the validity classifiers see the "unknown order" case.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd

from . import config
from .definitions import INSECT_GROUPS  # noqa: F401  (re-exported for csv_writer)


def _key(name: str) -> str:
    """Comparison key of an image name (file name or stem, see config.MATCH_ON)."""
    path = PurePosixPath(str(name).replace("\\", "/"))
    return path.name if config.MATCH_ON == "name" else path.stem


def group_from_path(path: str | Path) -> str | None:
    """The first component of `path` that names an insect order, case-insensitive."""
    known = {g.lower(): g for g in INSECT_GROUPS}
    for part in PurePosixPath(str(path).replace("\\", "/")).parts[:-1]:
        if part.lower() in known:
            return known[part.lower()]
    return None


class GroupIndex:
    """Holds the image-name -> order map and answers per-image lookups."""

    def __init__(self, index: dict[str, str], default: str | None = None):
        self._index = index
        self._default = default

    def __len__(self) -> int:
        return len(self._index)

    def group_of(self, image_name: str) -> str | None:
        return self._index.get(_key(image_name), self._default)

    def one_hot(self, image_name: str) -> dict[str, float]:
        group = self.group_of(image_name)
        return {f"{g}_one_hot": (1.0 if g == group else 0.0) for g in INSECT_GROUPS}


def _check_group(value: str, origin: str) -> str:
    known = {g.lower(): g for g in INSECT_GROUPS}
    group = known.get(str(value).strip().lower())
    if group is None:
        raise ValueError(f"{origin}: unknown insect order {value!r}; expected one of {INSECT_GROUPS}")
    return group


def build_group_index(relative_paths: list[str] | None = None, *, group: str | None = None,
                      groups_csv: str | Path | None = None) -> GroupIndex:
    """Build the index from a table, a single order and/or the image paths.

    `relative_paths` are the image paths relative to the input (folder source) or
    to the dataset root (Hugging Face source).
    """
    index: dict[str, str] = {}
    # --group outranks the folder names: they are only read without it.
    for rel in ([] if group else relative_paths or []):
        found = group_from_path(rel)
        if found:
            index[_key(rel)] = found
    n_from_paths = len(index)

    n_from_csv = 0
    if groups_csv is not None:
        table = pd.read_csv(groups_csv, dtype=str)
        missing = {"image_name", "group"} - set(table.columns)
        if missing:
            raise ValueError(f"{groups_csv}: missing column(s) {sorted(missing)}")
        for name, value in zip(table["image_name"], table["group"]):
            if isinstance(name, str) and isinstance(value, str) and value.strip():
                index[_key(name)] = _check_group(value, str(groups_csv))
                n_from_csv += 1

    default = _check_group(group, "--group") if group else None
    if groups_csv is not None:
        print(f"[group] {n_from_csv} image(s) from {groups_csv}")
    if default:
        print(f"[group] every other image: {default}")
    elif n_from_paths:
        print(f"[group] {n_from_paths} image(s) from their folder name")
    if not index and not default:
        print("[group] no insect order given (--group, --groups-csv or order-named folders): "
              "the one-hot columns are all 0 and the validity classifiers run without it.")
    return GroupIndex(index, default)
