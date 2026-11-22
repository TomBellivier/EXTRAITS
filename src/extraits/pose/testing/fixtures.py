"""Shared pytest fixtures: synthetic mini-corpus and throwaway project.

Imported by tests/pose/conftest.py and research/tests/conftest.py
(`from extraits.pose.testing.fixtures import *`). The smoke test runs end to end on
these fixtures in a few seconds (§10.3).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from extraits.pose.cli import CONFIG_DIR
from extraits.pose.data.adapters.synthetic import SyntheticAdapter
from extraits.pose.paths import ProjectPaths
from extraits.pose.registry import load_all_plugins

# The plugins are loaded AT IMPORT: the parametrisation of tests/test_smoke.py and of
# the approach tests reads the registry at pytest COLLECTION time, before any fixture.
# A deferred loading would leave the registry empty and make every fixture depending
# on it fail.
load_all_plugins()

DATASETS = ["coleoptera", "diptera"]
SCHEMA = "insect42_v1"   # ADR-0006: schema shared by the 4 datasets
N_KPTS = 42
# ADR-0016: the test corpus reproduces the absence of some points depending on the order
# (here: no annotated hind wings for the "diptera" of the toy corpus).
ABSENT_KEYPOINTS = {"coleoptera": [], "diptera": list(range(26, 34))}


@pytest.fixture()
def project(tmp_path: Path) -> ProjectPaths:
    """Throwaway project: real configs copied, synthetic data."""
    shutil.copytree(CONFIG_DIR, tmp_path / "configs")
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_writable_dirs()
    for dataset in DATASETS:
        adapter = SyntheticAdapter(
            dataset=dataset,
            source_dir=paths.raw_dir(dataset),
            options={
                "n_images": 24, "n_keypoints": N_KPTS, "n_groups": 8, "seed": 7,
                "keypoint_schema": SCHEMA, "image_size": 192,
                "absent_keypoints": ABSENT_KEYPOINTS[dataset],
                # Real images: the qualitative export (§8.5) is part of the smoke test.
                "write_images": True, "images_root": str(paths.images),
            },
        )
        adapter.run(paths)
    return paths


@pytest.fixture()
def raw_coco(project: ProjectPaths) -> ProjectPaths:
    """Add raw COCO annotations, to test the full `prepare` chain."""
    import json

    import numpy as np

    from extraits.pose.utils.io import read_parquet

    for dataset in DATASETS:
        annotations = read_parquet(project.annotations(dataset))
        images, anns = [], []
        for i, row in enumerate(annotations.itertuples(index=False)):
            images.append({"id": i, "file_name": f"{Path(row.image_path).stem}.png",
                           "width": int(row.image_width), "height": int(row.image_height)})
            kpts = np.asarray(row.kpts_xy, float).reshape(-1, 2)
            vis = np.asarray(row.kpts_vis, int).reshape(-1, 1)
            anns.append({"id": i, "image_id": i,
                         "keypoints": np.hstack([kpts, vis]).reshape(-1).tolist(),
                         "bbox": [float(v) for v in row.bbox_xywh], "area": float(row.area)})
        (project.raw_dir(dataset) / "annotations.json").write_text(
            json.dumps({"images": images, "annotations": anns}), encoding="utf-8"
        )
    return project


@pytest.fixture()
def config_factory(project: ProjectPaths):
    """Factory of configs pointing to the throwaway project.

    Changing approach REQUIRES recomposing the Hydra group (`approach=<name>`): patching
    `approach.name` would leave the keys of the previous approach in place, which is a
    source of silent errors.
    """
    from extraits.pose.cli import load_config

    def build(extra: list[str] | None = None) -> DictConfig:
        overrides = [
            # Every root of the throwaway project (ProjectPaths.default), which differs
            # from the workspace layout of paths.yaml (data/pose/, images/).
            f"paths.root={project.root}",
            f"paths.data={project.data}",
            f"paths.images={project.images}",
            f"paths.results={project.results}",
            f"paths.reports={project.reports}",
            f"paths.retained={project.retained}",
            f"paths.annotation_table={project.annotation_table}",
            "data=pooled",
            f"data.datasets=[{','.join(DATASETS)}]",
            "cv=kfold5_grouped",
            "cv.n_folds=3",
            "mode=smoke",
            "tag=test",
            "train.epochs=1",
            *(extra or []),
        ]
        config = load_config(overrides, config_dir=project.configs)
        OmegaConf.set_struct(config, False)
        return config

    return build


@pytest.fixture()
def cfg(config_factory) -> DictConfig:
    """Default config (reference approach)."""
    return config_factory()