"""Tests of `extraits process` that need no trained model, plus one smoke test that does."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extraits import kp_infos, workspace
from extraits.cli import build_parser, main
from extraits.process import config
from extraits.process.image_source import build_source, read_image_list
from extraits.process.insect_group import build_group_index, group_from_path


def _write_image(path: Path, size: int = 64) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size, size, 3), 200, dtype=np.uint8))
    return path


# --------------------------------------------------------------------------- #
# keypoint schema
# --------------------------------------------------------------------------- #
def test_kp_infos_ships_with_the_package():
    assert kp_infos.KP_INFOS_PATH.is_file()
    assert kp_infos.NUM_KEYPOINTS == len(kp_infos.KEYPOINT_NAMES) > 0
    assert set(kp_infos.INSECT_GROUPS) == {"coleoptera", "diptera", "hymenoptera", "lepidoptera"}
    for chain in kp_infos.MEASUREMENT_INDICES.values():
        assert all(0 <= i < kp_infos.NUM_KEYPOINTS for i in chain)


# --------------------------------------------------------------------------- #
# workspace and models folder
# --------------------------------------------------------------------------- #
def test_models_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv(workspace.MODELS_ENV, raising=False)
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(tmp_path))
    assert workspace.models_dir() == tmp_path.resolve() / "models"
    monkeypatch.setenv(workspace.MODELS_ENV, str(tmp_path / "m"))
    assert workspace.models_dir() == (tmp_path / "m").resolve()
    assert workspace.models_dir(tmp_path / "x") == (tmp_path / "x").resolve()


def test_use_models_dir_repoints_every_model(tmp_path):
    old = config.MODELS_DIR
    try:
        config.use_models_dir(tmp_path)
        assert config.POSE_MODELS_DIR == tmp_path.resolve() / "pose"
        assert config.SCALE_BAR_MODEL_PATH == tmp_path.resolve() / "scale_bar" / "best.pt"
        assert config.MEASUREMENT_CLASSIFIER_METRICS_CSV.parent == config.MEASUREMENT_CLASSIFIER_DIR
    finally:
        config.use_models_dir(old)


# --------------------------------------------------------------------------- #
# insect order
# --------------------------------------------------------------------------- #
def test_group_from_path():
    assert group_from_path("coleoptera/batch1/a.jpg") == "coleoptera"
    assert group_from_path("Lepidoptera/a.jpg") == "lepidoptera"
    assert group_from_path("misc/a.jpg") is None
    assert group_from_path("coleoptera.jpg") is None      # the file name itself does not count


def test_group_index_priorities(tmp_path):
    table = tmp_path / "groups.csv"
    pd.DataFrame({"image_name": ["b.jpg"], "group": ["Diptera"]}).to_csv(table, index=False)

    index = build_group_index(["coleoptera/a.jpg", "coleoptera/b.jpg", "x/c.jpg"],
                              groups_csv=table)
    assert index.group_of("a.jpg") == "coleoptera"         # folder name
    assert index.group_of("b.jpg") == "diptera"            # the table outranks the folder
    assert index.group_of("c.jpg") is None
    assert sum(index.one_hot("c.jpg").values()) == 0

    index = build_group_index(["coleoptera/a.jpg"], group="hymenoptera")
    assert index.group_of("a.jpg") == "hymenoptera"        # --group outranks the folder


def test_group_index_rejects_unknown_order():
    with pytest.raises(ValueError):
        build_group_index([], group="araneae")


# --------------------------------------------------------------------------- #
# image source
# --------------------------------------------------------------------------- #
def test_local_source_lists_relative_paths(tmp_path):
    _write_image(tmp_path / "coleoptera" / "a.jpg")
    _write_image(tmp_path / "b.png")
    (tmp_path / "notes.txt").write_text("not an image")

    items, load_fn = build_source("folder", folder=tmp_path)
    assert sorted(rel for _, _, rel in items) == ["b.png", "coleoptera/a.jpg"]
    assert load_fn(items[0][0]).shape == (64, 64, 3)


def test_image_list_filters_by_stem(tmp_path):
    _write_image(tmp_path / "a.jpg")
    _write_image(tmp_path / "b.jpg")
    listing = tmp_path / "keep.txt"
    listing.write_text("a.png\n\n")
    assert read_image_list(listing) == {"a"}
    items, _ = build_source("folder", folder=tmp_path, image_list=listing)
    assert [name for _, name, _ in items] == ["a.jpg"]


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def test_cli_lists_the_commands(capsys):
    parser, runners = build_parser()
    assert {"process", "analyze"} <= set(runners)
    assert main([]) == 1
    assert "process" in capsys.readouterr().out


def test_process_without_models_explains_how_to_get_them(tmp_path):
    _write_image(tmp_path / "imgs" / "a.jpg")
    with pytest.raises(SystemExit, match="download-models"):
        main(["process", "-i", str(tmp_path / "imgs"), "--models-dir", str(tmp_path / "none")])


# --------------------------------------------------------------------------- #
# smoke test with the trained models
# --------------------------------------------------------------------------- #
@pytest.mark.models
@pytest.mark.skipif(not (workspace.models_dir() / workspace.POSE_SUBDIR).is_dir(),
                    reason="no trained models (set EXTRAITS_MODELS or run `extraits download-models`)")
def test_process_end_to_end(tmp_path):
    _write_image(tmp_path / "imgs" / "coleoptera" / "blank.jpg", size=320)
    out = tmp_path / "out.csv"
    code = main(["process", "-i", str(tmp_path / "imgs"), "-o", str(out), "--workers", "1"])
    assert code == 0
    df = pd.read_csv(out)
    assert list(df["image_name"]) == ["blank.jpg"]
    assert df.loc[0, "coleoptera_one_hot"] == 1.0
    assert main(["analyze", "-i", str(out), "--no-gt"]) == 0
    assert (tmp_path / "out_analysis" / "summary.txt").is_file()
