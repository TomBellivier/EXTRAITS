"""Fine-tuning (produce and evaluate), end to end through the Ultralytics double.

Checks what fine-tuning adds on top of `train`: choosing the base members, reading the
new annotations (Label Studio exports or CSV), the produced ensemble and its
safeguards, the 10 folds and before/after runs of the evaluation, and that the main
annotations and the base models are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from extraits.pose.finetune import (
    cmd_finetune,
    find_base_model,
    load_finetune_data,
    select_members,
)
from extraits.pose.contracts import ContractError
from extraits.pose.data.keypoints import load_schema

N_IMAGES = 12
SIZE = 192


def _write_card(folder: Path, variant: str, fold: int) -> None:
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"weights")
    (folder / "model_card.json").write_text(
        json.dumps({"run_id": folder.name, "variant_hash": variant, "fold": fold}),
        encoding="utf-8")


def _finetune_images(root: Path, names: list[str]) -> Path:
    """finetune-images/coleoptera/{images/, export.json} in the Label Studio format."""
    from PIL import Image

    order_dir = root / "finetune-images" / "coleoptera"
    (order_dir / "images").mkdir(parents=True)
    tasks = []
    for i in range(N_IMAGES):
        name = f"ft_{i:02d}.png"
        Image.new("RGB", (SIZE, SIZE), (i * 10, 80, 120)).save(order_dir / "images" / name)
        results = [
            {"type": "keypointlabels", "original_width": SIZE, "original_height": SIZE,
             "value": {"keypointlabels": [point], "x": 20 + k * 1.2, "y": 30 + (k % 7) * 5}}
            for k, point in enumerate(names)
        ]
        tasks.append({"id": i, "data": {"img": f"/data/local-files/?d=images/{name}"},
                      "annotations": [{"id": i, "result": results}]})
    (order_dir / "export.json").write_text(json.dumps(tasks), encoding="utf-8")
    return root / "finetune-images"


@pytest.fixture()
def retained(tmp_path: Path) -> Path:
    pose = tmp_path / "retained" / "pose"
    _write_card(pose / "run_fold0", "abc123", 0)
    _write_card(pose / "run_fold1", "abc123", 1)
    _write_card(pose / "lonely", "solo99", 3)
    return pose


def test_base_model_is_fold0_or_the_only_member(retained: Path) -> None:
    weights, card = find_base_model(retained, "abc123")
    assert card["fold"] == 0 and weights.parent.name == "run_fold0"
    _, card = find_base_model(retained, "solo99")
    assert card["fold"] == 3
    with pytest.raises(ContractError):
        find_base_model(retained, "unknown")


def test_evaluate_runs_10_folds_and_compares(fake_ultralytics, config_factory, project,
                                             retained, tmp_path) -> None:
    names = list(load_schema("insect42_v1", project.configs).names)
    images_dir = _finetune_images(tmp_path, names)
    main_annotations = project.annotations("coleoptera").read_bytes()

    cfg = config_factory([
        "approach=yolo_pooled", "train.device=cpu",
        f"paths.retained={retained.parent}",
        "finetune.evaluate=true", "finetune.base=abc123", "finetune.epochs=1",
        f"finetune.images_dir={images_dir.as_posix()}",
    ])
    out_dir = cmd_finetune(cfg)

    per_fold = pd.read_csv(out_dir / "comparison_per_fold.csv")
    assert sorted(per_fold["fold"].unique()) == list(range(10))
    assert {"before", "after", "delta"} <= set(per_fold.columns)
    summary = pd.read_csv(out_dir / "comparison.csv")
    assert {"detection_rate", "n_images_detected"} <= set(summary["metric"])
    # Point metrics only on the images detected before AND after (none with the double,
    # which never matches the GT); every metric on every image in comparison_all_images.
    scopes = per_fold.groupby("metric")["evaluated_on"].first()
    assert scopes["detection_rate"] == "all_test_images"
    points = per_fold[per_fold["evaluated_on"] == "detected_before_and_after"]
    detected = per_fold[per_fold["metric"] == "n_images_detected"]
    assert points.empty == (detected[["before", "after"]].min(axis=1) == 0).all()
    raw = pd.read_csv(out_dir / "comparison_all_images.csv")
    assert "oks_ap" in set(raw["metric"])
    # One before | after figure per annotated image (each is in exactly one test fold).
    assert len(list((out_dir / "figures").glob("fold*__*.png"))) == N_IMAGES
    # One fine-tuning per fold, all from the base model, for the epochs asked.
    trains = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert len(trains) == 10 and all(c["epochs"] == 1 for c in trains)
    # The main datasets are never overwritten.
    assert project.annotations("coleoptera").read_bytes() == main_annotations


def test_member_selection(retained: Path) -> None:
    assert [m.name for m in select_members(retained, None, single=False)] == [
        "lonely", "run_fold0", "run_fold1"]
    assert [m.name for m in select_members(retained, "abc123", single=False)] == [
        "run_fold0", "run_fold1"]
    assert [m.name for m in select_members(retained, "lonely", single=True)] == ["lonely"]
    assert [m.name for m in select_members(retained, None, single=True)] == ["run_fold0"]
    with pytest.raises(ContractError, match="No member"):
        select_members(retained, "nope", single=False)


def test_csv_annotations_with_group_and_image_size(project, tmp_path) -> None:
    """The simple CSV format: image sizes read from the files, --group fills the order,
    images found by file name anywhere under --images."""
    from PIL import Image

    names = list(load_schema("insect42_v1", project.configs).names)
    images = tmp_path / "new" / "batch1"
    images.mkdir(parents=True)
    rows = []
    for i in range(3):
        Image.new("RGB", (100 + i, 80), (0, 0, 0)).save(images / f"img{i}.jpg")
        row = {"image_name": f"img{i}.jpg"}
        for k, point in enumerate(names):
            row.update({f"{point}_x": 10 + k, f"{point}_y": 20, f"{point}_v": 2})
        rows.append(row)
    rows.append({**rows[0], "image_name": "absent.jpg"})
    csv = tmp_path / "annotations.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    data = load_finetune_data(tmp_path / "new", [csv], group="Diptera")
    assert data.orders == ["diptera"]
    assert list(data.table["image_name"]) == ["batch1/img0.jpg", "batch1/img1.jpg",
                                              "batch1/img2.jpg"]      # absent.jpg left out
    assert list(data.table["width"]) == [100, 101, 102]
    with pytest.raises(ContractError, match="Unknown insect order"):
        load_finetune_data(tmp_path / "new", [csv])                    # no group anywhere


def _produce_cfg(config_factory, retained: Path, images_dir: Path, output: Path,
                 *extra: str):
    return config_factory([
        "approach=yolo_pooled", "train.device=cpu", "mode=full",   # mode=smoke never exports
        f"paths.retained={retained.parent}", "finetune.epochs=1",
        f"finetune.images_dir={images_dir.as_posix()}", f"finetune.output={output.as_posix()}",
        *extra,
    ])


def test_produce_writes_a_new_ensemble(fake_ultralytics, config_factory, project, retained,
                                       tmp_path) -> None:
    names = list(load_schema("insect42_v1", project.configs).names)
    images_dir = _finetune_images(tmp_path, names)
    (retained.parent / "scale_bar").mkdir()
    (retained.parent / "scale_bar" / "best.pt").write_bytes(b"scale bar")
    base_before = {p: p.read_bytes() for p in retained.rglob("*") if p.is_file()}
    output = tmp_path / "my_models"

    out_pose = cmd_finetune(_produce_cfg(config_factory, retained, images_dir, output,
                                         "finetune.base=abc123"))

    members = sorted(p.parent.name for p in out_pose.glob("*/model_card.json"))
    assert len(members) == 2                       # one per base member of abc123
    ensemble = json.loads((out_pose / "ensemble.json").read_text())
    assert ensemble["source"] == "finetune" and ensemble["n_members"] == 2
    assert sorted(ensemble["base_members"]) == ["run_fold0", "run_fold1"]
    cards = [json.loads(p.read_text()) for p in out_pose.glob("*/model_card.json")]
    assert {c["finetune"]["base_member"] for c in cards} == {"run_fold0", "run_fold1"}
    assert all(c["finetune"]["n_images"] == N_IMAGES for c in cards)
    assert all(list(p.parent.glob("*.pt")) for p in out_pose.glob("*/model_card.json"))
    assert (output / "scale_bar" / "best.pt").read_bytes() == b"scale bar"
    trains = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert len(trains) == 2 and all(c["epochs"] == 1 for c in trains)
    # The base models are untouched.
    assert {p: p.read_bytes() for p in retained.rglob("*") if p.is_file()} == base_before


def test_produce_safeguards(config_factory, project, retained, tmp_path) -> None:
    names = list(load_schema("insect42_v1", project.configs).names)
    images_dir = _finetune_images(tmp_path, names)
    # The base models / the workspace models are never the output.
    with pytest.raises(SystemExit, match="never overwrites"):
        cmd_finetune(_produce_cfg(config_factory, retained, images_dir, retained.parent))
    # A non-empty output needs --overwrite.
    busy = tmp_path / "busy"
    (busy / "pose" / "old").mkdir(parents=True)
    with pytest.raises(SystemExit, match="--overwrite"):
        cmd_finetune(_produce_cfg(config_factory, retained, images_dir, busy))
    # No output and no --evaluate.
    cfg = config_factory([f"paths.retained={retained.parent}",
                          f"finetune.images_dir={images_dir.as_posix()}"])
    with pytest.raises(SystemExit, match="--output"):
        cmd_finetune(cfg)
