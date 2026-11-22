"""Annotation tools (`extraits annotate`): Label Studio -> CSV -> merged table -> check."""

from __future__ import annotations

import json

import pandas as pd

from extraits.cli import main
from extraits.kp_infos import KEYPOINT_NAMES, MEASUREMENTS

SIZE = 200


def _workspace(tmp_path, n: int = 3):
    """A workspace with Label Studio exports, images and measurement statuses."""
    from PIL import Image

    ann = tmp_path / "annotations"
    export_dir = ann / "label_studio_annotations" / "diptera"
    export_dir.mkdir(parents=True)
    (tmp_path / "images" / "diptera").mkdir(parents=True)
    tasks = []
    for i in range(n):
        name = f"fly{i}.jpg"
        Image.new("RGB", (SIZE, SIZE)).save(tmp_path / "images" / "diptera" / name)
        results = [{"type": "keypointlabels", "original_width": SIZE, "original_height": SIZE,
                    "value": {"keypointlabels": [p], "x": 10 + k, "y": 20 + (k % 5) * 3}}
                   for k, p in enumerate(KEYPOINT_NAMES)]
        tasks.append({"id": i, "data": {"img": f"/data/local-files/?d=images/{name}"},
                      "annotations": [{"id": i, "result": results}]})
    (export_dir / "batch1.json").write_text(json.dumps(tasks), encoding="utf-8")
    (ann / "meas_classifier").mkdir()
    pd.DataFrame([{"image_name": f"fly{i}.jpg",
                   **{f"{m}_status": "measurable" for m in MEASUREMENTS}}
                  for i in range(n)]).to_csv(ann / "meas_classifier" / "batch1_measurements.csv",
                                             index=False)
    return tmp_path


def test_to_csv_build_table_check(tmp_path, capsys):
    ws = str(_workspace(tmp_path))
    assert main(["annotate", "--workspace", ws, "to-csv"]) == 0
    pose = pd.read_csv(tmp_path / "annotations" / "pose" / "pose_annotations.csv")
    assert len(pose) == 3 and set(pose["group"]) == {"diptera"}
    assert (pose["width"] == SIZE).all()
    assert (pose[[f"{p}_v" for p in KEYPOINT_NAMES]] == 2).all().all()

    assert main(["annotate", "--workspace", ws, "build-table"]) == 0
    table = pd.read_csv(tmp_path / "annotations" / "annotation_data.csv")
    assert len(table) == 3
    assert (table[f"{MEASUREMENTS[0]}_status"] == "measurable").all()

    code = main(["annotate", "--workspace", ws, "check"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert (tmp_path / "results" / "annotation_check" / "issues.csv").is_file()
    assert "0 error(s)" in out


def test_check_finds_a_stale_table(tmp_path, capsys):
    ws = str(_workspace(tmp_path))
    main(["annotate", "--workspace", ws, "to-csv"])
    main(["annotate", "--workspace", ws, "build-table"])
    table_path = tmp_path / "annotations" / "annotation_data.csv"
    table = pd.read_csv(table_path)
    table.loc[0, "width"] = 999                     # edited by hand: out of date
    table.to_csv(table_path, index=False)
    main(["annotate", "--workspace", ws, "check", "--no-report"])
    assert "out of date" in capsys.readouterr().out


def test_import_volunteer_points_tasks_at_local_images(tmp_path):
    images = tmp_path / "imgs" / "sub"
    images.mkdir(parents=True)
    (images / "IMG_0093.png").write_bytes(b"x")
    export = tmp_path / "export.json"
    export.write_text(json.dumps([{"data": {"img": "/data/upload/3/95c6e06b-IMG_0093.png"}}]),
                      encoding="utf-8")
    assert main(["annotate", "import-volunteer", "--json-file", str(export),
                 "--image-folders", str(tmp_path / "imgs")]) == 0
    tasks = json.loads((tmp_path / "export_local.json").read_text(encoding="utf-8"))
    assert tasks[0]["data"]["img"].startswith("/data/local-files/?d=")
    assert tasks[0]["data"]["img"].endswith("IMG_0093.png")


def test_shipped_annotation_files():
    from extraits.annotation import ANNOTATION_PROTOCOL, LABEL_STUDIO_TEMPLATE

    template = LABEL_STUDIO_TEMPLATE.read_text(encoding="utf-8")
    assert all(f'value="{p}"' in template for p in KEYPOINT_NAMES)
    assert ANNOTATION_PROTOCOL.stat().st_size > 0
