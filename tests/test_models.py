"""Tests of `extraits download-models` (the Hub is mocked: no network)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

from extraits import models
from extraits.cli import main

FAKE_FILES = {
    "pose/m0/best.pt": b"pose weights",
    "pose/ensemble.json": b"{}",
    "scale_bar/best.pt": b"scale bar weights",
    "measurement_validity/rf_related_head width.joblib": b"forest",
    "measurement_validity/metrics.csv": b"measure\n",
}


def _manifest(files: dict[str, bytes] = FAKE_FILES) -> models.Manifest:
    return models.Manifest(
        repo_id="someone/extraits-models", revision="abc123", tag="v0.0.1",
        files={rel: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
               for rel, data in files.items()})


class FakeHub:
    """Stands for huggingface_hub.snapshot_download: writes FAKE_FILES (or `served`)."""

    def __init__(self):
        self.calls: list[dict] = []
        self.served: dict[str, bytes] = {}  # rel -> bytes served instead of FAKE_FILES

    def __call__(self, repo_id, revision, allow_patterns, local_dir, token=None):
        self.calls.append({"repo_id": repo_id, "revision": revision,
                           "files": list(allow_patterns)})
        for rel in allow_patterns:
            path = Path(local_dir) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.served.get(rel, FAKE_FILES[rel]))
        return str(local_dir)


@pytest.fixture
def fake_hub(monkeypatch):
    import huggingface_hub

    hub = FakeHub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    return hub


def test_shipped_manifest_is_readable():
    m = models.load_manifest()
    assert m.repo_id.endswith("/extraits-models")
    for rel, info in m.files.items():
        assert rel.split("/")[0] in models.COMPONENTS.values()
        assert len(info["sha256"]) == 64 and info["size"] > 0


def test_select_components():
    m = _manifest()
    assert set(m.select(["scale_bar"])) == {"scale_bar/best.pt"}
    assert len(m.select(["pose", "measurement_validity"])) == 4
    assert m.select(None) == m.files


def test_download_then_skip_when_valid(tmp_path, fake_hub):
    out = models.download_models(tmp_path / "m", manifest=_manifest())
    assert (out / "scale_bar/best.pt").read_bytes() == FAKE_FILES["scale_bar/best.pt"]
    assert fake_hub.calls[0]["revision"] == "abc123" and len(fake_hub.calls[0]["files"]) == 5
    models.download_models(tmp_path / "m", manifest=_manifest())
    assert len(fake_hub.calls) == 1  # nothing downloaded the second time


def test_only_missing_or_corrupted_files_are_downloaded(tmp_path, fake_hub):
    models.download_models(tmp_path, manifest=_manifest())
    (tmp_path / "pose/m0/best.pt").write_bytes(b"tampered")
    (tmp_path / "scale_bar/best.pt").unlink()
    models.download_models(tmp_path, manifest=_manifest())
    assert sorted(fake_hub.calls[1]["files"]) == ["pose/m0/best.pt", "scale_bar/best.pt"]
    assert models.check_files(tmp_path, _manifest().files) == ([], [])


def test_corrupted_download_is_deleted_and_fails(tmp_path, fake_hub):
    fake_hub.served["scale_bar/best.pt"] = b"not the published file"
    with pytest.raises(RuntimeError, match="scale_bar/best.pt"):
        models.download_models(tmp_path, manifest=_manifest())
    assert not (tmp_path / "scale_bar/best.pt").exists()
    assert (tmp_path / "pose/m0/best.pt").exists()


def test_unpinned_revision_skips_the_checksums(tmp_path, fake_hub, capsys):
    models.download_models(tmp_path, manifest=_manifest(), revision="v9")
    assert fake_hub.calls[0]["revision"] == "v9"
    assert "not verified" in capsys.readouterr().out


def test_empty_manifest_gives_a_clear_error(tmp_path):
    empty = models.Manifest(repo_id="someone/extraits-models", revision=None)
    with pytest.raises(RuntimeError, match="No official models"):
        models.download_models(tmp_path, manifest=empty)


def test_cli_check(tmp_path, monkeypatch, fake_hub):
    monkeypatch.setattr(models, "load_manifest", lambda *a, **k: _manifest())
    assert main(["download-models", "--models-dir", str(tmp_path), "--check"]) == 1
    assert main(["download-models", "--models-dir", str(tmp_path), "--only", "scale_bar"]) == 0
    assert main(["download-models", "--models-dir", str(tmp_path), "--check",
                 "--only", "scale_bar"]) == 0
    assert fake_hub.calls[0]["files"] == ["scale_bar/best.pt"]


def test_upload_script_dry_run(tmp_path, monkeypatch):
    """The maintainer script builds the model card and the manifest without uploading."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    import upload_models

    src = tmp_path / "models"
    for rel, data in FAKE_FILES.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(data)
    (src / "pose/m0/model_card.json").write_text(
        '{"run_id": "m0", "fold": 0, "primary_metric": "oks_ap", "primary_value": 0.8,'
        ' "keypoint_names": ["a", "b"], "datasets": ["coleoptera"],'
        ' "source_run": "/home/someone/private/run", "git": {"commit": "x"}}')
    (src / "measurement_validity/metrics.csv").write_text(
        "measure,n,mcc,average_precision,balanced_accuracy\nhead width,10,0.5,0.6,0.7\n")
    out = tmp_path / "preview"
    assert upload_models.main(["--models-dir", str(src), "--tag", "v0.0.1",
                               "--dry-run", "--out", str(out)]) == 0
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: agpl-3.0") and "head width" in card
    manifest = yaml.safe_load((out / "models.yaml").read_text())
    assert manifest["tag"] == "v0.0.1"
    assert set(manifest["files"]) == set(FAKE_FILES) | {"pose/m0/model_card.json"}
    published = upload_models.payload(src, "pose/m0/model_card.json")
    assert b"/home/someone" not in published and b'"git"' not in published
    assert manifest["files"]["pose/m0/model_card.json"]["size"] == len(published)
    sys.modules.pop("upload_models", None)
