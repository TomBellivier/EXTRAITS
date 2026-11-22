"""`extraits download-models`: fetch the official models from the Hugging Face Hub.

The models are downloaded from a pinned revision (a commit hash, which cannot
change) into a models folder with the layout `extraits process` reads:

    <models>/pose/<run_id>/best.pt (+ model_card.json), pose/ensemble.json
    <models>/scale_bar/best.pt
    <models>/measurement_validity/rf_related_<measure>.joblib + metrics.csv

The repository, the revision and the sha256 of every file come from the manifest
shipped with the package (extraits/models.yaml, written by tools/upload_models.py).
Every file is checked against its sha256 after download; a file that does not
match is deleted and the command fails. Files already present and valid are not
downloaded again.

The models are licensed under AGPL-3.0 (they derive from Ultralytics YOLO); the
EXTRAITS code is MIT. See the model card on the Hub.

EasyOCR, used to read the scale-bar value, downloads its own weights (about
100 MB) the first time it runs. `--with-ocr` fetches them now, for offline use or
a Docker image.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from extraits import workspace

MANIFEST_PATH = Path(__file__).with_name("models.yaml")

# Top-level folder of the models folder -> component name used by --only.
COMPONENTS: dict[str, str] = {
    "pose": workspace.POSE_SUBDIR,
    "scale_bar": str(workspace.SCALE_BAR_SUBPATH.parent),
    "measurement_validity": workspace.VALIDITY_SUBDIR,
}


@dataclass
class Manifest:
    repo_id: str
    revision: str | None
    tag: str | None = None
    # relative path (posix) -> {"sha256": str, "size": int}
    files: dict[str, dict] = field(default_factory=dict)

    def select(self, components: list[str] | None) -> dict[str, dict]:
        """The files of the given components (all files when None)."""
        if not components:
            return dict(self.files)
        prefixes = tuple(COMPONENTS[c] + "/" for c in components)
        return {p: info for p, info in self.files.items() if p.startswith(prefixes)}


def load_manifest(path: str | Path = MANIFEST_PATH) -> Manifest:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Manifest(repo_id=data["repo_id"], revision=data.get("revision"),
                    tag=data.get("tag"), files=dict(data.get("files") or {}))


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_files(models_dir: Path, files: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Split `files` into (missing, corrupted) relative paths under `models_dir`."""
    missing, corrupted = [], []
    for rel, info in sorted(files.items()):
        path = models_dir / rel
        if not path.is_file():
            missing.append(rel)
        elif path.stat().st_size != info["size"] or sha256_of(path) != info["sha256"]:
            corrupted.append(rel)
    return missing, corrupted


def download_models(models_dir: str | Path, *, components: list[str] | None = None,
                    manifest: Manifest | None = None, repo_id: str | None = None,
                    revision: str | None = None, token: str | None = None,
                    force: bool = False) -> Path:
    """Download the official models into `models_dir` and verify them. Returns the folder.

    `repo_id` / `revision` override the manifest (a mirror, or another release).
    The sha256 checks then apply only if the revision is the manifest's one.
    """
    manifest = manifest or load_manifest()
    files = manifest.select(components)
    repo_id = repo_id or manifest.repo_id
    pinned = revision is None or revision in (manifest.revision, manifest.tag)
    revision = revision or manifest.revision
    if not revision or not files:
        raise RuntimeError(
            "No official models are published for this version of EXTRAITS yet "
            f"(empty manifest: {MANIFEST_PATH}). Point --models-dir at your own models.")

    models_dir = Path(models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    if force:
        to_get = sorted(files)
    else:
        missing, corrupted = check_files(models_dir, files)
        to_get = missing + corrupted
        if not to_get:
            print(f"All {len(files)} model files are already in {models_dir} and valid.")
            return models_dir

    from huggingface_hub import snapshot_download

    print(f"Downloading {len(to_get)} file(s) from {repo_id}@{manifest.tag or revision} "
          f"into {models_dir} ...")
    for rel in to_get:
        (models_dir / rel).unlink(missing_ok=True)
    snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=to_get,
                      local_dir=models_dir, token=token)

    if not pinned:
        print(f"Warning: revision {revision!r} is not the one pinned by this version of "
              f"EXTRAITS ({manifest.tag or manifest.revision}); the checksums were not verified.")
        return models_dir
    missing, corrupted = check_files(models_dir, {rel: files[rel] for rel in to_get})
    for rel in corrupted:
        (models_dir / rel).unlink(missing_ok=True)
    if missing or corrupted:
        raise RuntimeError(
            "Model download incomplete or corrupted (deleted, re-run the command):\n  "
            + "\n  ".join(missing + corrupted))
    print(f"Done: {len(files)} model files verified in {models_dir}.")
    return models_dir


def prefetch_ocr_weights() -> None:
    """Download the EasyOCR weights used to read scale bars (to ~/.EasyOCR by default)."""
    import easyocr

    from extraits.scale.scale_bar import OCR_LANGUAGES

    easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
    print("EasyOCR weights are ready.")


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models-dir", default=None,
                   help="Where to put the models (default: $EXTRAITS_MODELS, else "
                        "<workspace>/models).")
    p.add_argument("--only", nargs="+", choices=sorted(COMPONENTS), default=None,
                   help="Download only these components (default: all).")
    p.add_argument("--check", action="store_true",
                   help="Only check the models already in the folder; download nothing.")
    p.add_argument("--force", action="store_true",
                   help="Download every file again, even the valid ones.")
    p.add_argument("--with-ocr", action="store_true",
                   help="Also download the EasyOCR weights (otherwise fetched on first use).")
    p.add_argument("--repo-id", default=None,
                   help="Another Hugging Face repository (a mirror or a fork).")
    p.add_argument("--revision", default=None,
                   help="Another revision (tag or commit) than the one pinned by this "
                        "version of EXTRAITS; its checksums are not verified.")
    p.add_argument("--hf-token", default=None,
                   help="Hugging Face token (default: $HF_TOKEN or the cached login).")


def run(args: argparse.Namespace) -> int:
    models_dir = workspace.models_dir(args.models_dir, getattr(args, "workspace", None))
    if args.check:
        manifest = load_manifest()
        files = manifest.select(args.only)
        if not files:
            print("The manifest lists no model files: nothing to check.")
            return 1
        missing, corrupted = check_files(models_dir, files)
        for rel in missing:
            print(f"missing   {rel}")
        for rel in corrupted:
            print(f"corrupted {rel}")
        if missing or corrupted:
            print(f"{len(missing)} missing, {len(corrupted)} corrupted, in {models_dir}. "
                  f"Run `extraits download-models` to fix.")
            return 1
        print(f"All {len(files)} model files in {models_dir} are valid.")
        return 0
    try:
        download_models(models_dir, components=args.only, repo_id=args.repo_id,
                        revision=args.revision, token=args.hf_token, force=args.force)
    except RuntimeError as e:
        raise SystemExit(f"extraits download-models: {e}") from None
    if args.with_ocr:
        prefetch_ocr_weights()
    return 0
