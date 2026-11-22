"""Construction of the project paths (CONVENTIONS.md §2).

The ONLY module allowed to build paths. A `.py` that concatenates a hard-coded path
is a bug. No side effect at import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Single definition of the keypoints, the skeleton and the measurements of the WHOLE
# project: extraits/kp_infos.yaml. The `insect42_v1` schema and the measurements are
# read from it; `KP_INFOS` moves it.
from extraits.kp_infos import KP_INFOS_PATH  # noqa: E402



def pose_results_dir(workspace: str | Path | None = None) -> Path:
    """<workspace>/results/pose, for the scripts that have no Hydra config."""
    from extraits.workspace import results_dir

    return results_dir(workspace) / "pose"


@dataclass(frozen=True)
class ProjectPaths:
    """Roots of the project, resolved as absolute paths."""

    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    splits: Path
    runs: Path
    results: Path
    reports: Path
    configs: Path
    # Models folder (<workspace>/models by default): <retained>/pose/ is the ensemble
    # that `extraits process` reads.
    retained: Path
    # Root the `image_path` of the prepared annotations is relative to.
    images: Path
    # Annotation table read by the `annotation_csv` adapter.
    annotation_table: Path

    @classmethod
    def from_config(cls, cfg: Any) -> ProjectPaths:
        """Build the paths from the `paths` section of a Hydra config."""
        p = cfg.paths if hasattr(cfg, "paths") else cfg
        root = Path(str(p.root)).resolve()

        def sub(key: str, default: str) -> Path:
            value = getattr(p, key, None)
            # `.resolve()` on both sides: a default may go up outside the root
            # (retained), and a non-normalised path would break the comparisons.
            return Path(str(value)).resolve() if value is not None else (root / default).resolve()

        return cls(
            root=root,
            data=sub("data", "data"),
            raw=sub("raw", "data/raw"),
            interim=sub("interim", "data/interim"),
            processed=sub("processed", "data/processed"),
            splits=sub("splits", "data/splits"),
            runs=sub("runs", "runs"),
            results=sub("results", "results"),
            reports=sub("reports", "reports"),
            configs=sub("configs", "configs"),
            retained=sub("retained", "models"),
            images=sub("images", str(getattr(p, "data", None) or "data")),
            annotation_table=sub("annotation_table", "annotations/annotation_data.csv"),
        )

    @classmethod
    def default(cls, root: str | Path = ".") -> ProjectPaths:
        """Standard paths relative to a given root (images under data/, as in the tests)."""
        r = Path(root).resolve()
        return cls(
            root=r, data=r / "data", raw=r / "data/raw", interim=r / "data/interim",
            processed=r / "data/processed", splits=r / "data/splits", runs=r / "runs",
            results=r / "results", reports=r / "reports", configs=r / "configs",
            retained=r / "models", images=r / "data",
            annotation_table=r / "annotations" / "annotation_data.csv",
        )

    # --- artefacts -----------------------------------------------------------------
    def annotations(self, dataset: str) -> Path:
        """Contract 1: canonical annotations of a dataset."""
        return self.processed / dataset / "annotations.parquet"

    def raw_dir(self, dataset: str, subdir: str | None = None) -> Path:
        """IMMUTABLE source folder of a dataset."""
        return self.raw / (subdir or dataset)

    def split_file(self, split_id: str) -> Path:
        """Contract 2: fold table."""
        return self.splits / f"{split_id}.parquet"

    def split_meta(self, split_id: str) -> Path:
        """Metadata of the split (seed, strategy, content_hash)."""
        return self.splits / f"{split_id}.json"

    def run_dir(self, run_id: str) -> Path:
        """Root of the artefacts of a run. The ONLY place an approach writes to."""
        return self.runs / run_id

    def manifest(self, run_id: str) -> Path:
        """Contract 5. Its presence marks a complete run (§8.2)."""
        return self.run_dir(run_id) / "manifest.json"

    def predictions(self, run_id: str, split: str, fold: int) -> Path:
        """Contract 3."""
        return self.run_dir(run_id) / "predictions" / f"{split}_fold{fold}.parquet"

    def metrics(self, run_id: str) -> Path:
        """Contract 4."""
        return self.run_dir(run_id) / "metrics.parquet"

    def master_results(self) -> Path:
        """Aggregate of every run: the only source of the report tables (§8.4)."""
        return self.results / "master.parquet"

    def optuna_storage(self, study_name: str) -> Path:
        """Optuna database of a study (can be resumed)."""
        return self.runs / "optuna" / f"{study_name}.db"

    def keypoint_schema(self, name: str) -> Path:
        """Keypoint schema file (§3.1): kp_infos.yaml, or configs/keypoints/."""
        from extraits.pose.data.keypoints import schema_file

        return schema_file(name, self.configs)

    def retained_model(self, name: str, kind: str = "pose") -> Path:
        """Folder of a retained model, read directly by `extraits process`.

        `kind` is the model family ('pose' here; scale_bar/ and measurement_validity/
        are produced elsewhere).
        """
        return self.retained / kind / name

    def ensure_writable_dirs(self) -> None:
        """Create the folders the code may write to. Never touches `raw`."""
        for d in (self.interim, self.processed, self.splits, self.runs, self.results, self.reports):
            d.mkdir(parents=True, exist_ok=True)
