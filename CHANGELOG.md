# Changelog

All notable changes to EXTRAITS. The format follows
[Keep a Changelog](https://keepachangelog.com); versions follow
[Semantic Versioning](https://semver.org).

## [0.1.0] - unreleased

First public version.

### Added
- `extraits process`: 42 keypoints by an ensemble of 5 YOLO-pose models, 27
  measurements in pixels and millimetres, scale from a scale bar (YOLO + EasyOCR) or a
  ruler (Fourier analysis), per-measurement validity classifiers, confidences, one CSV
  row per image; local folders or Hugging Face datasets.
- `extraits analyze`: figures and summary of a results CSV, real errors against an
  annotation table.
- `extraits download-models`: official models from the Hugging Face Hub, pinned
  revision, checksums verified (models v0.1.0).
- `extraits finetune`: fine-tune the pose ensemble on your own images (Label Studio
  exports or CSV), or estimate the gain first (`--evaluate`).
- `extraits train pose` / `extraits train validity`, `extraits report`, and
  `extraits pose <verb>` for every step of the training framework.
- `extraits annotate`: Label Studio launcher and import of external exports,
  conversion to CSV, measurement-validation desktop app, merged annotation table and
  its checks.
- `extraits run`: the whole chain from one settings file.
- Workspace folder for every data and model (`--workspace`, `EXTRAITS_WORKSPACE`).
- Docker images (CPU and CUDA), GitHub Actions (tests, images on GHCR).
- `research/`: the approaches and comparisons of the paper.
