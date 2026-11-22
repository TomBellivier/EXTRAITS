"""`extraits finetune`: adapt the pose models to your own annotated images.

    extraits finetune --images <dir> --annotations <export.json|table.csv ...> --output <dir>
        fine-tune every model of the base ensemble on ALL your images and write the new
        ensemble to <dir>/pose/ (plus copies of scale_bar/ and measurement_validity/),
        then measure with: extraits process -i <images> --models-dir <dir>
    extraits finetune --evaluate --images <dir> --annotations ...
        estimate the gain first: k-fold before/after comparison of one base model,
        written to <workspace>/results/pose/finetune/<name>/ (nothing is exported)

Annotations: Label Studio JSON exports, or CSV files in the columns of
annotation_data.csv (image_name, group, <keypoint>_x, <keypoint>_y, <keypoint>_v; see
docs/finetune.md). Without --annotations, --images must follow the layout
<order>/{*.json, images/}. The base ensemble (default: <models>/pose/, the official
models) is never modified. Needs `pip install extraits[train]`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.epilog = __doc__.split("\n\n", 1)[1]
    p.add_argument("--images", required=True,
                   help="Folder of the new images (searched recursively by file name).")
    p.add_argument("--annotations", nargs="+", default=None, metavar="FILE",
                   help="Label Studio JSON export(s) and/or CSV file(s). Default: the "
                        "<order>/*.json files of --images.")
    p.add_argument("--group", default=None,
                   help="Insect order of every image whose annotations carry none "
                        "(coleoptera, diptera, hymenoptera, lepidoptera).")
    p.add_argument("--output", default=None,
                   help="Models folder to write (required unless --evaluate). Must not be "
                        "the base models nor <workspace>/models.")
    p.add_argument("--evaluate", action="store_true",
                   help="Before/after k-fold comparison of one base model instead.")
    p.add_argument("--base-models", default=None,
                   help="Ensemble folder to start from (default: <models>/pose/).")
    p.add_argument("--base", default=None,
                   help="Members to start from: 'all' (default when producing), a member "
                        "folder name or a variant_hash. With --evaluate: the fold-0 member "
                        "by default.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs (default: those of the base model).")
    p.add_argument("--folds", type=int, default=None,
                   help="With --evaluate: number of folds (default: 10).")
    p.add_argument("--name", default=None,
                   help="Name of the work folders runs/finetune/<name> (default: the output "
                        "folder name, or the base variant_hash with --evaluate).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace <output>/pose/ when it is not empty.")
    p.add_argument("--pose-only", action="store_true",
                   help="Do not copy scale_bar/ and measurement_validity/ to the output.")
    p.add_argument("--models-dir", default=None,
                   help="Models folder of the workspace (default: $EXTRAITS_MODELS, else "
                        "<workspace>/models).")
    p.add_argument("overrides", nargs="*", metavar="key=value",
                   help="Extra Hydra overrides, e.g. train.batch_size=8.")


def finetune_settings(args: argparse.Namespace) -> dict:
    """finetune.* config values equivalent to the options."""
    def path(value: str | None) -> str | None:
        return str(Path(value).expanduser().resolve()) if value else None

    if not args.evaluate and not args.output:
        raise SystemExit("extraits finetune: --output <folder> is required (or --evaluate).")
    settings = {
        "evaluate": bool(args.evaluate),
        "images_dir": path(args.images),
        "annotations": [path(a) for a in args.annotations] if args.annotations else None,
        "group": args.group.lower() if args.group else None,
        "base_models": path(args.base_models),
        "base": args.base,
        "output": path(args.output),
        "name": args.name,
        "overwrite": bool(args.overwrite),
        "copy_other_models": not args.pose_only,
        "epochs": args.epochs,
    }
    if args.folds is not None:
        settings["n_folds"] = int(args.folds)
    return settings


def run(args: argparse.Namespace) -> int:
    from extraits.pose.cli import print_result, run_verb

    settings = finetune_settings(args)
    print_result(run_verb("finetune", list(args.overrides), getattr(args, "workspace", None),
                          args.models_dir, settings={f"finetune.{k}": v
                                                     for k, v in settings.items()}))
    return 0
