"""`extraits train <model>`: train the models of EXTRAITS.

`pose` needs `pip install extraits[train]`; `validity` needs nothing more than the core.

    extraits train pose               prepare the annotations, build the folds, train the
                                      pose model on each fold and export the ensemble to
                                      <models>/pose/ (the folder `extraits process` reads)
    extraits train pose --optimise    same, after an Optuna search of the hyperparameters
    extraits train validity           train the measurement-validity classifiers on the
                                      measurement statuses of the annotation table and
                                      write them to <models>/measurement_validity/

Inputs (in the workspace): annotations/annotation_data.csv (and images/<order>/ for
the pose).
Outputs: data/pose/ (prepared tables, folds), runs/ (one folder per run),
<models>/pose/ (the retained ensemble, REPLACED by each command; models of another
command stay in runs/). `extraits pose <verb>` gives access to each step.
"""

from __future__ import annotations

import argparse

DEFAULT_EXPERIMENT = "exp_a_yolo_pooled"


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.epilog = __doc__.split("\n\n", 1)[1]
    sub = p.add_subparsers(dest="model", metavar="<model>", required=True)

    pose = sub.add_parser("pose", help="Pose ensemble (YOLO-pose, 42 keypoints).",
                          description="Train the pose ensemble.")
    pose.add_argument("--optimise", "--optimize", action="store_true",
                      help="Search the hyperparameters with Optuna first (tuning.* of the "
                           "config: budget of trials, nested or once), then train each fold.")
    pose.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                      help=f"Frozen experiment config (default: {DEFAULT_EXPERIMENT}).")
    pose.add_argument("--folds", default=None,
                      help="Outer folds to train: 'all', or a list like 0,2 (default: fold 0 "
                           "only for a plain training, every fold with --optimise).")
    pose.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 100).")
    pose.add_argument("--tag", default=None, help="Label of the runs (part of the run_id).")
    pose.add_argument("--no-export", action="store_true",
                      help="Keep the models in runs/ only; leave <models>/pose/ untouched.")
    pose.add_argument("--skip-prepare", action="store_true",
                      help="Reuse the prepared annotations and folds of data/pose/.")
    # Also accepted after `pose`; SUPPRESS keeps the value given before it.
    pose.add_argument("--workspace", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    pose.add_argument("--models-dir", default=None,
                      help="Models folder (default: $EXTRAITS_MODELS, else <workspace>/models).")
    pose.add_argument("overrides", nargs="*", metavar="key=value",
                      help="Extra Hydra overrides, e.g. train.batch_size=8 tuning.n_trials=40.")

    validity = sub.add_parser(
        "validity", help="Measurement-validity classifiers (one random forest per measurement).",
        description="Train the measurement-validity classifiers (no extra needed).")
    validity.add_argument("--workspace", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    from extraits.validity.cli import add_arguments as add_validity_arguments

    add_validity_arguments(validity)


def pose_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides equivalent to the options of `extraits train pose`."""
    out = [f"experiment={args.experiment}"]
    if args.folds is not None:
        folds = args.folds.strip()
        out.append("folds=all" if folds == "all" else f"folds=[{folds.strip('[]')}]")
    if args.epochs is not None:
        out.append(f"train.epochs={args.epochs}")
    if args.tag is not None:
        out.append(f"tag={args.tag}")
    if args.no_export:
        out.append("retain.enabled=false")
    return [*out, *args.overrides]


def run(args: argparse.Namespace) -> int:
    if args.model == "validity":
        from extraits.validity.cli import run as run_validity

        return run_validity(args)

    from extraits.pose.cli import print_result, run_verb

    workspace = getattr(args, "workspace", None)
    if args.model == "pose":
        overrides = pose_overrides(args)
        if not args.skip_prepare:
            run_verb("prepare", overrides, workspace, args.models_dir)
            run_verb("split", overrides, workspace, args.models_dir)
        print_result(run_verb("tune" if args.optimise else "train", overrides, workspace,
                              args.models_dir))
    return 0
