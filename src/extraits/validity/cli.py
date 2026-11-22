"""Command line of `extraits train validity` (light: the training imports happen in run)."""

from __future__ import annotations

import argparse

from extraits import workspace

MIN_MINORITY = 20   # same defaults as extraits.validity.train
N_FOLDS = 5
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--annotation-data", default=None,
                   help="Annotation table (default: <workspace>/annotations/annotation_data.csv).")
    p.add_argument("--models-dir", default=None,
                   help="Models folder; the classifiers go to <models-dir>/measurement_validity/ "
                        "(default: $EXTRAITS_MODELS, else <workspace>/models). Its previous "
                        "classifiers are REPLACED.")
    p.add_argument("--results-dir", default=None,
                   help="Evaluation outputs (default: <workspace>/results/meas_classifier/training).")
    p.add_argument("--min-per-class", type=int, default=MIN_MINORITY,
                   help="A measurement gets a classifier only if each class (measurable / non "
                        f"measurable) has at least this many images (default: {MIN_MINORITY}).")
    p.add_argument("--n-folds", type=int, default=N_FOLDS,
                   help=f"Stratified folds of the out-of-fold evaluation (default: {N_FOLDS}).")


def run(args: argparse.Namespace) -> int:
    ws = getattr(args, "workspace", None)
    annotation_data = args.annotation_data or workspace.annotation_table(ws)
    models_dir = workspace.models_dir(args.models_dir, ws) / workspace.VALIDITY_SUBDIR
    out_dir = args.results_dir or workspace.results_dir(ws) / "meas_classifier" / "training"
    from extraits.validity.train import run_training

    run_training(annotation_data, models_dir, out_dir, int(args.min_per_class), int(args.n_folds))
    return 0
