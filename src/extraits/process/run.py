"""
run.py
======

`extraits process`: runs the pose ensemble, the scale extraction and the
measurement-validity classifiers over every image of a source, and writes one
CSV row per image (measurements in px and mm, per-measurement confidences and
validity, overall pose confidence, scale and scale confidence).

Two interchangeable sources (--source):
    folder   a local folder of images, searched recursively (default)
    hf       a Hugging Face dataset repo, streamed into RAM

Parallelism: a pool of worker threads runs "read/download + decode + full
pipeline", with a bounded number of images in flight; rows are written as they
complete. Each thread holds its own model copies (worker.py). The number of
threads, their device(s) and compute threads are sized from the machine by
hardware.py; --workers / --buffer / --torch-threads force a value.

Examples
--------
    extraits process --input photos/ --output results.csv
    extraits process --input photos/ --group coleoptera
    extraits process --input photos/ --scale none --no-measurement-classifier
    extraits process --source hf --dataset <user>/<dataset> --hf-folders 1 2

Every other setting is in extraits/process/config.py. No model is trained.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# --scale value -> (config.USE_SCALE_BAR, config.USE_RULER_FALLBACK)
SCALE_METHODS = {
    "auto": (True, True),
    "scale_bar": (True, False),
    "ruler": (False, True),
    "none": (False, False),
}


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Options of `extraits process`."""
    src = p.add_argument_group("images")
    src.add_argument("--source", choices=["folder", "hf"], default="folder",
                     help="Where images come from: a local 'folder' (default) or a Hugging "
                          "Face dataset ('hf').")
    src.add_argument("--input", "-i", type=str, default=None,
                     help="[folder] images folder, searched recursively.")
    src.add_argument("--dataset", type=str, default=os.environ.get("HF_DATASET"),
                     help="[hf] dataset repo id (default: env HF_DATASET).")
    src.add_argument("--hf-folders", nargs="*", default=None,
                     help="[hf] restrict to these sub-folders (default: the whole repo).")
    src.add_argument("--hf-token", type=str, default=None,
                     help="[hf] token for private repos (default: env HF_TOKEN).")
    src.add_argument("--image-list", type=str, default=None,
                     help="Only process the images named in this file: one name per line "
                          "(.txt), an `image_name` column (.csv) or the keys of a .json.")

    grp = p.add_argument_group("insect order (input of the validity classifiers)")
    grp.add_argument("--group", type=str, default=None,
                     help="Insect order of every image (coleoptera, diptera, hymenoptera, "
                          "lepidoptera).")
    grp.add_argument("--groups-csv", type=str, default=None,
                     help="CSV with columns image_name, group: the order of each image. "
                          "Without --group/--groups-csv, a folder named after an order in "
                          "the image path is used, e.g. <input>/coleoptera/img.jpg.")

    out = p.add_argument_group("output and models")
    out.add_argument("--output", "-o", type=str, default=None,
                     help="Output CSV (default: extraits_results.csv in the current folder).")
    out.add_argument("--models-dir", type=str, default=None,
                     help="Models folder holding pose/, scale_bar/ and measurement_validity/ "
                          "(default: env EXTRAITS_MODELS, else <workspace>/models).")
    out.add_argument("--pose-models", type=str, default=None,
                     help="Another pose ensemble: a folder (every *.pt under it) or a single "
                          ".pt, e.g. a fine-tuned model (default: <models-dir>/pose).")

    steps = p.add_argument_group("optional steps")
    steps.add_argument("--scale", choices=list(SCALE_METHODS), default=None,
                       help="Scale extraction: 'auto' = scale bar, then ruler as a fallback; "
                            "'scale_bar' or 'ruler' alone; 'none' skips it (no millimetre "
                            "values). Default: auto.")
    steps.add_argument("--all-scale-methods", action="store_true",
                       help="Run the ruler even when the scale bar succeeded, to keep the raw "
                            "confidence of both methods (evaluation of the scale extraction).")
    steps.add_argument("--no-measurement-classifier", action="store_true",
                       help="Skip the measurement-validity classifiers.")

    hw = p.add_argument_group("hardware")
    hw.add_argument("--workers", type=int, default=None,
                    help="Parallel worker threads (default: sized from the GPUs, CPUs and "
                         "free memory).")
    hw.add_argument("--buffer", type=int, default=None,
                    help="Max images in flight (default: 2 x workers).")
    hw.add_argument("--torch-threads", type=int, default=None,
                    help="Compute threads PER worker (default: cpus // workers).")
    hw.add_argument("--flush-every", type=int, default=None,
                    help="Force the CSV to disk every N rows (default: "
                         "config.CSV_FLUSH_EVERY_N_ROWS; 0 = only at the end).")


def run(args: argparse.Namespace) -> int:
    """Run `extraits process` with parsed arguments. Returns the exit code."""
    # Imported here so `extraits --help` stays fast.
    from extraits import workspace

    from . import config
    from .csv_writer import CsvWriter
    from .hardware import apply_threads, plan_resources, set_plan
    from .image_source import build_source
    from .insect_group import build_group_index
    from .parallel import bounded_unordered_map
    from .pose_inference import pose_model_paths
    from .worker import make_task

    if args.source == "folder" and not args.input:
        raise SystemExit("extraits process: --input <folder> is required with --source folder.")

    config.use_models_dir(workspace.models_dir(args.models_dir, getattr(args, "workspace", None)))
    if args.pose_models:
        config.POSE_MODELS_DIR = Path(args.pose_models).expanduser().resolve()
    output_csv = Path(args.output) if args.output else config.OUTPUT_CSV

    # Optional steps: set before the workers start, since they read the config.
    if args.scale is not None:
        config.USE_SCALE_BAR, config.USE_RULER_FALLBACK = SCALE_METHODS[args.scale]
    if args.no_measurement_classifier:
        config.RUN_MEASUREMENT_CLASSIFIER = False
    config.RUN_ALL_SCALE_METHODS = args.all_scale_methods

    pose_models = pose_model_paths()
    if not pose_models:
        raise SystemExit(
            f"No pose model (*.pt) under {config.POSE_MODELS_DIR}.\n"
            f"Download the official models with `extraits download-models`, or point "
            f"--models-dir / --pose-models at your own.")
    if config.USE_SCALE_BAR and not config.SCALE_BAR_MODEL_PATH.is_file():
        raise SystemExit(
            f"Scale-bar detector not found: {config.SCALE_BAR_MODEL_PATH}.\n"
            f"Download it with `extraits download-models`, or run with --scale ruler "
            f"(ruler only) or --scale none.")

    # ---- the image source (list of items + a per-item loader) ---------------
    items, load_fn = build_source(
        args.source,
        folder=args.input,
        repo=args.dataset,
        hf_folders=args.hf_folders,
        hf_token=args.hf_token,
        image_list=args.image_list,
    )
    total = len(items)

    # ---- size the run to the machine (GPU(s), CPUs, free memory) ------------
    model_files = list(pose_models)
    if config.USE_SCALE_BAR:
        model_files.append(config.SCALE_BAR_MODEL_PATH)
    plan = plan_resources(model_files, workers=args.workers, threads=args.torch_threads,
                          buffer=args.buffer, source=args.source)
    set_plan(plan)
    apply_threads(plan)

    print("=" * 70)
    print(f"Source        : {args.source}"
          + (f"  ({args.input})" if args.source == "folder"
             else f"  ({args.dataset}, folders={args.hf_folders or 'ALL'})"))
    print(f"Output CSV    : {output_csv}")
    print(f"Pose ensemble : {len(pose_models)} model(s) under {config.POSE_MODELS_DIR}")
    print(f"Confidence    : {config.MEASUREMENT_CONFIDENCE_SIGNAL}")
    scale_steps = [name for name, on in (("scale bar", config.USE_SCALE_BAR),
                                         ("ruler", config.USE_RULER_FALLBACK)) if on]
    print(f"Scale         : {' then '.join(scale_steps) if scale_steps else 'disabled'}")
    print(f"Validity clf  : {'on' if config.RUN_MEASUREMENT_CLASSIFIER else 'disabled'}")
    print(f"Hardware      : {plan.describe()}")
    print(f"Images        : {total}")
    print("=" * 70)

    # ---- shared, read-only context ------------------------------------------
    group_index = build_group_index([item[2] for item in items], group=args.group,
                                    groups_csv=args.groups_csv)
    measurement_classifiers = None
    if config.RUN_MEASUREMENT_CLASSIFIER:
        from extraits.validity.predict import load_measurement_classifiers
        measurement_classifiers = load_measurement_classifiers(
            config.MEASUREMENT_CLASSIFIER_DIR, config.MEASUREMENT_CLASSIFIER_METRICS_CSV)
    task = make_task(load_fn, measurement_classifiers, group_index)

    # ---- process in parallel, write results as they complete ----------------
    ok, err = 0, 0
    start_time = time.time()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with CsvWriter(output_csv, flush_every=args.flush_every) as writer:
        for i, (item, record, error) in enumerate(
                bounded_unordered_map(task, items, plan.workers, plan.buffer), start=1):
            image_name = item[1]
            if error is not None:
                err += 1
                print(f"[error] {image_name}: {error}")
                # Still emit a row so the failed image is visible in the CSV.
                writer.write_record({"image_name": image_name})
            else:
                ok += 1
                writer.write_record(record)

            if i % 50 == 0 or i == total:
                elapsed = time.time() - start_time
                print(f"    {i}/{total} done in {elapsed:.1f}s ({i / elapsed:.1f} img/s) "
                      f"(ok={ok}, err={err})")

    print(f"\nDone. {ok} processed, {err} error(s). Results written to: {output_csv}")
    return 0 if err == 0 or ok > 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="extraits process", description=__doc__.split("\n\n")[1])
    add_arguments(p)
    return run(p.parse_args(argv))
