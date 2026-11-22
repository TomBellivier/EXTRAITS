"""`extraits run`: the whole chain, from the annotations to the analyses, in one command.

    extraits run --init               write a commented run.yaml in the workspace
    extraits run                      run the steps switched on in <workspace>/run.yaml
    extraits run --dry-run            print the commands, run nothing
    extraits run --only pose process  run these steps only

Steps, in order (each one can be switched off in run.yaml):

    annotations   extraits annotate to-csv + build-table
    check         extraits annotate check
    pose          extraits pose prepare / split / train or tune
    validity      extraits train validity
    process       extraits process on the images to measure
    analysis      extraits report, extraits analyze (+ research scripts when asked)

Each step runs the `extraits` command in its own process; every command and its output
are also written to <workspace>/results/run_logs/run_<date>.log. The settings and their
defaults are those of extraits/run_template.yaml.
"""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from extraits import workspace as ws

TEMPLATE = Path(__file__).with_name("run_template.yaml")
CONFIG_NAME = "run.yaml"
STEPS = ("annotations", "check", "pose", "validity", "process", "analysis")
CHOICES = {
    ("pose_training", "mode"): {"train", "optimise"},
    ("pose_training", "optimisation", "search"): {"tune_once", "nested"},
    ("process", "scale"): {"auto", "scale_bar", "ruler", "none"},
    ("process", "source"): {"folder", "hf"},
}


class StepFailed(Exception):
    """A step could not complete: the run stops, the reason is shown to the user."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def defaults() -> dict:
    """Every setting with its default value: the template itself."""
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


def merge(base: dict, values: dict, prefix: str, unknown: list[str]) -> dict:
    out = copy.deepcopy(base)
    for key, value in (values or {}).items():
        name = f"{prefix}{key}"
        if key not in base:
            unknown.append(name)
            out[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = merge(base[key], value, f"{name}.", unknown)
        else:
            out[key] = value
    return out


def load_settings(path: Path) -> tuple[dict, list[str]]:
    if not path.is_file():
        raise StepFailed(f"Settings file not found: {path}. Create one with "
                         f"`extraits run --init`.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown: list[str] = []
    cfg = merge(defaults(), raw, "", unknown)

    problems = []
    for keys, allowed in CHOICES.items():
        value = cfg
        for key in keys:
            value = value[key]
        if value not in allowed:
            problems.append(f"{'.'.join(keys)}: '{value}' is not one of {sorted(allowed)}")
    folds = cfg["pose_training"]["folds"]
    if not (folds == "all" or isinstance(folds, int)
            or (isinstance(folds, list) and folds and all(isinstance(f, int) for f in folds))):
        problems.append(f"pose_training.folds: {folds!r} must be a fold number, a list of "
                        "fold numbers (e.g. [0, 1, 2]) or \"all\"")
    if cfg["process"]["source"] == "hf" and not cfg["process"]["hf_dataset"]:
        problems.append("process.hf_dataset is required with process.source: hf")
    if problems:
        raise StepFailed("Invalid settings in " + path.name + ":\n  " + "\n  ".join(problems))
    return cfg, unknown


# --------------------------------------------------------------------------- #
# Command runner (console + log file)
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, log_path: Path | None, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        self.env = dict(os.environ)
        self.env.setdefault("PYTHONUNBUFFERED", "1")
        self.env.setdefault("PYTHONIOENCODING", "utf-8")
        self.env.setdefault("MPLBACKEND", "Agg")

    def echo(self, text: str = "") -> None:
        print(text, flush=True)
        if self.log is not None:
            self.log.write(text + "\n")
            self.log.flush()

    @staticmethod
    def shown(args: list[str]) -> str:
        args = ["python" if a == sys.executable else str(a) for a in args]
        return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)

    def run(self, args: list, check: bool = True) -> int:
        args = [str(a) for a in args]
        self.echo(f"$ {self.shown(args)}")
        if self.dry_run:
            return 0
        process = subprocess.Popen(args, env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            self.echo(line.rstrip("\n"))
        code = process.wait()
        if check and code != 0:
            raise StepFailed(f"Command failed (exit code {code}): {self.shown(args)}")
        return code

    def close(self) -> None:
        if self.log is not None:
            self.log.close()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
class Chain:
    """The steps, bound to one workspace and one runner."""

    def __init__(self, cfg: dict, workspace: Path, runner: Runner) -> None:
        self.cfg, self.workspace, self.run = cfg, workspace, runner
        self.extraits = [sys.executable, "-m", "extraits"]

    def path(self, value) -> Path:
        """A path of the settings: relative to the workspace, or absolute."""
        return (self.workspace / str(value)).expanduser().resolve()

    def cmd(self, command: str, *args) -> list:
        return [*self.extraits, command, "--workspace", str(self.workspace), *args]

    @property
    def table(self) -> Path:
        return ws.annotation_table(self.workspace)

    # --- annotations ---------------------------------------------------------
    def annotations(self) -> None:
        self.run.run(self.cmd("annotate", "to-csv"))
        self.run.run(self.cmd("annotate", "build-table"))

    def check(self) -> None:
        a = self.cfg["annotations"]
        args = self.cmd("annotate", "check", "--min-per-class",
                        self.cfg["validity"]["min_per_class"])
        if a["stop_on_warnings"]:
            args.append("--strict")
        if self.run.run(args, check=False) == 0:
            return
        if a["stop_on_errors"] or a["stop_on_warnings"]:
            raise StepFailed("The annotation check found problems (see above, and "
                             "results/annotation_check/issues.csv). Fix them, or set "
                             "annotations.stop_on_errors: false to go on anyway.")
        self.run.echo("/!\\ The annotation check found problems; going on "
                      "(annotations.stop_on_errors: false).")

    # --- training ------------------------------------------------------------
    def pose(self) -> None:
        p = self.cfg["pose_training"]
        common = [f"experiment={p['experiment']}"]
        self.run.run(self.cmd("pose", "prepare", *common))
        self.run.run(self.cmd("pose", "split", *common))
        folds = p["folds"]
        folds = "all" if folds == "all" else ",".join(
            str(int(f)) for f in ([folds] if isinstance(folds, int) else folds))
        overrides = [*common, "folds=all" if folds == "all" else f"folds=[{folds}]",
                     f"train.epochs={int(p['epochs'])}", f"train.device='{p['device']}'"]
        if p["tag"]:
            overrides.append(f"tag='{p['tag']}'")
        if p["force"]:
            overrides.append("force=true")
        if p["mode"] == "optimise":
            o = p["optimisation"]
            overrides += [f"tuning.mode={o['search']}", f"tuning.n_trials={int(o['n_trials'])}",
                          f"tuning.inner_folds={int(o['inner_folds'])}",
                          f"tuning.tuning_outer_fold={int(o['search_fold'])}",
                          f"tuning.final_full_fit={str(bool(o['final_full_fit'])).lower()}"]
        overrides += [str(o) for o in p["extra_overrides"]]
        self.run.run(self.cmd("pose", "train" if p["mode"] == "train" else "tune", *overrides))

    def validity(self) -> None:
        v = self.cfg["validity"]
        self.run.run(self.cmd("train", "validity", "--min-per-class", int(v["min_per_class"]),
                              "--n-folds", int(v["n_folds"])))

    # --- measuring -----------------------------------------------------------
    def process(self) -> None:
        p = self.cfg["process"]
        scale_bar = ws.models_dir(None, self.workspace) / ws.SCALE_BAR_SUBPATH
        if p["scale"] in ("auto", "scale_bar") and not scale_bar.is_file() and not self.run.dry_run:
            raise StepFailed(f"Scale-bar detector missing: {scale_bar}. Run `extraits "
                             "download-models`, or set process.scale: ruler (or none).")
        args = self.cmd("process", "--source", p["source"])
        if p["source"] == "folder":
            args += ["--input", self.path(p["input"])]
        else:
            args += ["--dataset", p["hf_dataset"]]
            if p["hf_folders"]:
                args += ["--hf-folders", *[str(f) for f in p["hf_folders"]]]
        args += ["--output", self.path(p["output"]), "--scale", p["scale"]]
        if not p["validity"]:
            args.append("--no-measurement-classifier")
        # Insect order of the annotated images (the images of the database are annotated).
        if self.table.is_file():
            args += ["--groups-csv", self.table]
        if p["pose_models"]:
            args += ["--pose-models", self.path(p["pose_models"])]
        if p["workers"]:
            args += ["--workers", int(p["workers"])]
        self.run.run(args)

    # --- analyses ------------------------------------------------------------
    def analysis(self) -> list[str]:
        """Every analysis runs even if another one fails: the failures are returned."""
        a, r = self.cfg["analysis"], self.cfg["research"]
        failed: list[str] = []

        def attempt(label: str, args: list) -> bool:
            if self.run.run(args, check=False) != 0:
                failed.append(label)
                self.run.echo(f"/!\\ {label} failed; the other analyses go on.")
                return False
            return True

        runs = self.workspace / "runs"
        has_runs = self.run.dry_run or any(runs.glob("*/manifest.json"))
        if a["pose_report"] and has_runs:
            attempt("pose report", self.cmd("report"))
        elif a["pose_report"]:
            self.run.echo(f"No complete pose run in {runs}: pose report skipped.")

        output = self.path(self.cfg["process"]["output"])
        if a["process_results"] and (self.run.dry_run or output.is_file()):
            args = [*self.extraits, "analyze", "--input", output,
                    "--output-dir", self.workspace / "results" / "pipeline" / output.stem,
                    "--review-threshold", float(a["review_threshold"])]
            if a["ground_truth"] and self.table.is_file():
                args += ["--annotation-data", self.table]
            elif not a["ground_truth"]:
                args.append("--no-gt")
            if a["annotated_copies"]:
                if self.cfg["process"]["source"] == "folder":
                    args += ["--print", "--images-dir", self.path(self.cfg["process"]["input"])]
                else:
                    self.run.echo("analysis.annotated_copies needs a local image folder: skipped.")
            attempt("process analysis", args)
        elif a["process_results"]:
            self.run.echo(f"No process output at {output}: its analysis is skipped.")

        scripts = {
            "pose_comparison": ("scripts/compare_models.py", ["--root", self.workspace]),
            "optuna_plots": ("scripts/plot_optuna.py", ["--root", self.workspace]),
            "validity_comparison": ("validity/compare_measure_validity_approaches.py",
                                    ["--workspace", self.workspace]),
            "ruler_evaluation": ("ruler/ruler_detection_evaluation.py", []),
        }
        wanted = [key for key in scripts if r[key]]
        if not has_runs and "pose_comparison" in wanted:
            wanted.remove("pose_comparison")
            self.run.echo("No complete pose run: pose comparison skipped.")
        if wanted:
            research = research_dir(r["research_dir"])
            if research is None:
                self.run.echo("/!\\ research/ not found (set research.research_dir): "
                              f"{', '.join(wanted)} skipped.")
            for key in wanted if research is not None else []:
                script, extra = scripts[key]
                attempt(key.replace("_", " "), [sys.executable, research / script, *extra])
        return failed


def research_dir(setting: str | None) -> Path | None:
    """The research/ folder of the repository: the setting, else the one holding the
    installed extraits-research package (an editable install)."""
    if setting:
        path = Path(setting).expanduser().resolve()
        return path if path.is_dir() else None
    try:
        import extraits_research
    except ImportError:
        return None
    path = Path(extraits_research.__file__).resolve().parents[2]
    return path if (path / "scripts").is_dir() else None


def plan_lines(cfg: dict) -> list[str]:
    p, v, proc = cfg["pose_training"], cfg["validity"], cfg["process"]
    pose = "off"
    if p["enabled"]:
        pose = (f"{p['mode']} {p['experiment']}, folds {p['folds']}, {p['epochs']} epochs, "
                f"device {p['device']}")
        if p["mode"] == "optimise":
            o = p["optimisation"]
            pose += f", {o['search']} search of {o['n_trials']} trials"
    source = proc["input"] if proc["source"] == "folder" else f"hf:{proc['hf_dataset']}"
    return [
        f"  annotation table     {'rebuilt' if cfg['annotations']['rebuild_table'] else 'as it is'}"
        f", check {'on' if cfg['annotations']['check'] else 'off'}",
        f"  pose training        {pose}",
        f"  validity classifiers {'retrained' if v['train'] else 'as they are'}",
        f"  process              {source if proc['enabled'] else 'off'} -> {proc['output']}"
        f" (scale {proc['scale']}, validity {'on' if proc['validity'] else 'off'})",
        f"  analyses             {'on' if cfg['analysis']['enabled'] else 'off'}",
    ]


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.epilog = __doc__.split("\n\n", 1)[1]
    p.add_argument("--config", default=None,
                   help=f"Settings file (default: <workspace>/{CONFIG_NAME}).")
    p.add_argument("--init", action="store_true",
                   help="Write a commented settings file (--config, else "
                        f"<workspace>/{CONFIG_NAME}) and stop.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands, run nothing.")
    p.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps.")
    p.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="Skip these steps.")


def run(args: argparse.Namespace) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # a console that cannot show a character
        except AttributeError:
            pass
    workspace = ws.workspace_dir(getattr(args, "workspace", None))
    config_path = Path(args.config).expanduser().resolve() if args.config \
        else workspace / CONFIG_NAME
    if args.init:
        if config_path.exists():
            print(f"{config_path} already exists: not overwritten.")
            return 1
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Settings written to {config_path}: edit it, then run `extraits run`.")
        return 0
    try:
        cfg, unknown = load_settings(config_path)
    except StepFailed as exc:
        print(f"ERROR: {exc}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None if args.dry_run else \
        (workspace / str(cfg["run"]["log_dir"])).resolve() / f"run_{stamp}.log"
    runner = Runner(log_path, args.dry_run)
    chain = Chain(cfg, workspace, runner)
    selected = [s for s in STEPS if (not args.only or s in args.only) and s not in args.skip]
    enabled = {   # what run.yaml switches on
        "annotations": bool(cfg["annotations"]["rebuild_table"]),
        "check": bool(cfg["annotations"]["check"]),
        "pose": bool(cfg["pose_training"]["enabled"]),
        "validity": bool(cfg["validity"]["train"]),
        "process": bool(cfg["process"]["enabled"]),
        "analysis": bool(cfg["analysis"]["enabled"]),
    }

    runner.echo("=" * 78)
    runner.echo(f"EXTRAITS run {stamp}{'  (dry run: nothing is executed)' if args.dry_run else ''}")
    runner.echo(f"Workspace: {workspace}   Settings: {config_path}")
    runner.echo("=" * 78)
    for line in plan_lines(cfg):
        runner.echo(line)
    for key in unknown:
        runner.echo(f"/!\\ Unknown setting '{key}' in {config_path.name} (typo?): ignored.")
    if log_path is not None and runner.log is not None:
        runner.echo(f"Log: {log_path}")
        runner.log.write("\n--- settings ---\n" + yaml.safe_dump(cfg, sort_keys=False) + "---\n")

    results: list[tuple[str, str, float]] = []
    failure = None
    for step in STEPS:
        if step not in selected:
            results.append((step, "skipped (--only / --skip)", 0.0))
            continue
        if not enabled[step]:
            results.append((step, "skipped (off in the settings)", 0.0))
            continue
        runner.echo("")
        runner.echo(f"===== {step} " + "=" * (70 - len(step)))
        start = time.time()
        try:
            outcome = getattr(chain, step)()
            status = "dry run" if args.dry_run else "done"
            if step == "analysis" and outcome:
                status = f"partial ({', '.join(outcome)} failed)"
                failure = failure or "analysis"
            results.append((step, status, time.time() - start))
        except StepFailed as exc:
            results.append((step, "FAILED", time.time() - start))
            runner.echo(f"\nERROR in step '{step}': {exc}")
            failure = step
            break

    runner.echo("")
    runner.echo("=" * 78)
    runner.echo("Summary")
    for step, status, seconds in results:
        duration = f"{seconds / 60:6.1f} min" if seconds >= 1 else ""
        runner.echo(f"  {step:12} {status:40} {duration}")
    if not args.dry_run:
        pose_dir = ws.models_dir(None, workspace) / ws.POSE_SUBDIR
        n_models = len(list(pose_dir.rglob("*.pt"))) if pose_dir.is_dir() else 0
        runner.echo(f"  pose ensemble: {n_models} model(s) in {pose_dir}")
        output = chain.path(cfg["process"]["output"])
        if output.is_file():
            runner.echo(f"  process output: {output}")
        runner.echo(f"  analyses: {workspace / 'results'}/   log: {log_path}")
    runner.close()
    return 1 if failure else 0
