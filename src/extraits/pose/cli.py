"""`extraits pose <verb> [key=value ...]`: every step of the pose framework (CONVENTIONS.md §5.4).

The overrides are standard Hydra overrides. `extraits train pose` and `extraits report`
(extraits/train.py) cover the usual use; this command gives access to every verb, for
the research and for re-evaluating a single run.

Paths: the workspace (--workspace / $EXTRAITS_WORKSPACE) and the models folder
(--models-dir / $EXTRAITS_MODELS) set paths.root and paths.retained (configs/paths.yaml).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from extraits.pose.registry import load_all_plugins, plugin_modules

if TYPE_CHECKING:
    from omegaconf import DictConfig

# The heavy imports (hydra, omegaconf, torch through pipeline) happen in the functions:
# `extraits --help` and `extraits process` must work without the `train` extra.

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
VERBS = ("prepare", "split", "train", "predict", "evaluate", "tune", "report", "finetune")

USAGE_EXAMPLES = """\
examples:
  extraits pose prepare
  extraits pose train experiment=exp_ref_mean_pose fold=0
  extraits pose train experiment=exp_a_yolo_pooled folds=[0,1,2]
  extraits pose tune experiment=exp_a_yolo_pooled folds=all
  extraits pose evaluate run_id=<run_id>
  extraits pose predict run_id=<run_id> split=test
  extraits pose finetune finetune.output=<dir>               (see `extraits finetune`)
  extraits pose finetune finetune.evaluate=true base=<variant_hash>
"""


def plugin_config_dirs() -> list[Path]:
    """Extra Hydra config folders of the installed plugins (e.g. research/)."""
    dirs = [getattr(m, "CONFIG_DIR", None) for m in plugin_modules()]
    return [Path(d) for d in dirs if d is not None]


def load_config(overrides: list[str], config_dir: Path | None = None) -> DictConfig:
    """Compose the Hydra configuration. No side effect.

    `config_dir` defaults to the configs shipped with the package; the configs of the
    installed plugins are added to the search path. paths.configs is set to `config_dir`
    unless an override sets it.
    """
    from hydra import compose, initialize_config_dir

    root = Path(config_dir or CONFIG_DIR).resolve()
    extra: list[str] = []
    searchpath = [f"file://{d}" for d in plugin_config_dirs()]
    if searchpath:
        extra.append(f"hydra.searchpath=[{','.join(searchpath)}]")
    if not any(o.startswith("paths.configs=") for o in overrides):
        extra.append(f"paths.configs={root}")
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="config", overrides=[*extra, *overrides])
    return cfg


def workspace_overrides(workspace: str | Path | None = None,
                        models_dir: str | Path | None = None) -> list[str]:
    """paths.root and paths.retained from the EXTRAITS workspace resolution."""
    from extraits import workspace as ws

    return [f"paths.root={ws.workspace_dir(workspace)}",
            f"paths.retained={ws.models_dir(models_dir, workspace)}"]


def _split_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Separate the Hydra overrides from the arguments specific to the CLI."""
    cli_args: dict[str, str] = {}
    overrides: list[str] = []
    for item in argv:
        if item.startswith(("run_id=", "split=")) and not item.startswith("split_id="):
            key, value = item.split("=", 1)
            cli_args[key] = value
        else:
            overrides.append(item)
    # `cv.fold=k` is a handy alias of `fold=k` (the current outer fold).
    overrides = [o.replace("cv.fold=", "fold=", 1) if o.startswith("cv.fold=") else o
                 for o in overrides]
    return overrides, cli_args


def run_verb(verb: str, argv: list[str], workspace: str | Path | None = None,
             models_dir: str | Path | None = None,
             settings: dict[str, Any] | None = None) -> Any:
    """Compose the config and dispatch to `pipeline`. Side effects: those of the step.

    `settings` ({dotted key: value}) are set after the composition: values such as
    paths with spaces or lists need no Hydra quoting there.
    """
    _require_train_extra()
    from extraits.pose import pipeline
    from extraits.pose.utils.logging import get_logger, setup_logging

    setup_logging()
    log = get_logger("cli")
    load_all_plugins()
    overrides, cli_args = _split_args(argv)
    if verb == "finetune":
        # The base models are yolo_pooled: its approach config is the default here.
        overrides = [o.replace("base=", "finetune.base=", 1) if o.startswith("base=") else o
                     for o in overrides]
        if not any(o.startswith("experiment=") for o in overrides):
            overrides.insert(0, "experiment=exp_a_yolo_pooled")
    cfg = load_config([*workspace_overrides(workspace, models_dir), *overrides])
    if settings:
        from omegaconf import OmegaConf

        for key, value in settings.items():
            OmegaConf.update(cfg, key, value, force_add=True)
    log.info("Verb '%s' | approach=%s | data=%s | fold=%s | workspace=%s",
             verb, cfg.approach.name, cfg.data.scope, cfg.fold, cfg.paths.root)

    if verb == "prepare":
        return pipeline.cmd_prepare(cfg)
    if verb == "split":
        return pipeline.cmd_split(cfg)
    if verb == "train":
        run_ids = [ctx.run_id for ctx in pipeline.cmd_train_folds(cfg)]
        return run_ids[0] if len(run_ids) == 1 else run_ids
    if verb == "tune":
        return pipeline.cmd_tune(cfg)
    if verb == "report":
        return pipeline.cmd_report(cfg)
    if verb == "finetune":
        from extraits.pose.finetune import cmd_finetune

        return cmd_finetune(cfg)

    run_id = cli_args.get("run_id")
    if not run_id:
        raise SystemExit(f"The '{verb}' verb requires run_id=<run_id>.")
    if verb == "predict":
        return pipeline.cmd_predict(cfg, run_id, cli_args.get("split", "test"))
    return pipeline.cmd_evaluate(cfg, run_id)


def _require_train_extra() -> None:
    missing = []
    for module in ("hydra", "omegaconf", "optuna", "pyarrow"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(f"Training needs the `train` extra (missing: {', '.join(missing)}):\n"
                         f"    pip install 'extraits[train]'")


def print_result(result: Any) -> None:
    from omegaconf import OmegaConf

    if isinstance(result, (str, Path)):
        print(result)
    elif isinstance(result, list):
        print("\n".join(str(item) for item in result))
    elif isinstance(result, dict):
        print(OmegaConf.to_yaml(OmegaConf.create(result)))


# --------------------------------------------------------------------------- #
# `extraits pose` (registered in extraits/cli.py)
# --------------------------------------------------------------------------- #
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.epilog = USAGE_EXAMPLES
    p.add_argument("verb", choices=VERBS, help="Step to run.")
    p.add_argument("overrides", nargs=argparse.REMAINDER,
                   help="Hydra overrides (key=value), plus run_id=... / split=... .")
    p.add_argument("--models-dir", default=None,
                   help="Models folder (default: $EXTRAITS_MODELS, else <workspace>/models).")


def run(args: argparse.Namespace) -> int:
    print_result(run_verb(args.verb, list(args.overrides), getattr(args, "workspace", None),
                          args.models_dir))
    return 0
