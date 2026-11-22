"""The `extraits` command.

    extraits process   measure the insects of a folder of images -> one CSV row per image
    extraits analyze   figures and summary of a results CSV
    extraits download-models   fetch the official models from the Hugging Face Hub
    extraits train pose        train the pose ensemble on your annotations   [train extra]
    extraits finetune          adapt the pose models to your own images       [train extra]
    extraits annotate <tool>   annotation tools (Label Studio, checks, table)
    extraits run               the whole chain, as set in <workspace>/run.yaml
    extraits report            tables and figures of every training run      [train extra]
    extraits pose <verb>       any step of the pose framework (research)     [train extra]

Run `extraits <command> --help` for the options of a command. Every command
accepts --workspace (default: $EXTRAITS_WORKSPACE, else the current folder), the
folder holding models/, annotations/, runs/ and results/ (see extraits/workspace.py).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from extraits import __version__

# name -> (help, module). Each module exposes add_arguments(parser) and run(args).
COMMANDS: dict[str, tuple[str, str]] = {
    "process": ("Measure the insects of a folder of images (one CSV row per image).",
                "extraits.process.run"),
    "analyze": ("Figures and summary of a results CSV.",
                "extraits.process.analyze"),
    "download-models": (("Download the official models from the Hugging Face Hub "
                         "(pinned revision, checksums verified)."),
                        "extraits.models"),
    "train": ("Train a model on the annotations of the workspace (extra: train).",
              "extraits.train"),
    "finetune": ("Fine-tune the pose models on your own annotated images (extra: train).",
                 "extraits.finetune"),
    "annotate": ("Annotation tools: label-studio, import-volunteer, to-csv, "
                 "validate-measurements, build-table, check.",
                 "extraits.annotate"),
    "run": ("Run the whole chain (annotations, training, process, analyses) as set in "
            "run.yaml.", "extraits.run"),
    "report": ("Tables and figures of every pose training run of the workspace.",
               "extraits.report"),
    "pose": ("Run one step of the pose framework: prepare, split, train, tune, evaluate, ...",
             "extraits.pose.cli"),
}


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Callable[[argparse.Namespace], int]]]:
    import importlib

    parser = argparse.ArgumentParser(
        prog="extraits",
        description="EXTRAITS: extract morphological traits from dorsal images of insects.")
    parser.add_argument("--version", action="version", version=f"extraits {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    runners: dict[str, Callable[[argparse.Namespace], int]] = {}
    for name, (help_text, module_name) in COMMANDS.items():
        module = importlib.import_module(module_name)
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--workspace", default=None,
                       help="Workspace folder (default: $EXTRAITS_WORKSPACE, else the current "
                            "folder).")
        module.add_arguments(p)
        runners[name] = module.run
    return parser, runners


def main(argv: list[str] | None = None) -> int:
    parser, runners = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return 1
    return int(runners[args.command](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
