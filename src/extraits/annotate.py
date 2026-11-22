"""`extraits annotate <tool>`: the annotation tools of EXTRAITS.

    extraits annotate label-studio            start Label Studio (local images served)
    extraits annotate import-volunteer ...    re-point a Label Studio export made elsewhere
                                              at the local image folders
    extraits annotate to-csv                  Label Studio exports -> pose CSV
    extraits annotate validate-measurements   desktop app: mark each measurement of each
                                              image measurable / non measurable
    extraits annotate build-table             merge pose + scale + measurement statuses into
                                              annotations/annotation_data.csv
    extraits annotate check                   check every annotation file against the others

Run `extraits annotate <tool> --help` for its options. The default paths are in the
workspace (see extraits/annotation/__init__.py). label-studio and validate-measurements
need `pip install extraits[annotation]`.
"""

from __future__ import annotations

import argparse
import os
import sys

# tool -> (help, module exposing main(argv, prog))
TOOLS: dict[str, tuple[str, str]] = {
    "label-studio": ("Start Label Studio with local image serving.", ""),
    "import-volunteer": ("Re-point a Label Studio export at the local image folders.",
                         "extraits.annotation.import_volunteer"),
    "to-csv": ("Convert Label Studio exports into the pose CSV.",
               "extraits.annotation.labelstudio"),
    "validate-measurements": ("Desktop app to mark each measurement measurable or not.",
                              "extraits.annotation.measurement_validation.main"),
    "build-table": ("Merge pose, scale and measurement statuses into annotation_data.csv.",
                    "extraits.annotation.build_table"),
    "check": ("Check every annotation file against the others.",
              "extraits.annotation.check"),
}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.epilog = __doc__.split("\n\n", 1)[1]
    p.add_argument("tool", choices=list(TOOLS), metavar="<tool>",
                   help="One of: " + ", ".join(TOOLS) + ".")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Options of the tool (--help).")


def label_studio(argv: list[str], prog: str) -> int:
    """Start Label Studio, with the local files of --document-root served to the tasks."""
    import shutil
    import subprocess

    from extraits.annotation import ANNOTATION_PROTOCOL, LABEL_STUDIO_TEMPLATE

    parser = argparse.ArgumentParser(
        prog=prog, description="Start Label Studio with local image serving. Any other "
                               "option is passed to `label-studio start`.")
    parser.add_argument("--document-root", default=None,
                        help="Root of the local images Label Studio may serve "
                             "(LOCAL_FILES_DOCUMENT_ROOT; default: Label Studio's own).")
    args, rest = parser.parse_known_args(argv)
    executable = shutil.which("label-studio")
    if executable is None:
        raise SystemExit("Label Studio is not installed: pip install 'extraits[annotation]'")
    env = dict(os.environ, LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true")
    if args.document_root:
        env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = os.path.abspath(args.document_root)
    print(f"Labelling interface (Settings > Labeling Interface > Code): {LABEL_STUDIO_TEMPLATE}")
    print(f"Annotation protocol: {ANNOTATION_PROTOCOL}")
    return subprocess.call([executable, "start", *rest], env=env)


def run(args: argparse.Namespace) -> int:
    import importlib

    # The tools resolve their default paths from the workspace when they build their
    # parser: --workspace is passed to them through the environment.
    if getattr(args, "workspace", None):
        os.environ["EXTRAITS_WORKSPACE"] = os.path.abspath(args.workspace)
    prog = f"extraits annotate {args.tool}"
    if args.tool == "label-studio":
        return label_studio(list(args.args), prog)
    module = importlib.import_module(TOOLS[args.tool][1])
    sys.stdout.flush()
    return int(module.main(list(args.args), prog=prog) or 0)
