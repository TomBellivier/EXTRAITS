"""`extraits report`: aggregate every pose run of the workspace into tables and figures.

Reads <workspace>/runs/*/ (the complete runs) and writes <workspace>/results/pose/
(master.parquet, summary tables, comparison figures). Needs `pip install extraits[train]`.
"""

from __future__ import annotations

import argparse


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-figures", action="store_true", help="Tables only (faster).")
    p.add_argument("overrides", nargs="*", metavar="key=value",
                   help="Extra Hydra overrides, e.g. report.dpi=300.")


def run(args: argparse.Namespace) -> int:
    from extraits.pose.cli import print_result, run_verb

    overrides = list(args.overrides)
    if args.no_figures:
        overrides.append("report.figures=false")
    print_result(run_verb("report", overrides, getattr(args, "workspace", None)))
    return 0
