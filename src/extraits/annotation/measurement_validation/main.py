"""`extraits annotate validate-measurements`: the measurement classification application.

    extraits annotate validate-measurements --annotations-csv annotations/pose/pose_annotations.csv

For each annotated image, the annotator marks every measurement measurable or non
measurable. The input CSV is the one produced by `extraits annotate to-csv`. The output
CSV (status of each measurement for each image) is written to
<workspace>/annotations/meas_classifier/, under the name "<input csv>_measurements.csv";
presets, resume state and logs go to <workspace>/annotations/measurement_validation/.
Needs `pip install extraits[annotation]` (PySide6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    from extraits import annotation as paths

    parser = argparse.ArgumentParser(
        prog=prog, description="Measurable / non measurable classification of the measurements.")
    parser.add_argument("--annotations-csv", "--annotations_csv", dest="annotations_csv",
                        default=None,
                        help="CSV produced by `extraits annotate to-csv` (asked at launch if "
                             "missing).")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir",
                        default=str(paths.measurements_dir()),
                        help="Folder of the status CSV "
                             "(default: <workspace>/annotations/meas_classifier).")
    parser.add_argument("--kp-infos", "--kp_infos", dest="kp_infos", default=None,
                        help="kp_infos.yaml the measurements are read from (default: the one "
                             "shipped with EXTRAITS).")
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    from extraits import annotation as paths

    args = build_parser(prog).parse_args(argv)
    try:
        import PySide6  # noqa: F401
    except ImportError:
        raise SystemExit("The measurement-validation app needs PySide6:\n"
                         "    pip install 'extraits[annotation]'") from None

    data_dir = paths.app_dir()
    from .app_logging import install_qt_message_handler, logger, setup_logging

    setup_logging(data_dir / "logs")
    install_qt_message_handler()
    try:
        return _run(args, data_dir)
    except Exception:
        logger.critical("Fatal exception in main()", exc_info=True)
        raise
    finally:
        from . import app_logging

        logger.info("Logs available in %s", app_logging.LOG_DIR)


def _run(args: argparse.Namespace, data_dir: Path) -> int:
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from . import csv_io, main_window
    from .app_logging import logger
    from .config import load_config

    main_window.set_data_dir(data_dir)
    app = QApplication(sys.argv[:1])
    logger.info("QApplication created")

    config = load_config(args.kp_infos)
    logger.info("%d measurement(s) loaded", len(config.measurements))

    csv_path = args.annotations_csv
    if not csv_path:
        from extraits import annotation as paths

        csv_path, _ = QFileDialog.getOpenFileName(
            None, "Select the annotation CSV", str(paths.annotations_dir()), "CSV (*.csv)")
    if not csv_path:
        logger.info("No CSV selected, exiting")
        return 0

    annotations = csv_io.load_annotations(csv_path)
    if not annotations:
        QMessageBox.warning(None, "No annotation", f"No usable annotation in {csv_path}.")
        return 1

    project_name = Path(csv_path).stem
    win = main_window.MainWindow(config, annotations, project_name, args.output_dir)
    # Resume: status per segment and images already validated in a previous session.
    csv_io.load_state(win.state_path, annotations)
    win.show()
    # Loaded after showing the window: the viewport then has its real size, which avoids
    # a broken zoom computation (window not laid out yet).
    win.load_image(win.first_unvalidated_index())
    logger.info("Project '%s': %d annotation(s), export to %s",
                project_name, len(annotations), win.csv_path)

    exit_code = app.exec()
    logger.info("End of the event loop (code %d)", exit_code)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
