"""
config.py
=========

Single place for every tunable parameter of `extraits process`. Nothing here runs
any heavy code; it only declares values that the other modules read. The command
line (`extraits process --help`) overrides the most common ones at run time.

Model paths are resolved from the models folder (see extraits/workspace.py):
--models-dir, else $EXTRAITS_MODELS, else <workspace>/models. `use_models_dir()`
re-points every model path at another folder.
"""

from pathlib import Path

from extraits import workspace

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODELS_DIR: Path = workspace.models_dir()

# Folder that holds the ENSEMBLE of pose models, one sub-folder per model:
#     <models>/pose/<run_id>/best.pt   (+ model_card.json)
# EVERY *.pt under it is loaded and run on each image; the keypoints are averaged
# over the models (mean + std in the CSV).
POSE_MODELS_DIR: Path = MODELS_DIR / workspace.POSE_SUBDIR

# YOLO scale-bar detector.
SCALE_BAR_MODEL_PATH: Path = MODELS_DIR / workspace.SCALE_BAR_SUBPATH

# Measurement-validity classifiers and their decision thresholds.
MEASUREMENT_CLASSIFIER_DIR: Path = MODELS_DIR / workspace.VALIDITY_SUBDIR
MEASUREMENT_CLASSIFIER_METRICS_CSV: Path = MEASUREMENT_CLASSIFIER_DIR / workspace.VALIDITY_METRICS_NAME


def use_models_dir(models_dir) -> None:
    """Point every model path at `models_dir` (layout: pose/, scale_bar/, measurement_validity/)."""
    global MODELS_DIR, POSE_MODELS_DIR, SCALE_BAR_MODEL_PATH
    global MEASUREMENT_CLASSIFIER_DIR, MEASUREMENT_CLASSIFIER_METRICS_CSV
    MODELS_DIR = Path(models_dir).expanduser().resolve()
    POSE_MODELS_DIR = MODELS_DIR / workspace.POSE_SUBDIR
    SCALE_BAR_MODEL_PATH = MODELS_DIR / workspace.SCALE_BAR_SUBPATH
    MEASUREMENT_CLASSIFIER_DIR = MODELS_DIR / workspace.VALIDITY_SUBDIR
    MEASUREMENT_CLASSIFIER_METRICS_CSV = MEASUREMENT_CLASSIFIER_DIR / workspace.VALIDITY_METRICS_NAME


# Two models only average the same insect: a model whose best-matching instance
# overlaps the reference instance (box IoU) less than this is left out of the mean.
ENSEMBLE_MIN_IOU = 0.5

# Default output CSV when --output is not given (relative to the current directory).
OUTPUT_CSV = Path("extraits_results.csv")

# Crash safety: force the CSV to disk every N rows (file.flush + os.fsync), so a
# crash loses at most the last N rows. 1 = safest (durable write per image);
# larger = a little faster; 0 = only flush at the very end.
CSV_FLUSH_EVERY_N_ROWS = 20

# Export the keypoints of the measured instance as extra CSV columns:
#   '<kp> [kp_x]', '<kp> [kp_y]', '<kp> [kp_conf]'   mean over the pose ensemble
#   '<kp> [kp_x_std]', '<kp> [kp_y_std]'              std over the ensemble (0 for 1 model)
# This adds 5 * NUM_KEYPOINTS columns and is what unlocks the keypoint-level
# error analysis (OKS, per-keypoint error vs confidence) in `extraits analyze`.
EXPORT_KEYPOINTS = True

# Accepted image extensions (lower-case, with the dot).
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# --------------------------------------------------------------------------- #
# Hardware (see hardware.py): by default everything adapts to the machine
# --------------------------------------------------------------------------- #
# Where the models run: "auto" = every CUDA GPU (workers take them in turn), else
# the Apple GPU ("mps"), else the CPU. Or force "cpu", "cuda:1", "mps"...
DEVICE = "auto"

# Parallel worker threads: "auto" (sized from the GPUs, CPUs and free memory) or a
# number. --workers on the command line overrides it.
WORKERS = "auto"
WORKERS_PER_GPU = 4            # threads sharing one GPU (they overlap its inference with CPU work)
CPU_THREADS_PER_WORKER = 2     # CPU only: compute threads given to each worker

# Every worker holds its own copy of the models: "auto" never plans more workers
# than MEMORY_BUDGET_FRACTION of the free RAM (and VRAM) can hold, counting
# WORKER_MEMORY_OVERHEAD_MB of buffers per worker on top of its models.
MEMORY_BUDGET_FRACTION = 0.7
WORKER_MEMORY_OVERHEAD_MB = 400

# FP16 inference of the pose models on a CUDA GPU (faster, half the memory). No
# effect on the CPU or the Apple GPU, which stay in FP32.
HALF_PRECISION_ON_GPU = True

# --------------------------------------------------------------------------- #
# Insect order (see insect_group.py)
# --------------------------------------------------------------------------- #
# How an image is matched against a --groups-csv table:
#   "name" -> exact file name incl. extension (e.g. "bee_001.jpg")
#   "stem" -> file name without extension     (e.g. "bee_001")
MATCH_ON = "name"

# --------------------------------------------------------------------------- #
# Pose inference
# --------------------------------------------------------------------------- #
POSE_CONF_THRESHOLD = 0.25      # YOLO object-detection confidence threshold

# When several insects are detected on one image, which one do we measure?
#   "highest_conf" -> the instance with the highest detection score
#   "largest_box"  -> the instance with the largest bounding box
INSTANCE_SELECTION = "highest_conf"

# A keypoint whose confidence is below this threshold is considered "not
# reliably seen". This is used (a) to flag measurements and (b) optionally to
# drop them (see below). Give per-keypoint overrides for points that are known
# to be systematically hard (antenna tips, wing apex, tarsi, ...).
KEYPOINT_VISIBILITY_THRESHOLD = 0.50
PER_KEYPOINT_VISIBILITY_THRESHOLD = {
    # Example (uncomment / edit by hand):
    # "left-antenna-2":  0.30,
    # "right-antenna-2": 0.30,
    # "left-forewing-tip": 0.35,
}

# If True, a measurement is set to NaN (pixels AND mm) as soon as one of its
# keypoints is below its visibility threshold. If False, the measurement is
# still computed and it is only the *confidence* value that reflects the risk.
DROP_MEASUREMENT_IF_KP_BELOW_THRESHOLD = False

# --------------------------------------------------------------------------- #
# Confidence -- POSE (measurements)
# --------------------------------------------------------------------------- #
# Which signal fills the per-measurement confidence columns.
#   "keypoint" -> aggregated keypoint confidence (cheap, always available)
#   "tta"      -> test-time-augmentation dispersion (slower, needs ENABLE_TTA)
# Both signals are implemented in confidence.py; switch here once you have
# decided (on your val split) which one predicts the real error best.
MEASUREMENT_CONFIDENCE_SIGNAL = "keypoint"

# Aggregation used by the keypoint-based signal over the keypoints of one
# measurement: "min" | "geometric_mean" | "mean".
# "min" is the safest default: a distance is ruined as soon as ONE endpoint is
# wrong, so the weakest keypoint should drive the confidence.
KEYPOINT_AGGREGATION = "min"

# --------------------------------------------------------------------------- #
# Confidence -- POSE (test-time augmentation, TTA)
# --------------------------------------------------------------------------- #
# TTA re-runs inference on slightly perturbed copies of the image and measures
# how stable each measurement is. Low dispersion -> high confidence.
ENABLE_TTA = False                       # master switch (True is slower)

TTA_INCLUDE_IDENTITY = True              # include the un-augmented pass
TTA_USE_HFLIP = True                     # horizontal flip (swaps L/R keypoints)
TTA_ROTATION_DEGREES = [-4.0, 4.0]       # one extra pass per angle
TTA_BRIGHTNESS_FACTORS = [0.85, 1.15]    # one extra pass per factor (photometric)

# Confidence transform from the coefficient of variation (cv = std / mean):
#   confidence = exp(-TTA_CV_BETA * cv), clamped to [0, 1].
# Larger beta -> stricter (a small dispersion already lowers the confidence).
TTA_CV_BETA = 8.0

# --------------------------------------------------------------------------- #
# Confidence -- overall pose
# --------------------------------------------------------------------------- #
# Overall pose confidence = detection_conf * mean(keypoint_conf).
# The detection score says "this really is an insect"; the mean keypoint score
# says "and it is well articulated". Their product is an honest global value.
# Set to "min" to instead take min(detection_conf, mean_keypoint_conf).
OVERALL_POSE_METHOD = "det_x_kp"         # "det_x_kp" | "min"

# --------------------------------------------------------------------------- #
# Scale bar
# --------------------------------------------------------------------------- #
SCALE_BAR_CONF_THRESHOLD = 0.1
SCALE_BAR_PADDING = 20

# The scale-bar model is expected to detect TWO boxes: the bar and the text.
# Set the class ids below to match YOUR model. When the model is loaded, its
# class map (model.names) is printed so you can verify these values.
# If your model has a single class, set SCALE_BAR_TEXT_CLASS_ID = None; the text
# confidence then falls back to SCALE_BAR_MISSING_BOX_CONF and OCR is read from
# the (padded) bar crop.
SCALE_BAR_BAR_CLASS_ID = 0               # <-- VERIFY against model.names
SCALE_BAR_TEXT_CLASS_ID = 1              # <-- VERIFY against model.names (or None)
SCALE_BAR_MISSING_BOX_CONF = 1.0         # neutral value when a box is absent

# --------------------------------------------------------------------------- #
# Ruler (Fourier analysis)
# --------------------------------------------------------------------------- #
RULER_RATIO = 5                          # image sub-sampling factor
RULER_GRADUATION_MM = 1.0                # physical spacing of the ruler ticks
RULER_CONF_THRESHOLD = 0.03

HORIZONTAL_RULER_ONLY = True

# --------------------------------------------------------------------------- #
# Scale strategy
# --------------------------------------------------------------------------- #
USE_SCALE_BAR = True                     # try the scale bar first
USE_RULER_FALLBACK = True                # if the bar fails, try the ruler

# The ruler detector runs twice per call (horizontal + vertical), so it is skipped
# once the scale bar has succeeded -- unless RUN_ALL_SCALE_METHODS is True
# (--all-scale-methods), which keeps the raw confidence of BOTH methods on every
# image: what an evaluation against manual scale annotations needs.
RUN_ALL_SCALE_METHODS = False

# --------------------------------------------------------------------------- #
# Converted-measurement confidence (millimetres)
# --------------------------------------------------------------------------- #
# A millimetre value depends on TWO reliable things: the pixel measurement and
# the scale. Combine their confidences with:
#   "min"     -> min(measurement_conf, scale_conf)      [conservative, simple]
#   "product" -> measurement_conf * scale_conf
CONVERTED_CONF_METHOD = "min"

# A detected scale that implies an unrealistic photographed extent is a
# scale-detection glitch rather than a real value: the mm conversion is
# skipped (mm -> NaN) for that image. scale_px_per_mm / scale_confidence
# themselves are left untouched (still diagnostic).
#
# Both bounds are expressed as a FRACTION of the image's largest dimension
# (image_size = max(width, height) in px), not a fixed absolute px/mm, so
# they scale with whatever resolution the photos actually are:
#   scale_px_per_mm > MAX_SCALE_PX_PER_MM_FRACTION * image_size
#       -> too zoomed IN (e.g. at 50%, one measured mm would already be half
#          the image -- implausible for a whole-insect photo)
#   scale_px_per_mm < MIN_SCALE_PX_PER_MM_FRACTION * image_size
#       -> too zoomed OUT (e.g. at 0.1%, one measured mm barely covers a
#          thousandth of the image -- implausibly far away / low-res)
MIN_SCALE_PX_PER_MM_FRACTION = 0.005     # 0.5% of image size
MAX_SCALE_PX_PER_MM_FRACTION = 0.5       # 50% of image size

# --------------------------------------------------------------------------- #
# Optional CSV columns
# --------------------------------------------------------------------------- #
# These are USEFUL additions that were proposed but are intentionally OFF by
# default (only the columns you asked for are written). Flip any of them to True
# to add the column; the plumbing already exists in csv_writer.py.
OPTIONAL_COLUMNS = {
    "scale_method":         True,   # "scale_bar" | "ruler" | "none"
    "detection_confidence": True,   # raw YOLO box score of the measured insect
    "image_width":          True,
    "image_height":         True,
    "needs_review":         True,   # derived boolean from confidence thresholds
    "scale_bar_confidence" :  True,
    "ruler_confidence" : True,
    "scale_bar_box":        True,   # scale-bar detection box, 'x1,y1,x2,y2'
    "scale_text_box":       True,   # scale-bar text box, 'x1,y1,x2,y2' (blank if no text class)
    "scale_ocr_text":       True,   # scale-bar raw OCR read
    "ruler_line":           True,   # ruler: row/col index the reading was taken at
    "ruler_orientation":    True,   # ruler: "horizontal" | "vertical"
}

# Threshold used only if OPTIONAL_COLUMNS["needs_review"] is True:
# a row is flagged when the overall pose confidence OR the scale confidence
# falls below this value.
NEEDS_REVIEW_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Measurement-validity classifier (pre-trained, inference only)
# --------------------------------------------------------------------------- #
# Per-measurement random forests (trained offline by
# `extraits train validity`, which writes them
# into the models folder) that predict whether a measurement is
# trustworthy from the keypoint COORDINATES of its anatomical neighbourhood
# (expressed inside the bounding box of the instance's keypoints, so the model
# does not depend on image resolution or framing) plus the insect's taxonomic
# group. Nothing is trained here -- the saved models are only loaded and queried.
RUN_MEASUREMENT_CLASSIFIER = True

# --------------------------------------------------------------------------- #
# Per-stage processing time
# --------------------------------------------------------------------------- #
# Export the wall-clock time of every pipeline stage for one image, in seconds:
#   '<stage> time [s]'
EXPORT_TIMINGS = True
