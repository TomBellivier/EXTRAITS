"""EXTRAITS: extract morphological traits from dorsal images of insects.

From a photograph of an insect in dorsal view, EXTRAITS predicts anatomical
keypoints with an ensemble of pose models, derives morphometric measurements
from them, flags the measurements that make no morphological sense, detects the
scale (scale bar or ruler) and converts the measurements to millimetres.

Sub-packages:
    extraits.process   `extraits process` / `extraits analyze`: inference over images
    extraits.scale     scale-bar (YOLO + OCR) and ruler (Fourier) detectors
    extraits.validity  measurement-validity classifiers
"""

__version__ = "0.1.0"
