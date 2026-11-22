# Third-party notices

EXTRAITS depends on the following projects, installed separately by `pip` (or bundled
in the Docker images). Each keeps its own licence; the versions are those the package
is developed with.

## Core (`pip install extraits`)

| Project | Licence | Use |
|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | pose estimation, scale-bar detection |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause style (with bundled components under Apache-2.0 and others) | deep learning |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 | reading the scale-bar value |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | Apache-2.0 | image processing |
| [NumPy](https://numpy.org) | BSD-3-Clause | arrays |
| [pandas](https://pandas.pydata.org) | BSD-3-Clause | tables |
| [SciPy](https://scipy.org) | BSD-3-Clause | signal processing (ruler) |
| [scikit-learn](https://scikit-learn.org) | BSD-3-Clause | validity classifiers |
| [joblib](https://joblib.readthedocs.io) | BSD-3-Clause | model files |
| [Matplotlib](https://matplotlib.org) | Matplotlib licence (PSF-based) | figures |
| [Pillow](https://python-pillow.org) | MIT-CMU (HPND) | images |
| [PyYAML](https://pyyaml.org) | MIT | settings |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | model download |

## Training (`[train]`)

| Project | Licence |
|---|---|
| [Hydra](https://hydra.cc) | MIT |
| [OmegaConf](https://github.com/omry/omegaconf) | BSD-3-Clause |
| [Optuna](https://optuna.org) | MIT |
| [Apache Arrow (pyarrow)](https://arrow.apache.org) | Apache-2.0 |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause |

## Annotation (`[annotation]`)

| Project | Licence |
|---|---|
| [Label Studio](https://github.com/HumanSignal/label-studio) | Apache-2.0 |
| [Qt for Python (PySide6)](https://www.qt.io/qt-for-python) | LGPL-3.0 (or GPL-2.0 / GPL-3.0) |

PySide6 is used through its LGPL-3.0 licence, as an unmodified, dynamically linked
library installed by `pip`.

## Research (`research/`, `extraits-research`)

| Project | Licence |
|---|---|
| [PEFT](https://github.com/huggingface/peft) | Apache-2.0 |
| [XGBoost](https://github.com/dmlc/xgboost) | Apache-2.0 |

## Models

The official models derive from Ultralytics `yolo26n-pose.pt` (AGPL-3.0): see
[MODEL_LICENSE.md](MODEL_LICENSE.md).
