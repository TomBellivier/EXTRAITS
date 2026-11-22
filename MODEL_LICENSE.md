# Licence of the models and of the Docker images

| What | Licence |
|---|---|
| EXTRAITS source code (this repository) | [MIT](LICENSE) |
| Official EXTRAITS models (pose ensemble, scale-bar detector, validity classifiers) | [AGPL-3.0](LICENSES/AGPL-3.0.txt) |
| EXTRAITS Docker images | [AGPL-3.0](LICENSES/AGPL-3.0.txt) |
| Models you train or fine-tune with EXTRAITS from the official ones | AGPL-3.0 (derived works) |

## Why the models are AGPL-3.0

The pose models and the scale-bar detector are Ultralytics YOLO models, fine-tuned
from `yolo26n-pose.pt` and trained with the Ultralytics library, both licensed by
Ultralytics under the GNU Affero General Public License v3.0. The weights are works
derived from them and are distributed under the same licence. The measurement-validity
classifiers (scikit-learn random forests) are distributed with them, under the same
licence, as one set of models.

The Docker images bundle the Ultralytics library, hence the same licence.

## What it means in practice

- You may use, study, modify and redistribute the models and the images, including
  commercially.
- If you distribute them, modified or not, or make a modified version available to
  users over a network, you must provide the corresponding source under the AGPL-3.0.
- The MIT licence of the EXTRAITS code is unaffected: you can reuse the code under the
  MIT terms. A program that loads the models and Ultralytics is subject to the
  AGPL-3.0 as a whole when it is distributed.

Ultralytics also sells an Enterprise licence for uses that cannot comply with the
AGPL-3.0: <https://www.ultralytics.com/license>.

## Data

The models were trained on images from natural-history image databases; the images
and their annotations are not distributed. Check the terms of use of the databases
listed in the model card before redistributing derived data.
<!-- TODO: databases and their licences (see CITATION.cff) -->

The full text of the AGPL-3.0 is in [LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt).
