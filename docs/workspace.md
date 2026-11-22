# The workspace

EXTRAITS never writes into its installation. Everything you own (images, annotations,
models, training runs, results) lives in a **workspace** folder:

- `--workspace DIR` on any command, else
- the environment variable `EXTRAITS_WORKSPACE`, else
- the current folder.

The models folder can be set apart: `--models-dir DIR`, else `EXTRAITS_MODELS`, else
`<workspace>/models`.

```
<workspace>/
├── run.yaml                          settings of `extraits run` (extraits run --init)
├── models/                           models `extraits process` uses (docs/models.md)
│   ├── pose/                         pose ensemble: <member>/best.pt + model_card.json, ensemble.json
│   ├── scale_bar/best.pt             scale-bar detector
│   └── measurement_validity/         one random forest per measurement + metrics.csv
├── images/<order>/...                annotated images, one folder per insect order (training)
├── annotations/                      annotation files (docs/annotation.md)
│   ├── label_studio_annotations/<order>/*.json
│   ├── pose/pose_annotations.csv
│   ├── scale/scale_annotations.csv
│   ├── meas_classifier/*_measurements.csv
│   ├── measurement_validation/       presets, resume state and logs of the validation app
│   └── annotation_data.csv           the merged table every training reads
├── finetune-images/                  default input of `extraits finetune` (docs/finetune.md)
├── data/pose/                        prepared annotations and cross-validation folds
├── runs/                             one folder per training run (manifest, metrics, weights)
└── results/                          analyses (docs/results.md)
```

Only `models/` is needed to measure images. A command creates the folders it writes
to; none of them must exist beforehand.

Which command reads and writes what:

| Command | Reads | Writes |
|---|---|---|
| `extraits download-models` | | `models/` |
| `extraits process` | `models/`, your images | the CSV you name |
| `extraits analyze` | a results CSV, `annotations/annotation_data.csv` | `<csv>_analysis/` |
| `extraits annotate ...` | `annotations/`, `images/` | `annotations/`, `results/annotation_check/` |
| `extraits train pose` | `annotations/annotation_data.csv`, `images/` | `data/pose/`, `runs/`, `models/pose/` |
| `extraits train validity` | `annotations/annotation_data.csv` | `models/measurement_validity/`, `results/meas_classifier/` |
| `extraits finetune` | your images and annotations, `models/pose/` | `data/pose/finetune/`, `runs/finetune/`, the `--output` folder |
| `extraits report` | `runs/` | `results/pose/` |
| `extraits run` | `run.yaml` | everything above, as configured |

`extraits train pose` and `extraits train validity` **replace** the models in
`models/`: keep a copy of the official ones, or train with `--models-dir` elsewhere.
`extraits finetune` never writes into the base models.
