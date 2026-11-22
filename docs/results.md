# results

Data analyses, in one place: `<workspace>/results/`. Nothing here is read back by a
model or by `extraits process`: these are tables, figures and reports for people.
Every command writes into its sub-folder by default (an `--output-dir` / `--out-dir`
option still overrides it).

```
results/
├── annotation_check/             <- `extraits annotate check`
│   ├── issues.csv                ... every problem found, one row per image
│   └── presence.csv              ... one row per image, one column per annotation source
├── run_logs/                     <- `extraits run`: every command and its output, one log per run
├── pose/                         <- pose training (extraits.pose)
│   ├── master.parquet            ... every evaluated run, one row per metric (`extraits report`)
│   ├── summary_*.parquet         ... report tables (`extraits report`)
│   ├── figures/, runs/<run_id>/  ... report figures, global and per run (`extraits report`)
│   ├── comparison/               ... heatmaps and cost figures (research/scripts/compare_models.py)
│   ├── optuna/<study>/           ... HPO diagnostics (research/scripts/plot_optuna.py)
│   ├── finetune/<variant_hash>/  ... before/after comparison (`extraits pose finetune`)
│   └── reports/                  ... paths.reports
├── meas_classifier/              <- measurement validity
│   ├── training/                 ... metrics, PR curves, confusion, importances (`extraits train validity`)
│   └── comparison/               ... the 8 approaches compared (research/validity/)
├── pipeline/                     <- `extraits analyze`
│   └── <results CSV name>/       ... confidence, scale, ensemble spread, GT error figures + summary.txt
└── ruler_detection/              <- research/ruler/ruler_detection_evaluation.py
    ├── evaluation.json           ... read by ruler_detection_evaluation.ipynb
    └── problems.json
```

## Commands

The workspace is `--workspace` / `--root`, else `$EXTRAITS_WORKSPACE`, else the current folder.

| Analysis | Command |
|----------|---------|
| Everything below, in one run | `extraits run` (settings in `<workspace>/run.yaml`) |
| Annotation check | `extraits annotate check` |
| Pose models: report | `extraits report` |
| Pose models: comparison | `python research/scripts/compare_models.py` |
| Pose models: HPO | `python research/scripts/plot_optuna.py` |
| Measurement validity | `extraits train validity` / `python research/validity/compare_measure_validity_approaches.py` |
| Output of `extraits process` | `extraits analyze --input <results CSV>` |
| Ruler detection | `python research/ruler/ruler_detection_evaluation.py` |

The output of `extraits process` (one row per image, with the mean and standard
deviation of every keypoint over the pose ensemble) goes where `--output` says;
`extraits analyze` writes its figures next to it by default.
