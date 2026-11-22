"""`extraits run`: settings, --init, and the commands of every step (dry run)."""

from __future__ import annotations

import yaml

from extraits.cli import main
from extraits.run import STEPS, defaults


def test_template_holds_every_step_setting():
    cfg = defaults()
    assert {"annotations", "pose_training", "validity", "process", "analysis", "research",
            "run"} <= set(cfg)
    assert cfg["pose_training"]["mode"] == "train" and cfg["process"]["scale"] == "auto"


def test_init_then_dry_run_lists_every_step(tmp_path, capsys):
    assert main(["run", "--workspace", str(tmp_path), "--init"]) == 0
    assert (tmp_path / "run.yaml").is_file()
    assert main(["run", "--workspace", str(tmp_path), "--init"]) == 1   # never overwritten
    capsys.readouterr()

    assert main(["run", "--workspace", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    for step in STEPS:
        assert f"===== {step} " in out
    for command in ("annotate --workspace", "to-csv", "build-table", "check --min-per-class 20",
                    "pose --workspace", "prepare experiment=exp_a_yolo_pooled",
                    "train --workspace", "validity", "process --workspace", "report --workspace",
                    "analyze --input"):
        assert command in out, command
    assert not (tmp_path / "results").exists()          # a dry run writes nothing


def test_settings_are_validated_and_typos_reported(tmp_path, capsys):
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({"pose_training": {"mode": "optimise", "folds": "all",
                                                        "optimisation": {"n_trials": 5}},
                                      "process": {"scale": "ruler", "pose_modles": "x"}}))
    assert main(["run", "--workspace", str(tmp_path), "--config", str(config), "--dry-run",
                 "--only", "pose", "process"]) == 0
    out = capsys.readouterr().out
    assert "tune experiment=exp_a_yolo_pooled folds=all" in out
    assert "tuning.n_trials=5" in out and "--scale ruler" in out
    assert "Unknown setting 'process.pose_modles'" in out
    assert "===== annotations" not in out

    config.write_text(yaml.safe_dump({"process": {"scale": "laser"}}))
    assert main(["run", "--workspace", str(tmp_path), "--config", str(config)]) == 2
    assert "process.scale: 'laser'" in capsys.readouterr().out
