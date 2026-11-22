"""Frozen HPO protocol of the research approaches (ADR-0031, ADR-0033)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("approach", ["yolo_pooled", "yolo_per_dataset", "detect_then_pose",
                                      "lora", "group_bn", "yolo_pooled_reduced"])
def test_all_approaches_share_the_same_hpo_budget(config_factory, approach) -> None:
    """§6.3: comparing approaches with different budgets would measure the budget."""
    cfg = config_factory([f"approach={approach}"])
    assert int(cfg.tuning.n_trials) == 20
    assert int(cfg.tuning.n_startup_trials) == 5
    assert int(cfg.tuning.inner_folds) == 3
    assert str(cfg.tuning.mode) == "tune_once"
    assert int(cfg.tuning.pruner_warmup_steps) == 1
    # Four dimensions: beyond that, 20 trials are not enough to explore the space
    assert len(dict(cfg.approach.search_space)) == 4


@pytest.mark.parametrize("approach", ["yolo_pooled", "yolo_per_dataset", "lora",
                                      "group_bn", "yolo_pooled_reduced"])
def test_base_weights_are_frozen_to_the_same_model(config_factory, approach) -> None:
    """ADR-0033: a different model size would make the comparison be about it."""
    cfg = config_factory([f"approach={approach}"])
    assert str(cfg.approach.weights) == "yolo26n-pose.pt"


def test_detect_then_pose_uses_the_same_base_family(config_factory) -> None:
    cfg = config_factory(["approach=detect_then_pose"])
    assert str(cfg.approach.detector.weights).startswith("yolo26n")
    assert str(cfg.approach.pose.weights).startswith("yolo26n")


def test_lora_alpha_is_derived_from_rank(config_factory) -> None:
    """ADR-0031: searching alpha on top of r would explore a redundancy."""
    from omegaconf import OmegaConf

    from extraits_research.approaches.lora import LoraApproach

    cfg = config_factory(["approach=lora"])
    assert "lora.alpha" not in dict(cfg.approach.search_space)

    approach = LoraApproach(cfg)
    OmegaConf.update(cfg, "approach.lora.r", 16)
    assert approach._alpha() == pytest.approx(32.0)      # 2 x r
    OmegaConf.update(cfg, "approach.lora.alpha", 8.0)
    assert approach._alpha() == pytest.approx(8.0)       # explicit value takes precedence
