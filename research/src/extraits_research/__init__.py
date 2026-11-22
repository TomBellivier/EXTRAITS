"""Research approaches of EXTRAITS, registered with the pose framework as a plugin.

Installed with `pip install -e research/`, this package adds to `extraits pose` and
`extraits train pose --experiment ...` the approaches compared in the paper
(yolo_per_dataset, detect_then_pose, lora, group_bn, yolo_pooled_reduced, head_only,
lora_per_dataset) and their experiment configs (exp_b ... exp_h). See research/README.md.
"""

from pathlib import Path

# Extra Hydra configs (approach/, experiment/), added to the search path of extraits.pose.
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def register() -> None:
    """Import the approaches: each one registers itself in extraits.pose.registry."""
    from extraits.pose.registry import load_plugins

    load_plugins("extraits_research.approaches")
