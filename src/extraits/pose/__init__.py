"""Pose-model training framework of EXTRAITS (`pip install extraits[train]`).

Prepares the annotation table, builds the cross-validation folds, trains, tunes and
evaluates the pose approaches, and exports the retained models to <models>/pose/.
Entry points: `extraits train pose`, `extraits report` and `extraits pose <verb>`.
The doctrine of the framework (contracts, protocol) is in docs/pose/CONVENTIONS.md.
"""
