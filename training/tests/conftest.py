"""Test configuration.

The training modules use flat imports (e.g. `from reward import compute_reward`)
and are meant to be run with `training/` on the path. Tests live in
`training/tests/`, so we add the parent `training/` dir to sys.path here. This
lets the existing modules import unchanged.
"""
import os
import sys

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
