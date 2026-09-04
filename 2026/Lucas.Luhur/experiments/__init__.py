"""
Experiments framework: compose the src/ components into swappable, sweepable runs.
config.py holds the Config schema and YAML loader, pipeline.py provides run_once
(one cell) and sweep (a grid over config axes), and run.py is the CLI.
"""

from .config import Config, Experiment, load_experiment, make_apply_cell
from .pipeline import run_once, sweep

__all__ = ["Config", "run_once", "sweep", "Experiment", "load_experiment", "make_apply_cell"]
