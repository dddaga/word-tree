"""
Experiment: phase propagates only via radiation, magnitude only via conduction (direct).
Use create_experiment_layer() and run_experiment.py; does not modify original code.
"""

from .gnn import PhaseRadMagConductionGNN
from .layer import create_experiment_layer, InProcessNeurographLayer
from .optim import ExperimentGNNAdam

__all__ = [
    "PhaseRadMagConductionGNN",
    "create_experiment_layer",
    "InProcessNeurographLayer",
    "ExperimentGNNAdam",
]
