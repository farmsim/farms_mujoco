"""Experiment"""

from dataclasses import dataclass
from farms_core.experiment.data import ExperimentData


@dataclass
class TaskData:
    """Experiment"""

    def __init__(self, **kwargs):
        super().__init__()
        self.iteration: int = 0
        self.data: ExperimentData = kwargs.pop('data', None)
        self.viewer = kwargs.pop('viewer', None)
