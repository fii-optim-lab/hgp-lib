"""What training and benchmarking return.

- `GenerationMetrics`: metrics of one generation (epoch) of a population.
- `PopulationHistory`: the generations of one population, returned by ``GPTrainer.fit``.
- `RunResult`: one benchmark run with k-fold cross-validation and a test evaluation.
- `ExperimentResult`: the runs of a benchmark, returned by ``GPBenchmarker.fit``.
"""

from .experiment import ExperimentResult, RunResult
from .generation import GenerationMetrics
from .history import PopulationHistory

__all__ = [
    "ExperimentResult",
    "GenerationMetrics",
    "PopulationHistory",
    "RunResult",
]
