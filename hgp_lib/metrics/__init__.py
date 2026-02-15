"""Metrics module for HGP library.

This module provides dataclasses for tracking metrics at various levels:
- GenerationMetrics: Metrics for a single generation/epoch
- PopulationHistory: History of a population across generations
- RunResult: Result of one complete run with k-fold CV
- ExperimentResult: Aggregated results across multiple runs
"""

from .core import GenerationMetrics
from .history import PopulationHistory
from .results import ExperimentResult, RunResult

__all__ = [
    "GenerationMetrics",
    "PopulationHistory",
    "RunResult",
    "ExperimentResult",
]
