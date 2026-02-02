import warnings
from functools import partial
from typing import Callable

import numpy as np
from numpy import ndarray

from hgp_lib.utils.validation import validate_callable

import inspect


# TODO: Add documentation
# TODO: Add tests


def accepts_sample_weight(scorer):
    try:
        sig = inspect.signature(scorer)
        for param in sig.parameters.values():
            if param.name == "sample_weight":
                return True

    except (TypeError, ValueError):
        pass

    try:
        labels = np.array([1, 0, 1], dtype=bool)
        count = np.array([2, 1, 1])
        scorer(labels, labels, sample_weight=count)
        return True
    except TypeError:
        return False


def transform_duplicates_to_sample_weight(data, labels):
    Xy = np.hstack((data, labels[:, None]))
    Xy_unique, sample_weight = np.unique(Xy, axis=0, return_counts=True)
    return Xy_unique[:, :-1], Xy_unique[:, -1], sample_weight


def optimize_scorer_for_data(
    scorer: Callable[[ndarray, ndarray], float], data: ndarray, labels: ndarray
):
    validate_callable(scorer)
    if not accepts_sample_weight(scorer):
        # TODO: Check best practices
        warnings.warn(
            'The scorer must accept "sample_weight" to be optimized by removing duplicates in the data'
        )
    else:
        data, labels, sample_weight = transform_duplicates_to_sample_weight(
            data, labels
        )
        scorer = partial(scorer, sample_weight=sample_weight)
    return scorer, data, labels
