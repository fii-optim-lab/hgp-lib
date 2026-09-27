from collections.abc import Sequence
from math import ceil

import numpy as np

from ..evaluation import Evaluator
from ..rules import And, Literal, Or, Rule
from ..utils.validation import (
    check_isinstance,
    validate_num_literals,
    validate_operator_types,
)
from .base_strategy import PopulationStrategy


class RandomStrategy(PopulationStrategy):
    """
    Generates rules by randomly selecting an operator and two literals.

    Attributes:
        num_literals (int): The total number of available literals.
        operator_types (Sequence[type[Rule]]): A sequence of allowed operator types
            (e.g., `(Or, And)`). Default: `(Or, And)`.

    Examples:
        >>> from hgp_lib.populations import RandomStrategy
        >>> from hgp_lib.rules import And, Or
        >>> strategy = RandomStrategy(num_literals=5, operator_types=(And, Or))
        >>> rules = strategy.generate(n=1)
        >>> rule = rules[0]
        >>> isinstance(rule, (And, Or))
        True
        >>> len(rule.subrules)
        2
    """

    def __init__(
        self, num_literals: int, operator_types: Sequence[type[Rule]] = (Or, And)
    ):
        validate_num_literals(num_literals)
        validate_operator_types(operator_types)

        self.num_literals = num_literals
        self.operator_types = operator_types

    def generate(self, n: int) -> list[Rule]:
        """
        Generates n rules with a random operator and two random literals.

        Args:
            n (int): Number of rules to generate.

        Returns:
            list[Rule]: A list of randomly generated operator rules, each containing two literal subrules.
        """
        if n <= 0:
            return []

        rules = []

        op_indices = np.random.randint(0, len(self.operator_types), size=n)
        idx1s = np.random.randint(0, self.num_literals, size=n)
        idx2s = np.random.randint(0, self.num_literals - 1, size=n)
        idx2s += idx2s >= idx1s  # Avoid duplicate indices.
        negations = np.random.randint(0, 2, size=(n, 3)).astype(bool)

        for i in range(n):
            operator_class = self.operator_types[op_indices[i]]

            rules.append(
                operator_class(
                    subrules=[
                        Literal(value=idx1s[i], negated=negations[i, 1]),
                        Literal(value=idx2s[i], negated=negations[i, 2]),
                    ],
                    negated=negations[i, 0],
                    copy_subrules=False,
                )
            )
        return rules


class BestLiteralStrategy(PopulationStrategy):
    """
    Generates rules by selecting the single best-performing literal on a random subset of data and features.

    For each generation call, a new subset of the training data (rows) and features (columns) is selected.
    All possible literals in the feature subset (both positive and negated) are evaluated against the data subset,
    and the one with the highest score is returned.

    The rows and the scorer come from the population's evaluator. When the rows were merged into sample weights,
    row subsets are drawn from the original rows the weights stand for, and scored with the weights of the subset.

    Attributes:
        num_literals (int): The total number of available literals.
        evaluator (Evaluator): The population's training rows with its scorer bound to them.
        sample_size (int | float | None): Size of the sample subset (rows) to use for evaluation. It counts
            original rows, also when they were merged into sample weights.
            - If `int`: Number of samples.
            - If `float`: Fraction of samples in (0.0, 1.0].
            - If `None`: Use all samples.
            Default: `None`.
        feature_size (int | float | None): Size of the feature subset (columns) to use for evaluation.
            - If `int`: Number of features.
            - If `float`: Fraction of features in (0.0, 1.0].
            - If `None`: Use all features.
            Default: `None`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.evaluation import Dataset, NumpyBackend, fast_accuracy_score, resolve_scorer
        >>> from hgp_lib.populations import BestLiteralStrategy
        >>> from hgp_lib.rules import Literal
        >>> data = np.array([[True, False], [False, True], [True, True]])
        >>> labels = np.array([1, 0, 1])
        >>> evaluator = NumpyBackend().bind(
        ...     Dataset(data, labels), resolve_scorer(fast_accuracy_score)
        ... )
        >>> strategy = BestLiteralStrategy(num_literals=2, evaluator=evaluator, sample_size=2)
        >>> rules = strategy.generate(n=1)
        >>> isinstance(rules[0], Literal)
        True
        >>> BestLiteralStrategy(num_literals=2, evaluator=evaluator).generate(n=1)
        [0]
    """

    def __init__(
        self,
        num_literals: int,
        evaluator: Evaluator,
        sample_size: int | float | None = None,
        feature_size: int | float | None = None,
    ):
        validate_num_literals(num_literals)
        check_isinstance(evaluator, Evaluator)

        num_features = evaluator.dataset.data.shape[1]
        if num_features != num_literals:
            raise ValueError(
                f"Number of features in the evaluator's data must be equal to num_literals, "
                f"got {num_features} != {num_literals}"
            )

        self.num_literals = num_literals
        self.evaluator = evaluator

        self._total_samples = evaluator.dataset.n_rows
        self._sample_count = self._resolve_size(sample_size, self._total_samples)
        self._feature_count = self._resolve_size(feature_size, num_literals)

    def _resolve_size(self, size: int | float | None, total: int) -> int:
        if size is None:
            return total
        if isinstance(size, float):
            if not (0.0 < size <= 1.0):
                raise ValueError(f"Float size must be between 0.0 and 1.0, got {size}")
            return ceil(total * size)
        if isinstance(size, int):
            if not (0 < size <= total):
                raise ValueError(
                    f"Integer size must be between 1 and {total}, got {size}"
                )
            return size
        raise TypeError(f"size must be int, float or None, got {type(size)}")

    def _score_literals(self, subset, feature_indices) -> list[float] | np.ndarray:
        """Scores of the literals ``i`` and ``~i``, interleaved, for each feature ``i``."""
        if subset is None:
            # All rows: the evaluator's kernels are bound to exactly these rows.
            literals = [
                Literal(value=int(i), negated=negated)
                for i in feature_indices
                for negated in (False, True)
            ]
            return self.evaluator.score(literals)

        # TODO: For the built-in scorers, score all literals at once from ``w @ X`` and
        #  ``(w * y) @ X``, and derive the negated literals from those counts.
        #  Not planned for 2.0.0.
        data, labels, sample_weight = subset
        scorer = self.evaluator.scorer
        scores = []
        for i in feature_indices:
            column = data[:, i]
            scores.append(scorer(labels, column, sample_weight))
            scores.append(scorer(labels, ~column, sample_weight))
        return scores

    def generate(self, n: int) -> list[Rule]:
        """
        Generates n literal rules that perform best on random data/feature subsets.

        Args:
            n (int): Number of rules to generate.

        Returns:
            list[Rule]: A list of Literal instances.
        """
        rules = []
        for _ in range(n):
            subset = None
            if self._sample_count != self._total_samples:
                rows = np.random.choice(
                    self._total_samples, self._sample_count, replace=False
                )
                subset = self.evaluator.dataset.take(rows)

            if self._feature_count == self.num_literals:
                feature_indices = range(self.num_literals)
            else:
                feature_indices = np.random.choice(
                    self.num_literals, self._feature_count, replace=False
                )

            scores = self._score_literals(subset, feature_indices)
            best = int(np.argmax(scores))  # the first best, like a strict ``>`` scan
            rules.append(
                Literal(value=int(feature_indices[best // 2]), negated=bool(best % 2))
            )
        return rules
