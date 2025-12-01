import random
from typing import Sequence, Type, Callable, List
import numpy as np

from .base_strategy import PopulationStrategy
from ..rules import Rule, Literal, And, Or
from ..utils.validation import validate_num_literals, validate_operator_types, check_X_y


class RandomStrategy(PopulationStrategy):
    """
    Generates rules by randomly selecting an operator and two literals.

    Attributes:
        num_literals (int): The total number of available literals.
        operator_types (Sequence[Type[Rule]]): A sequence of allowed operator types (e.g., `(Or, And)`). Default: `(Or, And)`.

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
        self, num_literals: int, operator_types: Sequence[Type[Rule]] = (Or, And)
    ):
        validate_num_literals(num_literals)
        validate_operator_types(operator_types)

        self.num_literals = num_literals
        self.operator_types = operator_types

    def generate(self, n: int) -> List[Rule]:
        """
        Generates n rules with a random operator and two random literals.

        Args:
            n (int): Number of rules to generate.

        Returns:
            List[Rule]: A list of randomly generated operator rules, each containing two literal subrules.
        """
        rules = []
        for _ in range(n):
            operator_class = random.choice(self.operator_types)

            idx1, idx2 = random.sample(range(self.num_literals), 2)

            rules.append(
                operator_class(
                    subrules=[
                        Literal(value=idx1, negated=random.random() < 0.5),
                        Literal(value=idx2, negated=random.random() < 0.5),
                    ],
                    negated=random.random() < 0.5,
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

    Attributes:
        num_literals (int): The total number of available literals.
        score_fn (Callable): Function to evaluate a rule. Signature: `score_fn(predictions, labels) -> float`.
        train_data (np.ndarray): The training data array.
        train_labels (np.ndarray): The training labels.
        sample_size (int | float | None): Size of the sample subset (rows) to use for evaluation.
            - If `int`: Number of samples.
            - If `float`: Fraction of samples.
            - If `None`: Use all samples.
            Default: `None`.
        feature_size (int | float | None): Size of the feature subset (columns) to use for evaluation.
            - If `int`: Number of features.
            - If `float`: Fraction of features.
            - If `None`: Use all features.
            Default: `None`.

    Examples:
        >>> import numpy as np
        >>> from hgp_lib.populations import BestLiteralStrategy
        >>> from hgp_lib.rules import Literal
        >>> data = np.array([[True, False], [False, True], [True, True]])
        >>> labels = np.array([1, 0, 1])
        >>> def simple_score(preds, y):
        ...     return np.mean(preds == y)
        >>> strategy = BestLiteralStrategy(
        ...     num_literals=2,
        ...     score_fn=simple_score,
        ...     train_data=data,
        ...     train_labels=labels,
        ...     sample_size=2,
        ...     feature_size=None
        ... )
        >>> rules = strategy.generate(n=1)
        >>> rule = rules[0]
        >>> isinstance(rule, Literal)
        True
    """

    def __init__(
        self,
        num_literals: int,
        score_fn: Callable[[np.ndarray, np.ndarray], float],
        train_data: np.ndarray,
        train_labels: np.ndarray,
        sample_size: int | float | None = None,
        feature_size: int | float | None = None,
    ):
        validate_num_literals(num_literals)
        if not callable(score_fn):
            raise TypeError(f"score_fn must be callable, is {type(score_fn)}")

        check_X_y(train_data, train_labels)

        self.num_literals = num_literals
        self.score_fn = score_fn
        self.train_data = train_data
        self.train_labels = train_labels

        self._sample_count = self._resolve_size(sample_size, len(train_data))
        self._feature_count = self._resolve_size(feature_size, num_literals)

        if len(train_data[0]) != num_literals:
            raise ValueError(
                f"Number of features in train_data must be equal to num_literals, got {len(train_data[0])} != {num_literals}"
            )

    def _resolve_size(self, size: int | float | None, total: int) -> int:
        if size is None:
            return total
        if isinstance(size, float):
            if not (0.0 < size <= 1.0):
                raise ValueError(f"Float size must be between 0.0 and 1.0, got {size}")
            return int(total * size)
        if isinstance(size, int):
            if not (0 < size <= total):
                raise ValueError(
                    f"Integer size must be between 1 and {total}, got {size}"
                )
            return size
        raise TypeError(f"size must be int, float or None, got {type(size)}")

    def generate(self, n: int) -> List[Rule]:
        """
        Generates n literal rules that perform best on random data/feature subsets.

        Args:
            n (int): Number of rules to generate.

        Returns:
            List[Rule]: A list of Literal instances.
        """
        rules = []
        total_samples = len(self.train_data)

        for _ in range(n):
            if self._sample_count == total_samples:
                row_indices = slice(None)
            else:
                row_indices = np.random.choice(
                    total_samples, self._sample_count, replace=False
                )

            if self._feature_count == self.num_literals:
                feature_indices = range(self.num_literals)
            else:
                feature_indices = np.random.choice(
                    self.num_literals, self._feature_count, replace=False
                )

            subset_data = self.train_data[row_indices]
            subset_labels = self.train_labels[row_indices]

            best_rule = None
            best_score = -float("inf")

            for i in feature_indices:
                # Optimization: Direct access to column data avoids object creation overhead
                preds_pos = subset_data[:, i]
                score_pos = self.score_fn(preds_pos, subset_labels)

                if score_pos > best_score:
                    best_score = score_pos
                    best_rule = Literal(value=i, negated=False)

                preds_neg = ~preds_pos
                score_neg = self.score_fn(preds_neg, subset_labels)

                if score_neg > best_score:
                    best_score = score_neg
                    best_rule = Literal(value=i, negated=True)

            rules.append(best_rule)

        return rules
