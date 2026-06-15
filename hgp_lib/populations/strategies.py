from math import ceil
from typing import Sequence, Type, Callable, List
import numpy as np

from .base_strategy import PopulationStrategy
from .ilp_model import solve_best_rule_ilp
from ..rules import Rule, Literal, And, Or
from ..utils.validation import (
    validate_num_literals,
    validate_operator_types,
    check_X_y,
    validate_callable,
)


class RandomStrategy(PopulationStrategy):
    """
    Generates rules by randomly selecting an operator and two literals.

    Attributes:
        num_literals (int): The total number of available literals.
        operator_types (Sequence[Type[Rule]]): A sequence of allowed operator types
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

    Attributes:
        num_literals (int): The total number of available literals.
        score_fn (Callable): Function to evaluate a rule. Signature: `score_fn(predictions, labels) -> float`.
        train_data (np.ndarray): The training data array.
        train_labels (np.ndarray): The training labels.
        sample_size (int | float | None): Size of the sample subset (rows) to use for evaluation.
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
        validate_callable(score_fn)
        check_X_y(train_data, train_labels)

        if len(train_data[0]) != num_literals:
            raise ValueError(
                f"Number of features in train_data must be equal to num_literals, "
                f"got {len(train_data[0])} != {num_literals}"
            )

        self.num_literals = num_literals
        self.score_fn = score_fn
        self.train_data = train_data
        self.train_labels = train_labels

        self._sample_count = self._resolve_size(sample_size, len(train_data))
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


class ILPStrategy(PopulationStrategy):
    """
    Generates rules by solving an ILP that finds the best AND or OR rule
    on a random subset of data and features.

    For each rule to generate, a fresh data/feature subsample is drawn and
    a Pyomo MIP is solved with HiGHS to select which literals (and their
    negations) to include in an AND or OR combination that maximises
    accuracy on that subsample.

    Attributes:
        num_literals (int): Total number of available literals (features).
            This equals the number of columns in the binarized training data.
        train_data (np.ndarray): Training data (2-D boolean array).
        train_labels (np.ndarray): Training labels (1-D integer array).
        sample_size (int | float | None): Row subsample size. ``int`` for
            absolute count, ``float`` in (0, 1] for fraction, ``None`` for all.
            Default: ``100``.
        feature_size (int | float | None): Column subsample size. Same
            semantics as ``sample_size``. Default: ``20``.
        max_literals (int): Maximum number of literals allowed in a rule.
        min_literals (int): Minimum number of literals in a rule.
        operator_type (str): ``"and"``, ``"or"``, or ``"random"`` (coin-flip
            per rule).
        time_limit (float): Solver wall-clock limit in seconds per rule.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = np.random.rand(100, 10) > 0.5
        >>> labels = np.random.randint(0, 2, 100)
        >>> strategy = ILPStrategy(
        ...     num_literals=10,
        ...     train_data=data,
        ...     train_labels=labels,
        ...     sample_size=50,
        ...     feature_size=5,
        ... )
        >>> rules = strategy.generate(n=2)
        >>> all(isinstance(r, (And, Or)) for r in rules)
        True
    """

    def __init__(
        self,
        num_literals: int,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        sample_size: int | float | None = 100,
        feature_size: int | float | None = 20,
        max_literals: int = 5,
        min_literals: int = 2,
        operator_type: str = "random",
        time_limit: float = 2.0,
    ):
        validate_num_literals(num_literals)
        check_X_y(train_data, train_labels)

        if len(train_data[0]) != num_literals:
            raise ValueError(
                f"Number of features in train_data must equal num_literals, "
                f"got {len(train_data[0])} != {num_literals}"
            )
        if operator_type not in ("and", "or", "random"):
            raise ValueError(
                f"operator_type must be 'and', 'or', or 'random', got '{operator_type}'"
            )
        if min_literals < 2:
            raise ValueError(f"min_literals must be >= 2, got {min_literals}")
        if max_literals < min_literals:
            raise ValueError(
                f"max_literals ({max_literals}) must be >= min_literals ({min_literals})"
            )

        self.num_literals = num_literals
        self.train_data = train_data
        self.train_labels = train_labels
        self.max_literals = max_literals
        self.min_literals = min_literals
        self.operator_type = operator_type
        self.time_limit = time_limit

        self._sample_count = self._resolve_size(sample_size, len(train_data))
        self._feature_count = self._resolve_size(feature_size, num_literals)

    @staticmethod
    def _resolve_size(size: int | float | None, total: int) -> int:
        if size is None:
            return total
        if isinstance(size, float):
            if not (0.0 < size <= 1.0):
                raise ValueError(f"Float size must be in (0.0, 1.0], got {size}")
            return ceil(total * size)
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"Int size must be > 0, got {size}")
            return min(size, total)
        raise TypeError(f"size must be int, float or None, got {type(size)}")

    def generate(self, n: int) -> List[Rule]:
        """
        Generate *n* rules by solving one ILP per rule.

        Args:
            n (int): Number of rules to generate.

        Returns:
            List[Rule]: Generated AND / OR rules.
        """
        if n <= 0:
            return []

        rules: List[Rule] = []
        for _ in range(n):
            rule = self._generate_one()
            rules.append(rule)
        return rules

    def _generate_one(self) -> Rule:
        """Sample data, solve ILP, convert solution to a Rule."""
        # --- subsample rows ---
        total_samples = len(self.train_data)
        if self._sample_count == total_samples:
            row_idx = np.arange(total_samples)
        else:
            row_idx = np.random.choice(total_samples, self._sample_count, replace=False)

        # --- subsample features ---
        if self._feature_count == self.num_literals:
            feat_idx = np.arange(self.num_literals)
        else:
            feat_idx = np.random.choice(
                self.num_literals, self._feature_count, replace=False
            )

        sub_data = self.train_data[np.ix_(row_idx, feat_idx)]
        sub_labels = self.train_labels[row_idx]

        # --- choose operator ---
        if self.operator_type == "random":
            use_and = bool(np.random.randint(0, 2))
        else:
            use_and = self.operator_type == "and"

        # --- solve ILP ---
        solution = solve_best_rule_ilp(
            data=sub_data,
            labels=sub_labels,
            use_and=use_and,
            min_literals=self.min_literals,
            max_literals=self.max_literals,
            time_limit=self.time_limit,
        )

        # --- convert to Rule ---
        if not solution.feasible:
            selected, negations = self._random_fallback(len(feat_idx))
            return self._build_rule(feat_idx, selected, negations, use_and)

        return self._build_rule(
            feat_idx, solution.selected_features, solution.negations, use_and
        )

    @staticmethod
    def _random_fallback(
        n_features: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return two random literals as a fallback when the solver fails."""
        idxs = np.random.choice(n_features, size=2, replace=False)
        negs = np.random.randint(0, 2, size=2).astype(bool)
        return idxs, negs

    @staticmethod
    def _build_rule(
        feat_idx: np.ndarray,
        selected: np.ndarray,
        negations: np.ndarray,
        use_and: bool,
    ) -> Rule:
        """Convert ILP solution indices back to a Rule in the full feature space.

        Args:
            feat_idx: Mapping from subsample feature index to global feature index.
            selected: Indices into the subsampled feature space.
            negations: Whether each selected literal is negated.
            use_and: True for AND rule, False for OR rule.

        Returns:
            An And or Or Rule instance.
        """
        subrules = [
            Literal(value=int(feat_idx[j]), negated=bool(neg))
            for j, neg in zip(selected, negations)
        ]
        op_class = And if use_and else Or
        return op_class(subrules=subrules, negated=False, copy_subrules=False)
