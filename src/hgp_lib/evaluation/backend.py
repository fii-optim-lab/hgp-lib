from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from numpy import ndarray

from ..rules import Rule
from ..utils.validation import validate_callable
from .scorers import ScoreFn, check_merge_rows, fast_f1_score


class Evaluator(ABC):
    """
    A dataset with a scoring function bound to it. Created by `EvaluationBackend.bind`.

    Binding computes per-dataset constants once (for example the number of positive
    labels), so scoring many rules on the same rows is fast. The scores and the confusion
    matrix read the same rows and weights.

    Attributes:
        backend (EvaluationBackend): The backend that created this evaluator.
        dataset: The rows that are scored, in the backend's own dataset type (for
            example `Dataset` for `NumpyBackend`), merged when ``merge_rows`` is
            ``True``. This is the only copy of the rows the library keeps.
        score_fn (ScoreFn): The scoring function. Backends replace `fast_f1_score` and
            `fast_accuracy_score` with their own kernels; other functions receive the
            backend's arrays.
        merge_rows (bool): Whether duplicate rows were merged into sample weights.
    """

    def __init__(
        self,
        backend: "EvaluationBackend",
        dataset: Any,
        score_fn: ScoreFn,
        merge_rows: bool,
    ):
        self.backend = backend
        self.dataset = dataset
        self.score_fn = score_fn
        self.merge_rows = merge_rows

    @abstractmethod
    def score(self, rules: Sequence[Rule]) -> Any:
        """
        Score each rule on the bound dataset.

        Args:
            rules (Sequence[Rule]): The rules to score.

        Returns:
            One score per rule, as a 1-D array of the backend (for example a NumPy array
            for `NumpyBackend`), in the backend's ``dtype``. Use
            ``backend.to_numpy`` to convert it.
        """

    @abstractmethod
    def confusion_matrix(self, rule: Rule) -> tuple[int, int, int, int]:
        """
        Count ``(tp, fp, fn, tn)`` for one rule, over the original rows.

        Args:
            rule (Rule): The rule to evaluate.

        Returns:
            tuple[int, int, int, int]: ``(tp, fp, fn, tn)``.
        """

    def subset(self, rows: ndarray) -> "Evaluator":
        """
        Bind the same scoring function to a subset of the original rows.

        The rows are selected with ``dataset.take``, so with merged rows the subset keeps
        the weights of the selected original rows. Rows are not merged again.

        Args:
            rows (ndarray): Distinct indices of original rows, in
                ``range(dataset.n_rows)``.

        Returns:
            Evaluator: An evaluator bound to the selected rows.
        """
        return self.backend._bind(
            self.dataset.take(rows), self.score_fn, self.merge_rows
        )


class EvaluationBackend(ABC):
    """
    Evaluates rules. Holds options only, so instances are stateless and picklable.

    A backend author implements `_bind`, `predict`, `from_numpy` and `to_numpy`, and a
    dataset type with the methods of `Dataset`. The shared `bind` decides whether rows
    are merged, so the rule "merged rows need a scorer that accepts ``sample_weight``"
    holds for every backend.
    """

    def bind(
        self, dataset: Any, score_fn: ScoreFn = fast_f1_score, *, merge_rows=False
    ) -> Evaluator:
        """
        Bind a scoring function to a dataset, for scoring many rules on the same rows.

        Merges duplicate rows first when ``merge_rows`` is ``True``.

        Args:
            dataset: The rows to score: a NumPy `Dataset`, or the backend's own dataset
                type (for example the rows of a child population).
            score_fn (ScoreFn): The scoring function. Default: `fast_f1_score`.
            merge_rows (bool): Merge duplicate rows into sample weights. Default: `False`.

        Returns:
            Evaluator: The dataset with the scoring function bound to it.

        Raises:
            TypeError: If ``score_fn`` is not callable.
            ValueError: If ``merge_rows`` is ``True`` and ``score_fn`` does not accept
                ``sample_weight``, or if ``dataset`` has weights and ``merge_rows`` is
                ``False``.
        """
        validate_callable(score_fn)
        # TODO: Skip merging when it removes too few rows to pay for weighted counting.
        #  The threshold is backend-specific.
        if merge_rows:
            check_merge_rows(score_fn)
            dataset = dataset.deduplicate()
        elif dataset.sample_weight is not None:
            raise ValueError(
                "The dataset has sample weights, which are only scored with "
                "merge_rows=True."
            )
        return self._bind(dataset, score_fn, merge_rows)

    @abstractmethod
    def _bind(self, dataset: Any, score_fn: ScoreFn, merge_rows: bool) -> Evaluator:
        """Build the backend's evaluator for an already merged dataset."""

    @abstractmethod
    def predict(self, rule: Rule, data: Any) -> Any:
        """
        Evaluate a rule on binarized data, without copying or merging the data.

        Args:
            rule (Rule): The rule to evaluate.
            data: 2-D boolean array of the backend (see `from_numpy`), with the rule's
                feature columns.

        Returns:
            1-D boolean array of the backend, one prediction per row.
        """

    @abstractmethod
    def from_numpy(self, data: ndarray) -> Any:
        """
        Convert a 2-D boolean NumPy array to the backend's array type, for `predict`.

        Args:
            data (ndarray): 2-D boolean array.

        Returns:
            The same data as an array of the backend.
        """

    @abstractmethod
    def to_numpy(self, values: Any) -> ndarray:
        """
        Convert an array of the backend (predictions or scores) to NumPy.

        Args:
            values: An array returned by `predict` or `Evaluator.score`.

        Returns:
            ndarray: The same values as a NumPy array.
        """
