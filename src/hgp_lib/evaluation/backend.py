from abc import ABC, abstractmethod
from collections.abc import Sequence

from numpy import ndarray

from ..rules import Rule
from .dataset import Dataset
from .scorers import Scorer


class Evaluator(ABC):
    """
    A dataset with its scorers bound to it. Created by `EvaluationBackend.bind`.

    Binding computes per-dataset constants once (for example the number of positive
    labels), so scoring many rules on the same rows is fast. The fitness scorer and
    the confusion matrix read the same rows and weights.

    Attributes:
        dataset (Dataset): The rows actually scored, merged when the scorer allows it.
            Backends may store the data in their preferred memory layout.
        scorer (Scorer): The unbound scorer. Code that scores subsets of
            ``dataset`` (for example with `Dataset.take`) calls it directly.
    """

    def __init__(self, dataset: Dataset, scorer: Scorer):
        self.dataset = dataset
        self.scorer = scorer

    @abstractmethod
    def score(self, rules: Sequence[Rule]) -> ndarray:
        """
        Score each rule on the bound dataset.

        Args:
            rules (Sequence[Rule]): The rules to score.

        Returns:
            ndarray: One ``float64`` score per rule.
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


class EvaluationBackend(ABC):
    """
    Evaluates rules. Holds options only, so instances are stateless and picklable.

    A backend author implements `_bind` and `predict`. The shared `bind` decides
    whether rows are merged, so the rule "merged rows need a scorer that accepts
    ``sample_weight``" holds for every backend.
    """

    def bind(self, dataset: Dataset, scorer: Scorer) -> Evaluator:
        """
        Bind a scorer to a dataset, for scoring many rules on the same rows.

        Merges duplicate rows first when ``scorer.merge_rows`` is ``True``.

        Args:
            dataset (Dataset): The rows to score.
            scorer (Scorer): The resolved scorer, see `resolve_scorer`.

        Returns:
            Evaluator: The dataset with its scorers bound to it.

        Raises:
            ValueError: If ``dataset`` has weights but ``scorer`` does not allow merging.
        """
        # TODO: Skip merging when it removes too few rows to pay for weighted counting.
        #  The threshold is backend-specific. Not planned for 2.0.0.
        if scorer.merge_rows:
            dataset = dataset.deduplicate()
        elif dataset.sample_weight is not None:
            raise ValueError(
                "The dataset has sample weights, but the scorer does not accept "
                "sample_weight."
            )
        return self._bind(dataset, scorer)

    @abstractmethod
    def _bind(self, dataset: Dataset, scorer: Scorer) -> Evaluator:
        """Build the backend's evaluator for an already merged dataset."""

    @abstractmethod
    def predict(self, rule: Rule, data: ndarray) -> ndarray:
        """
        Evaluate a rule on binarized data, without copying or merging the data.

        Args:
            rule (Rule): The rule to evaluate.
            data (ndarray): 2-D boolean array with the rule's feature columns.

        Returns:
            ndarray: 1-D boolean NumPy array, one prediction per row.
        """
