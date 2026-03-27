from typing import Callable

from .crossover_executor import CrossoverExecutor
from hgp_lib.rules import Rule
from ..utils.validation import check_isinstance


class CrossoverExecutorFactory:
    """
    Factory for creating configured `CrossoverExecutor` instances.

    This factory encapsulates all crossover configuration parameters except
    the optional `check_valid` callable, which is supplied at creation time.
    This is useful when the same crossover configuration is reused with
    different validation strategies.

    Args:
        crossover_p (float, optional):
            Probability of selecting each rule for crossover. Default: `0.7`.
        crossover_strategy (str, optional):
            Strategy for pairing rules. Must be `"best"` or `"random"`. Default: `"random"`.
        num_tries (int, optional):
            Maximum number of crossover attempts per pair when validation fails.
            Must be `1` when no validator is provided at creation time. Default: `1`.
        operator_p (float, optional):
            Probability of selecting an operator node (vs. a literal) when choosing
            a crossover point in the rule tree. Must be in [0.0, 1.0]. Default: `0.9`.

    Examples:
        >>> import random
        >>> import numpy as np
        >>> from hgp_lib.crossover import CrossoverExecutorFactory
        >>> from hgp_lib.rules import And, Or, Literal
        >>> factory = CrossoverExecutorFactory(crossover_p=1.0)
        >>> def validator(rule):
        ...     return True
        >>> executor = factory.create(validator)
        >>> rules = [
        ...     And([Literal(value=0), Literal(value=1)]),
        ...     Or([Literal(value=2), Literal(value=3)])
        ... ]
        >>> children, parent_indices = executor.apply(rules, [None, None])
        >>> len(children)
        2

        >>> # Without validator
        >>> factory = CrossoverExecutorFactory(crossover_p=1.0, num_tries=1)
        >>> executor = factory.create(None)
        >>> len(executor.apply(rules, [None, None])[0])
        2
    """

    def __init__(
        self,
        crossover_p: float = 0.7,
        crossover_strategy: str = "random",
        num_tries: int = 1,
        operator_p: float = 0.9,
    ):
        check_isinstance(crossover_p, float)
        check_isinstance(crossover_strategy, str)
        check_isinstance(num_tries, int)
        check_isinstance(operator_p, float)

        if crossover_p < 0.0 or crossover_p > 1.0:
            raise ValueError(
                f"crossover_p must be a float between 0.0 and 1.0, is '{crossover_p}'"
            )

        if operator_p < 0.0 or operator_p > 1.0:
            raise ValueError(
                f"operator_p must be a float between 0.0 and 1.0, is '{operator_p}'"
            )

        accepted_strategies = ("best", "random")
        if crossover_strategy not in accepted_strategies:
            raise ValueError(
                f"crossover_strategy must be one of {accepted_strategies}, is '{crossover_strategy}'"
            )

        if num_tries < 1:
            raise ValueError(f"num_tries must be greater than 0, is '{num_tries}'")

        self.crossover_p: float = crossover_p
        self.crossover_strategy: str = crossover_strategy
        self.num_tries: int = num_tries
        self.operator_p: float = operator_p

    def create(
        self, check_valid: Callable[[Rule], bool] | None = None
    ) -> CrossoverExecutor:
        """
        Create a new `CrossoverExecutor` using the factory configuration.

        Args:
            check_valid (Callable[[Rule], bool] | None, optional):
                Optional validator executed after crossover. When supplied, each
                child must pass validation or the crossover is retried up to
                `num_tries` times. Default: `None`.

        Returns:
            CrossoverExecutor:
                A configured crossover executor instance.

        Raises:
            ValueError:
                If `num_tries > 1` and `check_valid` is `None`.

        Examples:
            >>> factory = CrossoverExecutorFactory(crossover_p=1.0)
            >>> executor = factory.create(lambda r: True)
            >>> isinstance(executor, CrossoverExecutor)
            True
        """
        if check_valid is None and self.num_tries > 1:
            raise ValueError("num_tries must be 1 if check_valid is None")

        return CrossoverExecutor(
            crossover_p=self.crossover_p,
            crossover_strategy=self.crossover_strategy,
            check_valid=check_valid,
            num_tries=self.num_tries,
            operator_p=self.operator_p,
        )
