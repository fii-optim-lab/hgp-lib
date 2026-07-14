from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd


class Binarizer(ABC):
    """
    Abstract base class for binarizers.

    A binarizer converts a mixed-type ``pandas.DataFrame`` into a purely boolean
    ``DataFrame``, where every column is a boolean feature that a rule can test.
    Concrete implementations must preserve this contract so they are interchangeable,
    for example inside ``GPBenchmarker``, which fits a fresh copy per fold.

    Contract:
        - ``fit_transform(X, y=None)`` learns the encoding from ``X`` (and optional
          labels ``y``) and returns a boolean ``DataFrame``.
        - ``transform(X)`` applies the learned encoding to new data and returns a
          boolean ``DataFrame`` with the same columns, in the same order, as the
          output of ``fit_transform``.
        - ``get_feature_names_out()`` returns the output column names, in order, so a
          literal's feature index maps to a readable name via ``feature_names[index]``.
        - ``is_fitted`` reports whether the binarizer has been fitted.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from hgp_lib.preprocessing.base import Binarizer
        >>> class PassThrough(Binarizer):
        ...     def fit_transform(self, X, y=None):
        ...         self._columns = list(X.columns)
        ...         self._is_fitted = True
        ...         return X.astype(bool)
        ...     def transform(self, X):
        ...         return X.astype(bool)
        ...     def get_feature_names_out(self):
        ...         return list(self._columns)
        >>> b = PassThrough()
        >>> b.is_fitted
        False
        >>> out = b.fit_transform(pd.DataFrame({"x": [True, False]}))
        >>> b.is_fitted
        True
        >>> b.get_feature_names_out()
        ['x']
    """

    _is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the binarizer has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Learn the encoding from ``X`` (and optional labels ``y``) and return the
        transformed boolean ``DataFrame``.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned encoding to new data and return a boolean ``DataFrame``."""
        pass

    @abstractmethod
    def get_feature_names_out(self) -> List[str]:
        """
        Return the output feature (column) names in order.

        The returned list is index-aligned with the boolean columns produced by
        ``fit_transform`` / ``transform``, so ``feature_names[i]`` is the name of the
        feature a rule references with ``Literal(value=i)``. Following the scikit-learn
        convention, the names are returned as an ordered ``list[str]``. Implementations
        should raise if called before the binarizer is fitted.
        """
        pass
