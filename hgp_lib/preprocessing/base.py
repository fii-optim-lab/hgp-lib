from abc import ABC, abstractmethod
from typing import Optional, Dict, List

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
        - ``is_fitted`` reports whether the binarizer has been fitted.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from hgp_lib.preprocessing.base import Binarizer
        >>> class PassThrough(Binarizer):
        ...     def fit_transform(self, X, y=None):
        ...         self._is_fitted = True
        ...         return X.astype(bool)
        ...     def transform(self, X):
        ...         return X.astype(bool)
        >>> b = PassThrough()
        >>> b.is_fitted
        False
        >>> _ = b.fit_transform(pd.DataFrame({"x": [True, False]}))
        >>> b.is_fitted
        True
    """

    _is_fitted: bool = False

    @property
    def feature_names(self) -> Dict[int, str]:
        if not self._is_fitted:
            raise ValueError("Binarizer has not been fitted yet.")
        return self._get_feature_names()

    @abstractmethod
    def _get_feature_names(self) -> Dict[int, str]:
        pass

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
