import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from hgp_lib.utils.validation import check_isinstance

from .base import Binarizer


class SklearnBinarizer(Binarizer):
    """
    Adapter that lets a scikit-learn transformer be used as a :class:`Binarizer`.

    It wraps a transformer that outputs a dense one-hot or otherwise binary array, for
    example ``KBinsDiscretizer(encode="onehot-dense")``, and returns a boolean
    ``DataFrame`` with readable column names. This makes scikit-learn discretizers
    interchangeable with :class:`StandardBinarizer`, including inside ``GPBenchmarker``.

    The wrapped transformer must implement ``fit_transform(X, y)`` and ``transform(X)``.
    Column names are taken from ``get_feature_names_out`` when available, otherwise they
    are generated positionally.

    Args:
        transformer:
            An unfitted scikit-learn transformer producing a binary array.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.preprocessing import KBinsDiscretizer
        >>> from hgp_lib.preprocessing import SklearnBinarizer
        >>> disc = KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="uniform")
        >>> b = SklearnBinarizer(disc)
        >>> out = b.fit_transform(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}))
        >>> bool(out.to_numpy().dtype == bool)
        True
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self._columns: list[str] | None = None
        self._is_fitted = False

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray | None = None
    ) -> pd.DataFrame:
        check_isinstance(X, pd.DataFrame)
        array = np.asarray(self.transformer.fit_transform(X, y))
        self._columns = self._resolve_columns(X, array.shape[1])
        self._is_fitted = True
        return pd.DataFrame(array.astype(bool), columns=self._columns, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_isinstance(X, pd.DataFrame)
        if not self._is_fitted:
            raise NotFittedError("Binarizer must be fitted before calling transform")
        array = np.asarray(self.transformer.transform(X))
        return pd.DataFrame(array.astype(bool), columns=self._columns, index=X.index)

    def get_feature_names_out(self) -> list[str]:
        """
        Return the output column names in order (see :meth:`Binarizer.get_feature_names_out`).

        Names come from the wrapped transformer's ``get_feature_names_out`` when
        available, otherwise they are generated positionally.

        Returns:
            list[str]: The boolean output column names.

        Raises:
            NotFittedError: If the binarizer has not been fitted yet.
        """
        if not self._is_fitted:
            raise NotFittedError(
                "Binarizer must be fitted before calling get_feature_names_out"
            )
        return list(self._columns)

    def _resolve_columns(self, X: pd.DataFrame, n_features: int) -> list[str]:
        if hasattr(self.transformer, "get_feature_names_out"):
            return [
                str(c) for c in self.transformer.get_feature_names_out(list(X.columns))
            ]
        return [f"feature_{i}" for i in range(n_features)]
