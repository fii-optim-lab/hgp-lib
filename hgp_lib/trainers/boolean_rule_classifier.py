from dataclasses import replace
from typing import List, Optional

import numpy as np
import pandas as pd

from ..configs import TrainerConfig, validate_trainer_config
from ..metrics import PopulationHistory
from ..preprocessing import Binarizer, StandardBinarizer
from ..rules import Rule
from ..utils.validation import check_isinstance
from .gp_trainer import GPTrainer


class BooleanRuleClassifier:
    """
    End-to-end classifier that binarizes raw tabular data and evolves a Boolean rule.

    This is the easiest way to go from a raw ``pandas.DataFrame`` to a trained,
    human-readable rule. It owns a :class:`Binarizer` and a :class:`GPTrainer`:
    ``fit`` binarizes the raw features (label-aware) and evolves a rule on them, and
    ``predict`` binarizes new raw data with the same fitted binarizer before evaluating
    the rule. It mirrors the scikit-learn estimator API (``fit``/``predict``) so it can
    be dropped into places that expect an estimator.

    Args:
        trainer_config (TrainerConfig):
            Training configuration (epochs, scorer, evolutionary operators, ...). Its
            nested ``gp_config`` does not need ``train_data``/``train_labels``; they are
            filled from the data passed to ``fit``.
        binarizer (Binarizer | None):
            Binarizer used to turn raw features into boolean columns. When ``None``
            (default), a :class:`StandardBinarizer` with default settings is used. The
            binarizer must not be already fitted.

    Examples:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from hgp_lib import BooleanRuleClassifier
        >>> from hgp_lib.configs import BooleanGPConfig, TrainerConfig
        >>> from hgp_lib.utils.metrics import fast_f1_score
        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, stratify=y, random_state=0
        ... )
        >>> X_train, X_val, y_train, y_val = train_test_split(
        ...     X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
        ... )
        >>> config = TrainerConfig(
        ...     gp_config=BooleanGPConfig(score_fn=fast_f1_score),
        ...     num_epochs=10,
        ...     val_every=5,
        ...     progress_bar=False,
        ... )
        >>> clf = BooleanRuleClassifier(config)
        >>> history = clf.fit(X_train, y_train, X_val, y_val)
        >>> predictions = clf.predict(X_test)
        >>> predictions.shape
        (114,)
        >>> history.best_val_score is not None
        True
        >>> isinstance(clf.format_rule(), str)
        True
    """

    def __init__(
        self, trainer_config: TrainerConfig, binarizer: Optional[Binarizer] = None
    ):
        validate_trainer_config(trainer_config, require_data=False)
        if binarizer is None:
            binarizer = StandardBinarizer()
        else:
            check_isinstance(binarizer, Binarizer)
            if binarizer.is_fitted:
                raise ValueError(
                    "binarizer must not be fitted before passing to BooleanRuleClassifier"
                )

        self.trainer_config = trainer_config
        self.binarizer = binarizer
        self._history: Optional[PopulationHistory] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> PopulationHistory:
        """
        Binarize raw features and evolve a Boolean rule on them.

        The binarizer is fitted on ``X`` (with labels ``y``, enabling supervised binning
        of numeric columns) and the resulting boolean matrix is used to train a
        :class:`GPTrainer`.

        Optionally, a raw validation set ``(X_val, y_val)`` can be supplied. It is
        transformed with the *same* fitted binarizer (no leakage) and used to track a
        validation score during training (every ``val_every`` epochs, as configured in
        the ``TrainerConfig``). When given, it overrides any ``val_data`` already set on
        the trainer configuration.

        Args:
            X (pd.DataFrame): Raw (non-binarized) training features. Columns may be
                boolean, categorical, or numeric.
            y (np.ndarray): Binary training labels, one per row of ``X``.
            X_val (pd.DataFrame | None): Optional raw validation features, with the same
                schema as ``X``. Default: `None`.
            y_val (np.ndarray | None): Optional validation labels. Must be provided if
                and only if ``X_val`` is. Default: `None`.

        Returns:
            PopulationHistory: The training history, whose ``global_best_rule`` is the
                rule used for prediction. When validation data is supplied,
                ``history.best_val_score`` reports the best validation score.

        Raises:
            TypeError: If ``X`` (or ``X_val``, when given) is not a ``pandas.DataFrame``.
            ValueError: If exactly one of ``X_val`` / ``y_val`` is provided.
        """
        check_isinstance(X, pd.DataFrame)
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both be None")
        y = np.asarray(y)

        train_bin = self.binarizer.fit_transform(X, y).to_numpy(dtype=bool)
        gp_config = replace(
            self.trainer_config.gp_config, train_data=train_bin, train_labels=y
        )

        val_kwargs = {}
        if X_val is not None:
            check_isinstance(X_val, pd.DataFrame)
            val_bin = self.binarizer.transform(X_val).to_numpy(dtype=bool)
            val_kwargs = {"val_data": val_bin, "val_labels": np.asarray(y_val)}

        fit_config = replace(self.trainer_config, gp_config=gp_config, **val_kwargs)

        self._history = GPTrainer(fit_config).fit()
        return self._history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels for raw ``X`` using the evolved rule.

        The raw features are binarized with the fitted binarizer, then the best rule
        found during ``fit`` is evaluated on them. Must be called after ``fit``.

        Args:
            X (pd.DataFrame): Raw features with the same columns, in the same order and
                dtypes, as the data used to fit.

        Returns:
            np.ndarray: 1-D boolean array with one prediction per input row.

        Raises:
            TypeError: If ``X`` is not a ``pandas.DataFrame``.
            RuntimeError: If called before ``fit``.
        """
        check_isinstance(X, pd.DataFrame)
        self._check_fitted("predict")
        data = self.binarizer.transform(X).to_numpy(dtype=bool)
        return self._history.global_best_rule.evaluate(data)

    @property
    def rule(self) -> Rule:
        """The best rule found during ``fit``. Requires a fitted classifier."""
        self._check_fitted("rule")
        return self._history.global_best_rule

    @property
    def feature_names(self) -> List[str]:
        """
        The binarized feature names in order (from the fitted binarizer), so that
        ``feature_names[i]`` names the feature a literal references with index ``i``.
        """
        self._check_fitted("feature_names")
        return self.binarizer.get_feature_names_out()

    def format_rule(self) -> str:
        """
        Return the evolved rule as a readable logical expression over the original
        binarized feature names, e.g. ``Or(mean radius < 15.0, ~worst area < 880.0)``.
        """
        return self.rule.to_str(self.feature_names)

    def _check_fitted(self, attr: str) -> None:
        if self._history is None:
            raise RuntimeError(
                f"BooleanRuleClassifier must be fit before accessing '{attr}'"
            )
