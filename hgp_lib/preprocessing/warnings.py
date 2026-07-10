class BinarizerWarning(Warning):
    """Base class for warnings raised by binarizers."""


class StringColumnWarning(BinarizerWarning):
    """
    A string or object column is being treated as categorical.

    Setting the column to a pandas ``category`` dtype makes the intent explicit and
    avoids this warning.

    Examples:
        >>> from hgp_lib.preprocessing.warnings import StringColumnWarning
        >>> str(StringColumnWarning("city"))
        "Column 'city' has a string or object dtype and is treated as categorical. Set it to a 'category' dtype to make this explicit."
    """

    def __init__(self, column: str):
        super().__init__(
            f"Column '{column}' has a string or object dtype and is treated as "
            f"categorical. Set it to a 'category' dtype to make this explicit."
        )


class HighCardinalityWarning(BinarizerWarning):
    """
    An all-distinct categorical column is skipped.

    One-hot encoding a column whose values are all distinct would produce one boolean
    feature per row, which carries no generalization, so the column is dropped.

    Examples:
        >>> from hgp_lib.preprocessing.warnings import HighCardinalityWarning
        >>> str(HighCardinalityWarning("user_id"))
        "Column 'user_id' has all-distinct values and is skipped, since one-hot encoding it would produce one feature per row."
    """

    def __init__(self, column: str):
        super().__init__(
            f"Column '{column}' has all-distinct values and is skipped, since "
            f"one-hot encoding it would produce one feature per row."
        )


class UnseenNaNWarning(BinarizerWarning):
    """
    A column contains NaN at transform time but did not at fit time.

    No new column is created for these values, so the affected rows fall into no bin
    and are encoded as all-false for that column.

    Examples:
        >>> from hgp_lib.preprocessing.warnings import UnseenNaNWarning
        >>> str(UnseenNaNWarning("amount"))
        "Column 'amount' has NaN values that were not seen during fit; these rows are encoded as all-false."
    """

    def __init__(self, column: str):
        super().__init__(
            f"Column '{column}' has NaN values that were not seen during fit; "
            f"these rows are encoded as all-false."
        )
