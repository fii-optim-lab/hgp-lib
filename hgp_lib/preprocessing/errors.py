class SchemaMismatchError(RuntimeError):
    """
    Data passed to `transform` does not match the schema seen during `fit`.

    Raised when the columns, their dtypes, or the number of produced features
    differ from the fitting data. Distinguishing this from a plain `ValueError`
    lets callers tell "this data does not fit the binarizer" apart from "these
    arguments are invalid", and react by refitting.

    Examples:
        >>> from hgp_lib.preprocessing.errors import SchemaMismatchError
        >>> issubclass(SchemaMismatchError, RuntimeError)
        True
    """
