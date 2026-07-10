import warnings
from typing import Set

# Messages of warnings already emitted in this process, used to deduplicate.
_emitted_messages: Set[str] = set()


def warn_once(warning: Warning) -> None:
    """
    Emit a warning at most once per process, keyed by its message.

    The warning is passed in already constructed, so its message (including any
    dynamic values) lives on the warning class itself. Repeated warnings with the
    same message are suppressed, independent of the active warning filters.

    Args:
        warning (Warning):
            A constructed warning instance to emit.

    Examples:
        >>> import warnings
        >>> from hgp_lib.utils.warnings import warn_once
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     warn_once(UserWarning("a unique warn_once doctest message"))
        ...     [str(entry.message) for entry in caught]
        ['a unique warn_once doctest message']
    """
    message = str(warning)
    if message in _emitted_messages:
        return
    _emitted_messages.add(message)
    warnings.warn(warning, stacklevel=2)
