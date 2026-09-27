import sys
import warnings

# Messages of warnings already emitted in this process, used to deduplicate.
_emitted_messages: set[str] = set()


def warn_once(warning: Warning, stacklevel: int = 1) -> None:
    """
    Emit a warning at most once per process, keyed by its message.

    The warning is passed in already constructed, so its message (including any
    dynamic values) lives on the warning class itself. Repeated warnings with the
    same message are suppressed, independent of the active warning filters.

    Args:
        warning (Warning):
            A constructed warning instance to emit.
        stacklevel (int):
            Which caller the warning is attributed to, as in `warnings.warn`, counted
            from the caller of ``warn_once``. ``1`` points at the caller itself, ``2`` at
            its caller. Deprecations should point at the user's line, otherwise Python
            hides ``DeprecationWarning`` raised inside library code. Default: `1`.

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
    warnings.warn(warning, stacklevel=stacklevel + 1)


def warn_moved(old: str, new: str) -> None:
    """
    Warn once that ``old`` is deprecated in favor of ``new``.

    Meant to be called from a module-level ``__getattr__`` that serves a moved name.
    The warning is attributed to the line importing the name: ``from package import
    name`` first looks the name up from inside ``importlib``, and those frames are
    skipped.

    Args:
        old (str): The deprecated import path, e.g. ``"hgp_lib.metrics.RunResult"``.
        new (str): The import path to use instead.
    """
    frame = sys._getframe(2)  # the frame that looked up the name
    skipped = 0
    while frame is not None and frame.f_code.co_filename.startswith(
        "<frozen importlib"
    ):
        frame = frame.f_back
        skipped += 1
    warn_once(
        DeprecationWarning(
            f"{old} is deprecated and will be removed in a future 2.x release. "
            f"Use {new} instead."
        ),
        stacklevel=3 + skipped,
    )
