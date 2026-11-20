import sys
import warnings

# Messages of warnings already emitted in this process, used to deduplicate.
_emitted_messages: set[str] = set()


def _was_shown(message: str, category: type[Warning]) -> bool:
    """
    Whether ``warnings.warn`` just showed a warning with this message and category.

    The ``"default"``, ``"module"`` and ``"once"`` actions record a shown warning in the
    ``__warningregistry__`` of the module it is attributed to. ``"ignore"`` and
    ``"always"`` record nothing there; see `_ignored`.
    """
    frame = sys._getframe(2)
    while frame is not None:
        registry = frame.f_globals.get("__warningregistry__")
        if registry:
            for key in registry:
                if isinstance(key, tuple) and key[:2] == (message, category):
                    return True
        frame = frame.f_back
    return False


def _ignored(message: str, category: type[Warning]) -> bool:
    """
    Whether the warning filters ignore this message and category.

    Only filters that apply to every module and line are considered, such as those of
    ``warnings.simplefilter``. Filters for one module (for example the default filter
    that shows ``DeprecationWarning`` in ``__main__``) are detected by `_was_shown`.
    """
    for action, text, filter_category, module, lineno in warnings.filters:
        if (
            module is None
            and lineno == 0
            and issubclass(category, filter_category)
            and (text is None or text.match(message))
        ):
            return action == "ignore"
    return warnings.defaultaction == "ignore"


def warn_once(warning: Warning, stacklevel: int = 1) -> None:
    """
    Emit a warning at most once per process, keyed by its message.

    The warning is passed in already constructed, so its message (including any
    dynamic values) lives on the warning class itself. Once a warning has been shown,
    repeated warnings with the same message are suppressed, also when the warning
    filters change. A warning that the filters hid (for example ``"ignore"``) is not
    recorded, so it is shown the first time the filters allow it.

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
    warnings.warn(warning, stacklevel=stacklevel + 1)
    if _was_shown(message, type(warning)) or not _ignored(message, type(warning)):
        _emitted_messages.add(message)


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
