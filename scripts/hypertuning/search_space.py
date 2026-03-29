"""
Load hyperparameter search ranges from a YAML config file.

Each YAML entry maps directly to the positional and keyword args of
a ``trial.suggest_*`` call.

Range params (list of scalars + optional dict kwargs)::

    population_size:
      - 50
      - 150
      - step: 25

Categorical params (list containing one list)::

    regeneration:
      - [true, false]

Fixed params (plain scalar)::

    selection_type: "tournament"
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _parse_entry(raw: Any) -> Tuple[tuple, dict]:
    """
    Parse a single YAML entry into ``(args, kwargs)``.

    - Plain scalar  → ``((value,), {})``  (fixed value, single-element args)
    - List           → positional scalars become args, dict items become kwargs
    """
    if not isinstance(raw, list):
        return (raw,), {}

    args = []
    kwargs = {}
    for item in raw:
        if isinstance(item, dict):
            kwargs.update(item)
        else:
            args.append(item)
    return tuple(args), kwargs


def load_search_space(
    config_path: str | Path,
) -> Dict[str, Tuple[tuple, dict]]:
    """
    Load a YAML search-space config.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dict mapping parameter names to ``(args, kwargs)`` tuples.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {name: _parse_entry(spec) for name, spec in raw.items()}
