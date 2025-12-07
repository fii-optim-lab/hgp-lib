# Contributing to hgp-lib

## Setup

Install the development environment:

```bash
pip install -e .[dev]
```

## Contributing Guidelines

1. **Pull Requests**: All contributions must be made through pull requests.

2. **Documentation**: Add docstrings and doctests where possible to maintain code clarity and provide usage examples.

3. **Testing**: Write unit tests for new functionality to ensure code reliability.

## Pre-commit Checklist

Before each commit, run the following commands in order:

```bash
# 1. Auto-fix linting errors
ruff check --fix

# 2. Check for critical errors (fix manually if any appear)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 3. Format code
ruff format

# 4. Optional: Check code complexity (not mandatory to fix)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Combined check
ruff check --fix && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics && ruff format && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Following these steps ensures consistent code quality and style across the project.
