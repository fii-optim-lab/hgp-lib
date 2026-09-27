import importlib.util

# The PyTorch backend is optional: skip its doctests when PyTorch is not installed.
collect_ignore = []
if importlib.util.find_spec("torch") is None:
    collect_ignore.append("src/hgp_lib/evaluation/torch")
