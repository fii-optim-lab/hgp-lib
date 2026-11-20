"""
Attribute access to the package's modules after a bare ``import hgp_lib``.

Each check runs in a new interpreter, because the other tests import submodules
directly, which makes them reachable as attributes anyway.
"""

import json
import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

import hgp_lib

PACKAGE_DIR = Path(hgp_lib.__file__).parent

# Deprecated shims. They are imported only by code that still uses the 1.x names.
DEPRECATED = {"hgp_lib.metrics", "hgp_lib.utils.metrics"}

# Imported on first access to hgp_lib.evaluation.TorchBackend, so that hgp_lib does not
# import PyTorch.
TORCH = "hgp_lib.evaluation.torch"

# Makes ``import torch`` fail in the child interpreter, as if PyTorch were missing.
HIDE_TORCH = """
from importlib.abc import MetaPathFinder

class HideTorch(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, HideTorch())
"""

# Child prelude: imports hgp_lib and defines the helpers the checks use.
PRELUDE = """
import hgp_lib

def unreachable(names):
    missing = []
    for name in names:
        obj = hgp_lib
        try:
            for part in name.split(".")[1:]:
                obj = getattr(obj, part)
        except AttributeError:
            missing.append(name)
    return missing

def error(code):
    try:
        exec(code)
    except Exception as exc:
        return [type(exc).__name__, str(exc)]
    return None
"""


def module_names() -> list[str]:
    """Every module of the package, read from the file system without importing."""
    names = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def is_torch_module(name: str) -> bool:
    return name == TORCH or name.startswith(TORCH + ".")


def run_python(body: str, hide_torch: bool = False) -> dict:
    """
    Run ``body`` after the prelude in a new interpreter. ``body`` assigns ``result``,
    which is returned through JSON.
    """
    code = "\n".join(
        [
            "import json, sys",
            HIDE_TORCH if hide_torch else "",
            PRELUDE,
            body,
            "print(json.dumps(result))",
        ]
    )
    child = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    if child.returncode != 0:
        raise AssertionError(f"Child interpreter failed:\n{child.stderr}")
    return json.loads(child.stdout.strip().splitlines()[-1])


class TestPackageAttributes(unittest.TestCase):
    def setUp(self):
        names = [name for name in module_names() if name not in DEPRECATED]
        self.current = [name for name in names if not is_torch_module(name)]
        self.torch = [name for name in names if is_torch_module(name)]

    def test_module_list_is_complete(self):
        self.assertIn("hgp_lib.evaluation.numpy.kernels", self.current)
        self.assertIn("hgp_lib.testing", self.current)
        self.assertIn("hgp_lib.evaluation.torch.backend", self.torch)

    def test_current_modules_reachable_without_importing_torch(self):
        result = run_python(
            f"result = {{'unreachable': unreachable({self.current!r}),"
            " 'torch_imported': 'torch' in sys.modules}"
        )
        self.assertEqual(result["unreachable"], [])
        self.assertFalse(result["torch_imported"])

    @unittest.skipUnless(find_spec("torch") is not None, "PyTorch is not installed")
    def test_torch_modules_reachable_after_torch_backend(self):
        result = run_python(
            f"before = unreachable({self.torch!r})\n"
            "hgp_lib.evaluation.TorchBackend\n"
            f"result = {{'before': before, 'after': unreachable({self.torch!r})}}"
        )
        self.assertEqual(result["before"], self.torch)
        self.assertEqual(result["after"], [])

    def test_without_torch(self):
        result = run_python(
            f"result = {{'unreachable': unreachable({self.current!r}),"
            " 'from_import': error('from hgp_lib.evaluation import TorchBackend'),"
            " 'hasattr': error('hasattr(hgp_lib.evaluation, \"TorchBackend\")')}",
            hide_torch=True,
        )
        self.assertEqual(result["unreachable"], [])
        # Decided behavior: both forms raise ImportError with the install hint, so
        # hasattr does not return False.
        for form in ("from_import", "hasattr"):
            with self.subTest(form=form):
                self.assertIsNotNone(result[form])
                name, message = result[form]
                self.assertEqual(name, "ImportError")
                self.assertIn('pip install "hgp-lib[torch]"', message)


if __name__ == "__main__":
    unittest.main()
