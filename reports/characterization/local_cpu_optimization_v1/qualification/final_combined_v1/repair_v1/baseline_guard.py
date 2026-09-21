"""Same-process guard for the current checkout baseline source run."""
from pathlib import Path
import os
import sys


def pytest_configure(config):
    import solweig_light
    target = Path(os.environ["SOLWEIG_BASELINE_SOURCE_TARGET"]).resolve()
    package = Path(solweig_light.__file__).resolve()
    if target not in package.parents:
        raise RuntimeError(f"baseline package escaped source target: {package}")
    if "torch" in sys.modules:
        raise RuntimeError("torch imported before baseline source test")
