"""Same-process installed-wheel and source-target guard for final evidence."""
from pathlib import Path
import os


def pytest_configure(config):
    import solweig_light
    target = Path(os.environ["SOLWEIG_INSTALLED_TARGET"]).resolve()
    package = Path(solweig_light.__file__).resolve()
    if target not in package.parents:
        raise RuntimeError(f"package escaped explicit installed target: {package}")
    if "torch" in __import__("sys").modules:
        raise RuntimeError("torch imported before installed core tests")
    for name in ("solweig_light.api", "solweig_light.pipeline", "solweig_light.runtime"):
        module = __import__(name, fromlist=["*"])
        origin = Path(module.__file__).resolve()
        if target not in origin.parents:
            raise RuntimeError(f"module escaped explicit installed target: {name}={origin}")
