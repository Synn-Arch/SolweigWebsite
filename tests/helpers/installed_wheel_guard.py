"""Pytest guard that binds release-CI tests to the installed wheel."""
from __future__ import annotations

from pathlib import Path


def pytest_configure(config):
    import solweig_light

    package = Path(solweig_light.__file__).resolve()
    checkout_source = Path(__file__).resolve().parents[2] / "src"
    if checkout_source == package or checkout_source in package.parents:
        raise RuntimeError(f"release CI imported checkout source: {package}")
    if "site-packages" not in package.parts:
        raise RuntimeError(f"release CI did not import an installed wheel: {package}")

