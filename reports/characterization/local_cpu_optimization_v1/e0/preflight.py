#!/usr/bin/env python3
"""Record source, wheel, dependency, and import provenance for E0."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[4]
E0 = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and not any(".egg-info" in part for part in path.parts)}


def main() -> int:
    source = E0 / "source_snapshot"
    current = ROOT / "src"
    target = E0 / "site-packages"
    wheel = next((E0 / "dist").glob("*.whl"))
    sys.path.insert(0, str(target))
    import solweig_light
    import solweig_light.api
    if "torch" in sys.modules:
        raise RuntimeError("torch imported during package preflight")
    source_manifest, current_manifest = tree_manifest(source), tree_manifest(current)
    source_equal = source_manifest == current_manifest
    package_files = {k.removeprefix("solweig_light/"): v for k, v in tree_manifest(target / "solweig_light").items()}
    package_source = {k: v for k, v in source_manifest.items() if k.startswith("solweig_light/")}
    package_source = {k.removeprefix("solweig_light/"): v for k, v in package_source.items()}
    snapshot_manifest_path = E0 / "source_snapshot_manifest.json"
    snapshot_manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    record = {
        "schema": "local-cpu-optimization-e0-preflight.v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                          ("numpy", "numba", "scipy", "psutil", "GDAL")},
        "source_snapshot": {"path": str(source), "files": len(source_manifest),
                            "sha256": sha(snapshot_manifest_path)},
        "current_source": {"path": str(current), "files": len(current_manifest)},
        "source_snapshot_matches_current": source_equal,
        "wheel": {"path": str(wheel), "bytes": wheel.stat().st_size, "sha256": sha(wheel)},
        "installed_target": str(target),
        "installed_package_origin": str(Path(solweig_light.__file__).resolve()),
        "installed_package_files": len(package_files),
        "installed_package_matches_source": package_files == package_source,
        "torch_imported": "torch" in sys.modules,
        "performance_claim": None,
    }
    if not source_equal or not record["installed_package_matches_source"]:
        raise RuntimeError(json.dumps(record, indent=2, sort_keys=True))
    (E0 / "preflight.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
