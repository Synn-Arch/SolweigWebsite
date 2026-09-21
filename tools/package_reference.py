"""Package a verified small original CPU oracle for offline differential tests.

Run only in the isolated upstream environment. The resulting archive directory
contains original outputs; candidate code is never imported or executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from verify_boundary_capture import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    environment = json.loads((run / "environment.json").read_text())
    outcome = json.loads((run / "outcome.json").read_text())
    if environment["evidence_class"] != "original_upstream_cpu" or environment["oracle_patch_hash"] is not None:
        raise ValueError("Only unpatched original upstream CPU output is admitted by this packager")
    if outcome["status"] != "executed_not_yet_verified":
        raise ValueError("Oracle did not complete")
    verification = verify(run)
    if verification["timesteps"] != 24:
        raise ValueError("This packager is scoped to the initial 24-step small fixture")
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    # Verify all raw arrays before publishing a self-contained reference directory.
    destination.mkdir(parents=True)
    for name in ("scene", "boundaries", "harness"):
        shutil.copytree(run / name, destination / name)
    for name in ("environment.json", "fixture_hashes.json", "kwargs.json", "outcome.json", "stdout.log", "stderr.log"):
        shutil.copyfile(run / name, destination / name)
    files = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest = {"schema_version": 1, "evidence_class": "original_upstream_cpu_instrumented",
                "source_commit": environment["source_commit"], "oracle_patch_hash": None,
                "fixture_id": "isolated_block_sparse_trees_32x35",
                "producer_run": str(run), "files_sha256": files,
                "verification": verification,
                "limitations": ["One scene/day/configuration on macOS ARM64",
                                "Read-only instrumentation; not a performance baseline",
                                "Scientific quirks preserved; this is an equivalence oracle"]}
    (destination / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(files)} provenance-hashed original reference files at {destination}")


if __name__ == "__main__":
    main()
