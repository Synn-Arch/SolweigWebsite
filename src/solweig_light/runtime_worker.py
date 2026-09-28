"""Subprocess entry point for runtime tile jobs.

This module intentionally delays importing the package pipeline until after
the parent has supplied native thread environment variables.

Two entry modes share this module.  The one-shot entry (``--job``) runs a
single tile job and exits with its return code.  When the scheduler sets
``SOLWEIG_LIGHT_WORKER_POOL`` in the child environment (see
``solweig_light.runtime``), the same process additionally serves job paths
from stdin, one per line, until end-of-input: imports and JIT state persist
across jobs while every tile's scene, state, and output references are
released after each job.  Only bounded immutable resources are retained.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback


# Thread-limit variables for the startup diagnostic snapshot.  The scheduler
# sets these before our first import, so the values are this worker's
# effective native thread budgets.
_THREAD_LIMIT_VARS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _json_error_arg(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_error_arg(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_error_arg(item) for key, item in value.items()}
    return repr(value)


def write_failure(job_path: Path, error: BaseException) -> None:
    payload = {
        "schema_version": 1,
        "exception_type": type(error).__name__,
        "exception_module": type(error).__module__,
        "args": [_json_error_arg(value) for value in error.args],
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    job_path.with_suffix(".failure.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _run_job(job_path: Path, options_data: dict) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        # Keep this import after process startup/environment construction.  The
        # runtime parent sets OMP/MKL/Numba variables in Popen(env=...).
        from .pipeline import run_tile
        from .runtime import RuntimeOptions

        run_tile(**job, runtime=RuntimeOptions(**options_data))
        return 0
    except BaseException as error:
        try:
            write_failure(job_path, error)
        except BaseException:
            # A crashed worker still exits nonzero and is reported by parent.
            pass
        return 1
    finally:
        # Release this tile's scene, state, and output references before the
        # next assignment.  Bounded module-level resources (JIT caches and
        # in-module GeometryStore bounds) are intentionally retained.
        gc.collect()


def _write_done(job_path: Path, code: int) -> None:
    """Atomically report one finished job to the scheduler."""
    marker = job_path.with_suffix(".done")
    staging = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
    staging.write_text(json.dumps({"returncode": code}, sort_keys=True), encoding="utf-8")
    os.replace(staging, marker)


def _redirect_job_streams(job_path: Path) -> None:
    """Own this job's stdout/stderr files; later jobs must not share them."""
    for descriptor, suffix in ((1, ".stdout"), (2, ".stderr")):
        handle = os.open(
            os.fspath(job_path.with_suffix(suffix)),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        os.dup2(handle, descriptor)
        os.close(handle)


def _serve(pool_root: str, job_path: Path, options_data: dict, code: int) -> int:
    """Report the CLI job, then serve scheduler assignments until end-of-input."""
    root = Path(pool_root)
    pid = os.getpid()
    # Startup diagnostics: worker identity plus the thread limits the parent
    # placed in the environment before our first import.
    (root / f"worker-{pid}.pid").write_text(str(pid), encoding="utf-8")
    (root / f"worker-{pid}.threads.json").write_text(
        json.dumps(
            {name: os.environ.get(name, "") for name in _THREAD_LIMIT_VARS},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_done(job_path, code)
    while True:
        line = sys.stdin.readline()
        if not line:  # end-of-input: the scheduler is done with this worker
            return 0
        job_path = Path(line.strip())
        if not job_path.name:
            continue
        _redirect_job_streams(job_path)
        _write_done(job_path, _run_job(job_path, options_data))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run solweig-light tile jobs")
    parser.add_argument("--job", required=True)
    parser.add_argument("--options", required=True)
    args = parser.parse_args(argv)
    options_data = json.loads(args.options)
    job_path = Path(args.job)
    code = _run_job(job_path, options_data)
    pool_root = os.environ.get("SOLWEIG_LIGHT_WORKER_POOL")
    if pool_root is None:
        return code
    return _serve(pool_root, job_path, options_data, code)


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())
