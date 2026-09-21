"""Subprocess entry point for one runtime tile job.

This module intentionally delays importing the package pipeline until after
the parent has supplied native thread environment variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one solweig-light tile job")
    parser.add_argument("--job", required=True)
    parser.add_argument("--options", required=True)
    args = parser.parse_args(argv)
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    options_data = json.loads(args.options)
    try:
        # Keep this import after process startup/environment construction.  The
        # runtime parent sets OMP/MKL/Numba variables in Popen(env=...).
        from .pipeline import run_tile
        from .runtime import RuntimeOptions

        run_tile(**job, runtime=RuntimeOptions(**options_data))
        return 0
    except BaseException as error:
        try:
            write_failure(Path(args.job), error)
        except BaseException:
            # A crashed worker still exits nonzero and is reported by parent.
            pass
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())
