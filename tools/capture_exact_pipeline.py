#!/usr/bin/env python3
"""Capture an exact, streaming correctness trace of the real tile pipeline.

This harness is deliberately unsuitable for performance measurement: it wraps
the numerical return, checkpoint, and optional writer boundaries and hashes
every captured value before allowing the original call sequence to continue.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
MEMORY_BUDGET_BYTES = 12 * 1024**3
INVENTORY_EXCLUDES = {"__pycache__"}
INVENTORY_SUFFIX_EXCLUDES = {".pyc", ".nbi", ".nbc"}


def qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in INVENTORY_EXCLUDES for part in relative.parts):
            continue
        if not path.is_file() or path.suffix in INVENTORY_SUFFIX_EXCLUDES:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def inventory_digest(records: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def assert_disjoint(path: Path, protected: Iterable[Path]) -> None:
    resolved = path.expanduser().resolve()
    for item in protected:
        protected_path = item.resolve()
        if resolved == protected_path or resolved.is_relative_to(protected_path):
            raise ValueError(f"run directory must be outside protected input: {protected_path}")


def load_profile_tool(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("exact_trace_profile_p7", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load profile helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "prepared_job", None)):
        raise RuntimeError(f"profile helper has no prepared_job(): {path}")
    return module


def package_version() -> str | None:
    try:
        return importlib.metadata.version("solweig-light")
    except importlib.metadata.PackageNotFoundError:
        return None


class ExactValueEncoder:
    """Describe values without retaining array payloads after each event."""

    def __init__(self, numpy_module: Any, chunk_bytes: int) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.np = numpy_module
        self.chunk_bytes = chunk_bytes

    def encode(self, value: Any) -> dict[str, Any]:
        np = self.np
        if isinstance(value, np.ndarray):
            return self._array(value)
        if isinstance(value, np.generic):
            return self._numpy_scalar(value)
        if value is None:
            return {"kind": "none", "type": qualified_type(value)}
        if isinstance(value, bool):
            return {"kind": "python_bool", "type": qualified_type(value), "value": value}
        if isinstance(value, int):
            return {
                "kind": "python_int",
                "type": qualified_type(value),
                "decimal": str(value),
            }
        if isinstance(value, float):
            return self._python_float(value)
        if isinstance(value, complex):
            return {
                "kind": "python_complex",
                "type": qualified_type(value),
                "real_binary64_be_hex": struct.pack(">d", value.real).hex(),
                "imag_binary64_be_hex": struct.pack(">d", value.imag).hex(),
            }
        if isinstance(value, str):
            return {"kind": "python_str", "type": qualified_type(value), "value": value}
        if isinstance(value, bytes):
            return {
                "kind": "python_bytes",
                "type": qualified_type(value),
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": "sequence",
                "type": qualified_type(value),
                "length": len(value),
                "items": [self.encode(item) for item in value],
            }
        raise TypeError(f"unsupported captured value type: {qualified_type(value)}")

    def _iter_array_chunks(self, array: Any) -> Iterable[Any]:
        itemsize = max(1, int(array.dtype.itemsize))
        buffer_items = max(1, self.chunk_bytes // itemsize)
        iterator = self.np.nditer(
            array,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            order="C",
            buffersize=buffer_items,
        )
        for chunk in iterator:
            captured = self.np.asarray(chunk)
            if captured.dtype != array.dtype:
                raise TypeError(
                    f"array iterator changed dtype from {array.dtype!r} to {captured.dtype!r}"
                )
            yield captured

    def _dtype_record(self, dtype: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"str": dtype.str}
        if dtype.fields is not None:
            record["descr"] = dtype.descr
        return record

    def _array(self, array: Any) -> dict[str, Any]:
        np = self.np
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot be captured as stable raw bytes")
        raw_digest = hashlib.sha256()
        mask_names = ("finite", "nan", "posinf", "neginf", "negative_zero")
        mask_digests = {name: hashlib.sha256() for name in mask_names}
        mask_counts = {name: 0 for name in mask_names}
        is_float = array.dtype.kind == "f"

        for chunk in self._iter_array_chunks(array):
            raw_digest.update(chunk.tobytes(order="C"))
            if is_float:
                masks = {
                    "finite": np.isfinite(chunk),
                    "nan": np.isnan(chunk),
                    "posinf": np.isposinf(chunk),
                    "neginf": np.isneginf(chunk),
                    "negative_zero": np.equal(chunk, 0) & np.signbit(chunk),
                }
                for name, mask in masks.items():
                    mask_digests[name].update(mask.tobytes(order="C"))
                    mask_counts[name] += int(np.count_nonzero(mask))

        record: dict[str, Any] = {
            "kind": "ndarray",
            "type": qualified_type(array),
            "dtype": self._dtype_record(array.dtype),
            "shape": list(array.shape),
            "ndim": int(array.ndim),
            "size": int(array.size),
            "strides": list(array.strides),
            "c_contiguous": bool(array.flags.c_contiguous),
            "f_contiguous": bool(array.flags.f_contiguous),
            "byte_order": "logical-C",
            "raw_sha256": raw_digest.hexdigest(),
        }
        if is_float:
            record["float_masks"] = {
                name: {
                    "count": mask_counts[name],
                    "uint8_logical_c_sha256": mask_digests[name].hexdigest(),
                }
                for name in mask_names
            }
        return record

    def _numpy_scalar(self, value: Any) -> dict[str, Any]:
        np = self.np
        array = np.asarray(value)
        raw = array.tobytes(order="C")
        record: dict[str, Any] = {
            "kind": "numpy_scalar",
            "type": qualified_type(value),
            "dtype": self._dtype_record(array.dtype),
            "raw_hex": raw.hex(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
        if array.dtype.kind == "f":
            scalar = array.reshape(()).item()
            record["float_masks"] = self._float_classification(scalar)
        return record

    def _python_float(self, value: float) -> dict[str, Any]:
        raw = struct.pack(">d", value)
        return {
            "kind": "python_float",
            "type": qualified_type(value),
            "binary64_be_hex": raw.hex(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "float_masks": self._float_classification(value),
        }

    @staticmethod
    def _float_classification(value: float) -> dict[str, bool]:
        return {
            "finite": math.isfinite(value),
            "nan": math.isnan(value),
            "posinf": math.isinf(value) and value > 0,
            "neginf": math.isinf(value) and value < 0,
            "negative_zero": value == 0 and math.copysign(1.0, value) < 0,
        }


class TraceSink:
    def __init__(self, path: Path, encoder: ExactValueEncoder) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.encoder = encoder
        self.stream = path.open("xb")
        self.sequence = 0
        self.previous_payload_sha256: str | None = None
        self.counts: Counter[str] = Counter()
        self.closed = False

    def capture_fields(
        self,
        event: str,
        timestep: int,
        container: Any,
        names: Sequence[str],
        values: Sequence[Any],
        **metadata: Any,
    ) -> None:
        if len(names) != len(values):
            raise ValueError(
                f"{event} field count mismatch: {len(names)} names, {len(values)} values"
            )
        payload: dict[str, Any] = {
            "schema": 1,
            "sequence": self.sequence,
            "event": event,
            "timestep": int(timestep),
            "container_type": qualified_type(container),
            "fields": [
                {"ordinal": ordinal, "name": name, "value": self.encoder.encode(value)}
                for ordinal, (name, value) in enumerate(zip(names, values))
            ],
        }
        payload.update(metadata)
        payload_bytes = canonical_bytes(payload)
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        record = dict(payload)
        record["previous_payload_sha256"] = self.previous_payload_sha256
        record["payload_sha256"] = payload_sha256
        self.stream.write(canonical_bytes(record) + b"\n")
        self.stream.flush()
        self.previous_payload_sha256 = payload_sha256
        self.sequence += 1
        self.counts[event] += 1

    def finish(self) -> dict[str, Any]:
        if not self.closed:
            self.stream.close()
            self.closed = True
        return {
            "records": self.sequence,
            "event_counts": dict(sorted(self.counts.items())),
            "last_payload_sha256": self.previous_payload_sha256,
            "jsonl_bytes": self.path.stat().st_size,
            "jsonl_sha256": sha256_file(self.path),
        }


def configure_process(source: Path, threads: int, numba_cache: Path) -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    for variable in THREAD_ENV:
        os.environ[variable] = str(threads)
    source_text = str(source)
    os.environ["PYTHONPATH"] = source_text
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)


def module_origin(module: Any) -> str:
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise RuntimeError(f"module has no file origin: {module!r}")
    return str(Path(origin).resolve())


def require_source_origin(origin: str, source: Path) -> None:
    path = Path(origin).resolve()
    if not path.is_relative_to(source):
        raise RuntimeError(f"import escaped candidate source tree: {path}")


def make_parser(default_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=default_root / "tests/reference/small_original_cpu/scene",
    )
    parser.add_argument("--source", type=Path, default=default_root / "src")
    parser.add_argument(
        "--profile-tool",
        type=Path,
        default=default_root / "tools/profile_p7_pipeline.py",
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--block-pixels", type=int, default=128)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument(
        "--geometry-state",
        choices=("fixture", "cold"),
        default="cold",
        help="retain fixture geometry products or remove the copied SVF directory",
    )
    parser.add_argument(
        "--capture-writer-outputs",
        action="store_true",
        help="capture every requested field at the boundary before writer.write",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the instrumented pipeline; without this flag, only emit a plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    harness_path = Path(__file__).resolve()
    default_root = Path.cwd().resolve()
    args = make_parser(default_root).parse_args(argv)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.block_pixels <= 0:
        raise ValueError("--block-pixels must be positive")
    if args.chunk_bytes <= 0:
        raise ValueError("--chunk-bytes must be positive")

    fixture = require_directory(args.fixture, "fixture")
    source = require_directory(args.source, "source")
    profile_tool = require_file(args.profile_tool, "profile tool")
    if not (source / "solweig_light/pipeline.py").is_file():
        raise ValueError(f"source lacks solweig_light/pipeline.py: {source}")
    run = args.run.expanduser().resolve()
    assert_disjoint(run, (fixture, source))
    if run.exists():
        raise FileExistsError(f"run directory already exists: {run}")

    source_before = inventory(source)
    fixture_before = inventory(fixture)
    static_binding = {
        "harness": {"path": str(harness_path), "sha256": sha256_file(harness_path)},
        "profile_tool": {"path": str(profile_tool), "sha256": sha256_file(profile_tool)},
        "source": {
            "path": str(source),
            "inventory": source_before,
            "inventory_sha256": inventory_digest(source_before),
        },
        "fixture": {
            "path": str(fixture),
            "inventory": fixture_before,
            "inventory_sha256": inventory_digest(fixture_before),
        },
    }
    plan = {
        "schema": 1,
        "evidence_class": "candidate_exact_correctness_trace",
        "performance_evidence": False,
        "must_not_be_used_for_performance": True,
        "instrumentation_effect": (
            "Every captured value is traversed and hashed synchronously; runtime and memory "
            "measurements from this process are invalid as performance evidence."
        ),
        "run": str(run),
        "execute": bool(args.execute),
        "geometry_state": args.geometry_state,
        "capture_writer_outputs": bool(args.capture_writer_outputs),
        "runtime": {
            "workers": 1,
            "threads_per_worker": args.threads,
            "cpu_budget": args.threads,
            "block_pixels": args.block_pixels,
            "memory_budget_bytes": MEMORY_BUDGET_BYTES,
            "checkpoint_interval": 1,
            "resume": False,
        },
        "capture": {
            "chunk_bytes": args.chunk_bytes,
            "array_byte_order": "logical-C",
            "raw_array_hash": "sha256",
            "float_masks": ["finite", "nan", "posinf", "neginf", "negative_zero"],
        },
        "binding": static_binding,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0

    run.mkdir(parents=True, exist_ok=False)
    trace_dir = run / "trace"
    trace_dir.mkdir()
    atomic_json(trace_dir / "plan.json", plan)

    setup = run / "setup"
    scene = setup / "scene"
    shutil.copytree(fixture, scene)
    shutil.rmtree(scene / "output_folder", ignore_errors=True)
    shutil.rmtree(scene / ".solweig-light", ignore_errors=True)
    if args.geometry_state == "cold":
        shutil.rmtree(scene / "processed_inputs/SVF", ignore_errors=True)
        (scene / "processed_inputs/SVF").mkdir(parents=True, exist_ok=True)
    copied_input = inventory(scene)

    numba_cache = setup / "numba_cache"
    numba_cache.mkdir()
    configure_process(source, args.threads, numba_cache)

    # Numerical imports occur only after all thread and module-path controls are set.
    import numpy as np
    import solweig_light
    from solweig_light import pipeline, persistence
    from solweig_light.models import SimulationState
    from solweig_light.runtime import RuntimeOptions
    import numba

    if numba.get_num_threads() != args.threads:
        raise RuntimeError("Actual Numba thread limit differs from the requested limit")

    origins = {
        "solweig_light": module_origin(solweig_light),
        "pipeline": module_origin(pipeline),
        "persistence": module_origin(persistence),
    }
    for origin in origins.values():
        require_source_origin(origin, source)

    profile_module = load_profile_tool(profile_tool)
    job = profile_module.prepared_job(scene)
    met_lines = Path(job["paths"]["metfiles"]).read_text(encoding="utf-8").splitlines()
    timeline_rows = sum(1 for line in met_lines if line.strip()) - 1
    if timeline_rows <= 0:
        raise ValueError("prepared meteorological input has no data rows")
    runtime = RuntimeOptions(
        cache_dir=str(setup / "geometry_cache"),
        legacy_cache_policy="recompute",
        cache_enabled=True,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
        cpu_budget=args.threads,
        workers=1,
        threads_per_worker=args.threads,
        block_pixels=args.block_pixels,
        checkpoint_interval=1,
        resume=False,
    )

    sink = TraceSink(
        trace_dir / "records.jsonl",
        ExactValueEncoder(np, chunk_bytes=args.chunk_bytes),
    )
    original_calc = pipeline.Solweig_2022a_calc
    original_write = persistence.TransactionalOutputs.write
    original_checkpoint = persistence.TransactionalOutputs.checkpoint

    def captured_calc(*call_args: Any, **call_kwargs: Any) -> Any:
        result = original_calc(*call_args, **call_kwargs)
        timestep = int(call_kwargs["i"])
        names = tuple(pipeline.RETURN_NAMES)
        values = tuple(result)
        sink.capture_fields("engine_return", timestep, result, names, values)
        return result

    def captured_write(writer: Any, timestep: int, fields: dict[str, Any]) -> Any:
        if args.capture_writer_outputs:
            requested_names = tuple(writer.fields)
            requested_values = tuple(fields[name] for name in requested_names)
            sink.capture_fields(
                "writer_inputs",
                int(timestep),
                fields,
                requested_names,
                requested_values,
                requested_fields=list(requested_names),
            )
        return original_write(writer, timestep, fields)

    def captured_checkpoint(writer: Any, next_timestep: int, state: Any) -> Any:
        if not isinstance(state, SimulationState):
            raise TypeError(f"unexpected checkpoint state type: {qualified_type(state)}")
        state_names = tuple(field.name for field in dataclasses.fields(SimulationState))
        state_values = tuple(getattr(state, name) for name in state_names)
        sink.capture_fields(
            "state_checkpoint",
            int(next_timestep) - 1,
            state,
            state_names,
            state_values,
            next_timestep=int(next_timestep),
        )
        return original_checkpoint(writer, next_timestep, state)

    manifest: dict[str, Any] = {
        "schema": 1,
        "evidence_class": "candidate_exact_correctness_trace",
        "performance_evidence": False,
        "must_not_be_used_for_performance": True,
        "status": "running",
        "plan": plan,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_version": package_version(),
            "module_origins": origins,
            "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
            "pythonpath": os.environ.get("PYTHONPATH"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
            "actual_numba_threads": numba.get_num_threads(),
        },
        "prepared_job": {
            "timeline_rows": timeline_rows,
            "selected_date_str": job["selected_date_str"],
            "tile": job["tile"],
            "save_flags": dict(job["flags"]),
        },
        "copied_scene_before_run": {
            "path": str(scene),
            "inventory": copied_input,
            "inventory_sha256": inventory_digest(copied_input),
        },
    }
    atomic_json(trace_dir / "manifest.json", manifest)

    exit_code = 0
    pipeline.Solweig_2022a_calc = captured_calc
    persistence.TransactionalOutputs.write = captured_write
    persistence.TransactionalOutputs.checkpoint = captured_checkpoint
    try:
        pipeline.run_tile(**job, runtime=runtime)
        expected_counts = {
            "engine_return": timeline_rows,
            "state_checkpoint": timeline_rows,
        }
        if args.capture_writer_outputs:
            expected_counts["writer_inputs"] = timeline_rows
        actual_counts = dict(sink.counts)
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"trace event count mismatch: expected {expected_counts}, got {actual_counts}"
            )
        manifest["status"] = "complete"
    except BaseException as error:
        exit_code = 1
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": qualified_type(error),
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        pipeline.Solweig_2022a_calc = original_calc
        persistence.TransactionalOutputs.write = original_write
        persistence.TransactionalOutputs.checkpoint = original_checkpoint
        manifest["trace"] = sink.finish()
        source_after = inventory(source)
        fixture_after = inventory(fixture)
        manifest["immutability_checks"] = {
            "source_before_sha256": inventory_digest(source_before),
            "source_after_sha256": inventory_digest(source_after),
            "source_unchanged": source_before == source_after,
            "fixture_before_sha256": inventory_digest(fixture_before),
            "fixture_after_sha256": inventory_digest(fixture_after),
            "fixture_unchanged": fixture_before == fixture_after,
        }
        if not manifest["immutability_checks"]["source_unchanged"]:
            manifest["status"] = "failed"
            manifest.setdefault("failure", {})["source_mutated"] = True
            exit_code = 1
        if not manifest["immutability_checks"]["fixture_unchanged"]:
            manifest["status"] = "failed"
            manifest.setdefault("failure", {})["fixture_mutated"] = True
            exit_code = 1
        atomic_json(trace_dir / "manifest.json", manifest)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
