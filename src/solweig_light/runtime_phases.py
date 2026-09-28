"""Private geometry precompute phase adapter for v6 phase scheduling (C6-40).

One cold logical tile's numerical geometry is produced once through the
shared C6-10 recipe while a K-tile batch runs ``ceil(K / W_g)`` production
widths instead of K serial productions.  The adapter runs the GEOMETRY
construction phase concurrently across spatial tiles and publishes the
results in the ORIGINAL SORTED TILE ORDER, under the C6-42 phase-aware
memory admission and real native thread caps, as a complete stage barrier
before any simulation job can start.

This module is inert until the integrator wires it (see
``optimization_v6_continue/evidence/phases/integration_recipe_C6-40.diff``);
nothing in ``src`` imports it.

Contract mapping (C6-03 m3 §7, binding on this adapter):

1. Original tile order.  Jobs carry the position in the consumer's sort as
   ``order``; dispatch is strictly ascending in ``order`` (m3: dispatch
   ``next_index`` ascending) and publication commits only the contiguous
   staged prefix, so the published set is always a prefix of the original
   order regardless of worker completion order (m3 §5).
2. Transactional per-tile publication.  A worker stages one validated,
   atomically written record per tile; the parent commits tile records at
   the phase publication boundary journal-first, manifest-last, one atomic
   rename per tile (m3 §2).  A tile with no committed record has no partial
   public state.
3. First-failure rule.  On the first observed failure dispatch stops,
   started lower-index jobs get a bounded grace to report their own recorded
   failures, the contiguous staged prefix below the numerically first
   recorded failure is committed, and every live child is reaped with
   ``terminate=True`` (m3 §3/§7.3).  The raised error is that failure
   rebuilt through the existing ``runtime._child_exception``, so the public
   error type and message are the serial pipeline's own (m3 §7.4: never
   plain pickling).  Committed lower-index tiles keep their records; the
   failed tile and every later tile stay private, and the adapter-owned
   staging tree (their private work) is safely cleaned.
4. ``ResourceAdmissionError`` stays in-process, parent-side: admission runs
   via ``runtime_memory.plan_phase_admission`` before any child exists
   (m7 §2), and a width the plan does not admit is never dispatched.
5. Barrier.  ``execute_geometry_phase`` returns only after every tile is
   committed and the phase manifest is written last, so no simulation can
   be admitted earlier (m3 §4).  No overlap credit is taken.
6. Stage thread caps.  Every worker is a child process spawned with the
   native thread variables in ``env=`` before its first import (the C6-01
   finding: ``RuntimeOptions`` does not set Numba's mask in-process); the
   child pins and records the actual native mask at kernel entry, and the
   parent asserts configured == actual before committing the tile.
   ``GDAL_CACHEMAX`` is set per child to the per-worker allowance charged by
   the admission calculator (m7 §1 W3).

Boundary decisions (D04):

- The adapter publishes ready native-handle records at the phase boundary;
  legacy TIFF/ZIP/NPZ export publication remains owned by the existing
  ordered serial loops, which consume the ready handles afterwards.
- ``cache_enabled`` is required: workers produce through the shared
  ``GeometryStore`` under the C6-10 recipe's native key, so one production
  serves both consumers.  The cache-disabled mode retains the sequential
  route and is refused here explicitly, never silently degraded.
- Parent-side recipe rebuilding fingerprints each job's source rasters
  before dispatch (input validation up front, mirroring the serial guard)
  and again at commit; fingerprinting is streamed with the store's bounded
  workspace, no arrays are retained.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from .runtime import (
    RuntimeOptions,
    TileExecutionError,
    _child_environment,
    _json_safe,
    _job_failure,
    _read_done,
    _reap_worker,
    _WORKER_POOL_ENV,
    get_runtime_options,
)
from .runtime_memory import (
    MIB,
    PHASE_GEOMETRY,
    PhaseAdmissionPlan,
    PhaseJob,
    admission_inputs_from_options,
    plan_phase_admission,
)

__all__ = [
    "GEOMETRY_PHASE_WORKER",
    "GeometryPhaseJob",
    "GeometryPhaseOutcome",
    "GeometryTileResult",
    "PHASE_STAGE_POLICY",
    "PHASE_STAGE_SCHEMA",
    "execute_geometry_phase",
    "geometry_phase_jobs",
]


GEOMETRY_PHASE_WORKER = "solweig_light.runtime_phases"
PHASE_STAGE_SCHEMA = "sw6-geometry-phase-stage-v1"
PHASE_STAGE_POLICY = "ordered-prefix-publication-v1"
PHASE_OUTCOME_SCHEMA = "sw6-geometry-phase-outcome-v1"

# Thread-limit variables recorded by the child at startup, the identical set
# to runtime_worker._THREAD_LIMIT_VARS; this module's top level stays
# stdlib-only so the child's pre-import environment is the scheduler's env=.
_THREAD_LIMIT_VARS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class GeometryPhaseJob:
    """JSON-safe geometry phase job (D09 ``PhaseJob`` scheduling dimension).

    ``order`` is the tile's position in the consumer's original sorted order
    (zero-based, contiguous); dispatch and publication both follow it.  The
    memory dimension (shape, visibility mode, publication window) is carried
    by the C6-42 descriptor derived in ``memory_job``.
    """

    order: int
    tile: str
    paths: Mapping[str, str]
    patch_option: int
    rows: int
    cols: int
    patches: int
    windchannels: int
    block_pixels: int
    visibility_mode: str = "unknown_cold"
    dense_fallback: bool = False
    export_overlap: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError(f"order must be a non-negative integer; got {self.order!r}")
        if not isinstance(self.tile, str) or not self.tile:
            raise ValueError("tile must be a non-empty string")
        paths = {str(name): str(path) for name, path in dict(self.paths).items()}
        for name in ("Building_DSM", "Trees", "DEM"):
            if not paths.get(name):
                raise ValueError(f"{self.tile}: missing numerical geometry input {name!r}")
        # A plain dict (JSON round-trips it losslessly); read-only by
        # convention exactly like the recipe identity payload.
        object.__setattr__(self, "paths", paths)
        _positive_int(self.patch_option, "patch_option")

    @property
    def memory_job(self) -> PhaseJob:
        """The C6-42 memory descriptor this job reserves under."""
        return PhaseJob(
            PHASE_GEOMETRY,
            self.rows,
            self.cols,
            patches=self.patches,
            windchannels=self.windchannels,
            block_pixels=self.block_pixels,
            visibility_mode=self.visibility_mode,
            dense_fallback=self.dense_fallback,
            export_overlap=self.export_overlap,
            tile=self.tile,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            order=self.order,
            tile=self.tile,
            paths=dict(self.paths),
            patch_option=self.patch_option,
            rows=self.rows,
            cols=self.cols,
            patches=self.patches,
            windchannels=self.windchannels,
            block_pixels=self.block_pixels,
            visibility_mode=self.visibility_mode,
            dense_fallback=self.dense_fallback,
            export_overlap=self.export_overlap,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryPhaseJob":
        return cls(**dict(payload))


def geometry_phase_jobs(
    ordered_tiles: Sequence[Mapping[str, Any]],
    *,
    patch_option: int,
    windchannels: int,
    block_pixels: int,
    visibility_mode: str = "unknown_cold",
    dense_fallback: bool = False,
    export_overlap: bool = True,
) -> tuple[GeometryPhaseJob, ...]:
    """Build contiguous ``order`` values over tiles already in consumer order.

    ``ordered_tiles`` must be the original sorted sequence of per-tile
    descriptors, each a mapping with ``tile``, ``paths`` and the shape fields
    (``rows``/``cols``/``patches``) — exactly the order the existing serial
    loops dispatch (api.py ``_calculate_svf`` lexicographic, ``run_utci_tiles``
    numeric-tuple).
    """
    jobs = []
    for order, descriptor in enumerate(ordered_tiles):
        data = dict(descriptor)
        jobs.append(
            GeometryPhaseJob(
                order=order,
                tile=data["tile"],
                paths=data["paths"],
                patch_option=patch_option,
                rows=data["rows"],
                cols=data["cols"],
                patches=data["patches"],
                windchannels=windchannels,
                block_pixels=block_pixels,
                visibility_mode=data.get("visibility_mode", visibility_mode),
                dense_fallback=bool(data.get("dense_fallback", dense_fallback)),
                export_overlap=bool(data.get("export_overlap", export_overlap)),
            )
        )
    return tuple(jobs)


@dataclasses.dataclass(frozen=True)
class GeometryTileResult:
    """One committed tile: original order, native handle identity, actual mask."""

    order: int
    tile: str
    native_key: str
    recipe_digest: str
    store_root: str
    cache_hit: bool
    published_record: str
    native: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(
            order=self.order,
            tile=self.tile,
            native_key=self.native_key,
            recipe_digest=self.recipe_digest,
            store_root=self.store_root,
            cache_hit=self.cache_hit,
            published_record=self.published_record,
            native=dict(self.native),
        )


@dataclasses.dataclass(frozen=True)
class GeometryPhaseOutcome:
    """Phase result: every tile committed in original order (stage barrier met)."""

    plan: PhaseAdmissionPlan
    publication_root: str
    results: tuple[GeometryTileResult, ...]
    phase_manifest: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(
            schema=PHASE_OUTCOME_SCHEMA,
            plan=self.plan.to_dict(),
            publication_root=self.publication_root,
            results=[item.to_dict() for item in self.results],
            phase_manifest=dict(self.phase_manifest),
        )


# ---------------------------------------------------------------------------
# Child side (python -m solweig_light.runtime_phases).  Top level stays
# stdlib-only; numerical imports happen inside the job after the parent's
# env= has placed the native thread limits, mirroring runtime_worker.
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    staging = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    staging.write_text(json.dumps(_json_safe(dict(payload)), sort_keys=True), encoding="utf-8")
    os.replace(staging, path)


def _native_snapshot(tag: str) -> dict[str, Any]:
    """Actual native thread state of this child at ``tag``.

    ``numba.config.NUMBA_NUM_THREADS`` is fixed at numba import from the
    environment the scheduler placed before first import; ``get_num_threads``
    is the pool mask a parallel kernel actually uses at entry (C6-01).
    """
    import numba

    try:
        layer = numba.threading_layer()
        layer_error = None
    except Exception as error:  # not initialised before the first parallel kernel
        layer, layer_error = None, f"{type(error).__name__}: {error}"
    return {
        "tag": tag,
        "env_at_import": {name: os.environ.get(name, "") for name in _THREAD_LIMIT_VARS},
        "numba_config_num_threads": int(numba.config.NUMBA_NUM_THREADS),
        "numba_mask": int(numba.get_num_threads()),
        "numba_threading_layer": layer,
        "threading_layer_error": layer_error,
        "gdal_cachemax_env": os.environ.get("GDAL_CACHEMAX", ""),
    }


def _run_geometry_job(job_path: Path, options_data: dict) -> int:
    """Produce one tile's geometry through the shared recipe; stage the record.

    Publication is NOT done here: the worker only stages a validated record
    of the native cache generation it produced (D04: workers do not publish
    promised outputs out of order).  The scheduler commits records in the
    original tile order.
    """
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        import numba

        descriptor = GeometryPhaseJob.from_dict(job["job"])
        configured = _positive_int(job["configured_threads"], "configured_threads")
        store_root = Path(job["store_root"])
        if not job.get("cache_enabled"):
            raise ValueError(
                "geometry phase requires cache_enabled; the cache-disabled mode "
                "retains the sequential route"
            )

        # Native limits exist in env= before this import; pin and verify the
        # actual mask instead of trusting the configuration (C6-01).
        at_import = _native_snapshot("after_import")
        if at_import["numba_config_num_threads"] != configured:
            raise AssertionError(
                f"native numba config {at_import['numba_config_num_threads']} != "
                f"configured {configured} threads; env-before-import failed"
            )
        numba.set_num_threads(configured)
        if int(numba.get_num_threads()) != configured:
            raise AssertionError(
                f"numba.get_num_threads()={int(numba.get_num_threads())} != "
                f"configured {configured} after set_num_threads"
            )

        from .cache import GeometryStore
        from .cache.geometry import _manifest_digest, _read_json
        from .geometry.recipe import numerical_geometry_recipe

        recipe = numerical_geometry_recipe(descriptor.paths, descriptor.patch_option)
        store = GeometryStore(store_root)

        kernel_entry: dict[str, Any] = {}

        def producer():
            if not kernel_entry:  # first production call = kernel entry
                kernel_entry.update(_native_snapshot("kernel_entry"))
                kernel_entry["numba_mask_at_kernel_entry"] = kernel_entry["numba_mask"]
            return recipe.produce()

        handle = store.get_or_create(recipe.identity, producer)
        try:
            manifest = _read_json(store.root / handle.key / "manifest.json")
            if manifest.get("manifest_sha256") != _manifest_digest(manifest):
                raise ValueError("store manifest digest mismatch after production")
            record = {
                "schema": PHASE_STAGE_SCHEMA,
                "order": descriptor.order,
                "tile": descriptor.tile,
                "recipe_digest": recipe.digest,
                "native_key": handle.key,
                "store_root": str(store_root),
                "cache_hit": bool(handle.hit),
                "store_manifest": {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "identity": manifest["identity"],
                    "arrays": manifest["arrays"],
                    "visibility": manifest["visibility"],
                },
                "native": {
                    "configured_threads": configured,
                    "mask_pinned": int(numba.get_num_threads()),
                    "mask_at_kernel_entry": kernel_entry.get("numba_mask_at_kernel_entry"),
                    "snapshot_at_import": at_import,
                    "snapshot_at_kernel_entry": dict(kernel_entry) or None,
                },
            }
        finally:
            handle.close()
        _atomic_write_json(job_path.with_suffix(".stage.json"), record)
        return 0
    except BaseException as error:
        from .runtime_worker import write_failure

        try:
            write_failure(job_path, error)
        except BaseException:
            pass
        return 1
    finally:
        import gc

        gc.collect()  # per-tile memory retirement (m3 §6)


# --- persistent-pool protocol, kept in sync with runtime_worker.main --------

def _serve(pool_root: str, job_path: Path, options_data: dict, code: int) -> int:
    root = Path(pool_root)
    pid = os.getpid()
    (root / f"phase-worker-{pid}.pid").write_text(str(pid), encoding="utf-8")
    (root / f"phase-worker-{pid}.threads.json").write_text(
        json.dumps(
            {name: os.environ.get(name, "") for name in _THREAD_LIMIT_VARS},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    from .runtime_worker import _redirect_job_streams, _write_done

    _write_done(job_path, code)
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        job_path = Path(line.strip())
        if not job_path.name:
            continue
        _redirect_job_streams(job_path)
        _write_done(job_path, _run_geometry_job(job_path, options_data))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run solweig-light geometry phase jobs")
    parser.add_argument("--job", required=True)
    parser.add_argument("--options", required=True)
    args = parser.parse_args(argv)
    options_data = json.loads(args.options)
    job_path = Path(args.job)
    code = _run_geometry_job(job_path, options_data)
    pool_root = os.environ.get(_WORKER_POOL_ENV)
    if pool_root is None:
        return code
    return _serve(pool_root, job_path, options_data, code)


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Parent side: bounded scheduler with ordered prefix publication.
# ---------------------------------------------------------------------------

class _PhaseSlot:
    """One child process, its job channel, and its current job order."""

    __slots__ = ("process", "order", "confirmed")

    def __init__(self, process: subprocess.Popen[bytes], order: int) -> None:
        self.process = process
        self.order: int | None = order
        self.confirmed = False


def _tile_directory(root: Path, order: int) -> Path:
    return root / f"tile-{order:04d}"


def _stage_record(root: Path, order: int) -> dict[str, Any] | None:
    """The atomically staged record of tile ``order``, or None while absent."""
    path = root / f"job-{order}.stage.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("schema") != PHASE_STAGE_SCHEMA:
        return None
    return record


def _record_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in record.items() if k != "record_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _append_journal(journal: Path, entry: Mapping[str, Any]) -> None:
    line = json.dumps(_json_safe(dict(entry)), sort_keys=True) + "\n"
    with open(journal, "a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_stage_record(record: Mapping[str, Any], job: GeometryPhaseJob,
                           recipe, store) -> None:
    """Validate a staged record against the job and the live store manifest.

    Cross-process transfer is a cache key/path with validation, not a pointer
    (D09): the parent rebuilds the recipe identity from the job's own source
    paths and requires the staged native key, recipe digest, native thread
    record and store manifest to match it exactly.
    """
    from .cache.geometry import _manifest_digest, canonical_json

    if record.get("order") != job.order or record.get("tile") != job.tile:
        raise ValueError(
            f"staged record tile {record.get('tile')!r} order {record.get('order')!r} "
            f"does not match job {job.tile!r} order {job.order}"
        )
    if record.get("recipe_digest") != recipe.digest:
        raise ValueError(f"tile {job.tile}: staged recipe digest differs from the "
                         "recipe rebuilt from the job source paths")
    store_manifest = record.get("store_manifest")
    if not isinstance(store_manifest, dict):
        raise ValueError(f"tile {job.tile}: staged record carries no store manifest")
    key = store.key_for(recipe.identity)
    if record.get("native_key") != key:
        raise ValueError(f"tile {job.tile}: staged native key does not match the "
                         "recipe identity key")
    manifest = json.loads(
        (Path(record["store_root"]) / key / "manifest.json").read_text(encoding="utf-8")
    )
    if _manifest_digest(manifest) != manifest.get("manifest_sha256"):
        raise ValueError(f"tile {job.tile}: store manifest digest mismatch at "
                         "publication time")
    if _manifest_digest(manifest) != store_manifest.get("manifest_sha256"):
        raise ValueError(f"tile {job.tile}: staged store manifest digest differs "
                         "from the live store")
    for section in ("identity", "arrays", "visibility"):
        if (canonical_json(manifest.get(section))
                != canonical_json(store_manifest.get(section))):
            raise ValueError(f"tile {job.tile}: store manifest {section} differs "
                             "from the staged record")

    native = record.get("native")
    if not isinstance(native, dict):
        raise ValueError(f"tile {job.tile}: staged record carries no native record")
    configured = native.get("configured_threads")
    if native.get("mask_pinned") != configured:
        raise ValueError(
            f"tile {job.tile}: native mask {native.get('mask_pinned')!r} != "
            f"configured {configured!r}"
        )
    if not record.get("cache_hit"):
        at_entry = native.get("mask_at_kernel_entry")
        if at_entry != configured:
            raise ValueError(
                f"tile {job.tile}: native mask at kernel entry {at_entry!r} != "
                f"configured {configured!r}"
            )


def execute_geometry_phase(
    jobs: Sequence[GeometryPhaseJob | Mapping[str, Any]],
    options: RuntimeOptions | None = None,
    *,
    publication_root: str | os.PathLike[str],
    store_root: str | os.PathLike[str],
    policy: str = "queue",
    worker_module: str = GEOMETRY_PHASE_WORKER,
    python_executable: str | None = None,
    failure_grace_seconds: float = 5.0,
    gdal_cache_bytes: int | None = None,
) -> GeometryPhaseOutcome:
    """Run the geometry construction phase for ``jobs`` and return only when
    every tile is published in the original sorted order (stage barrier).

    Admission: ``runtime_memory.plan_phase_admission`` with ``policy``
    (default ``"queue"`` per m7 §4) decides the width; the scheduler never
    dispatches more than ``plan.admissible_workers`` children, and a request
    that cannot fit raises the existing ``ResourceAdmissionError`` in this
    process before any child exists.  Width comes from that plan alone.

    Publication: workers stage validated records; this scheduler commits
    them journal-first, manifest-last, strictly in ascending ``order``
    (higher-index success stays private until lower jobs reach the same
    boundary).  On failure the committed set is exactly the contiguous
    staged prefix below the numerically first recorded failure, the public
    error is the child's original rebuilt error, and every live child is
    reaped before anything propagates.  The parent environment and its
    process-global numerical settings are never changed.
    """
    from .cache import GeometryStore
    from .geometry.recipe import numerical_geometry_recipe

    options = options or get_runtime_options()
    if not options.cache_enabled:
        raise ValueError(
            "execute_geometry_phase requires cache_enabled; the cache-disabled "
            "mode retains the sequential route"
        )
    normalized = [
        item if isinstance(item, GeometryPhaseJob) else GeometryPhaseJob.from_dict(item)
        for item in jobs
    ]
    if not normalized:
        raise ValueError("execute_geometry_phase requires at least one job")
    ordered = sorted(normalized, key=lambda item: item.order)
    if [item.order for item in ordered] != list(range(len(ordered))):
        raise ValueError("job orders must be a contiguous 0-based permutation")
    if len({item.tile for item in ordered}) != len(ordered):
        raise ValueError("tile names must be unique within a phase")

    inputs = admission_inputs_from_options(options)
    plan = plan_phase_admission(
        [item.memory_job for item in ordered],
        budget_bytes=inputs["budget_bytes"],
        active_workers=inputs["active_workers"],
        threads_per_worker=inputs["threads_per_worker"],
        policy=policy,
        gdal_cache_bytes=gdal_cache_bytes,
    )
    width = plan.admissible_workers
    if gdal_cache_bytes is None:
        gdal_cache_bytes = plan.reservations[0].gdal_cache_bytes
    gdal_cache_mb = max(1, gdal_cache_bytes // MIB)

    # Input validation up front, mirroring the serial guard: each job's
    # recipe identity is rebuilt (source fingerprints streamed, bounded) so
    # missing or changed inputs fail before any child exists.
    store = GeometryStore(store_root)
    recipes = {
        job.order: numerical_geometry_recipe(job.paths, job.patch_option)
        for job in ordered
    }

    publication = Path(publication_root)
    publication.mkdir(parents=True, exist_ok=True)
    journal = publication / "publication.journal.jsonl"
    results: list[GeometryTileResult | None] = [None] * len(ordered)

    with tempfile.TemporaryDirectory(prefix="solweig-phase-") as directory:
        root = Path(directory)
        env = _child_environment(options)
        env["GDAL_CACHEMAX"] = str(gdal_cache_mb)
        env[_WORKER_POOL_ENV] = str(root)
        slots: list[_PhaseSlot] = []
        next_order = 0
        staged: dict[int, dict[str, Any]] = {}
        committed_upto = -1  # 0..committed_upto published, nothing beyond
        failure: BaseException | None = None

        def write_job(order: int) -> Path:
            path = root / f"job-{order}.json"
            payload = {
                "job": ordered[order].to_dict(),
                "configured_threads": options.threads_per_worker,
                "store_root": str(store_root),
                "cache_enabled": bool(options.cache_enabled),
            }
            path.write_text(
                json.dumps(_json_safe(payload), sort_keys=True), encoding="utf-8"
            )
            return path

        def spawn(order: int) -> None:
            """Start a fresh child on ``order`` (first assignment or fallback)."""
            path = write_job(order)
            stdout = (root / f"job-{order}.stdout").open("wb")
            stderr = (root / f"job-{order}.stderr").open("wb")
            try:
                process = subprocess.Popen(
                    [python_executable or sys.executable, "-m", worker_module,
                     "--job", os.fspath(path),
                     "--options", json.dumps(options.as_dict(), sort_keys=True)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                )
            except BaseException:
                _close_quietly(stdout)
                _close_quietly(stderr)
                raise
            _close_quietly(stdout)
            _close_quietly(stderr)
            slots.append(_PhaseSlot(process, order))

        def dispatch(slot: _PhaseSlot, order: int) -> None:
            """Hand a confirmed persistent worker its next job path."""
            path = write_job(order)
            slot.order = order
            try:
                stdin = slot.process.stdin
                if stdin is None:  # pragma: no cover - Popen(stdin=PIPE) sets it
                    raise OSError("worker job channel is missing")
                stdin.write(os.fspath(path).encode("utf-8") + b"\n")
                stdin.flush()
            except OSError:
                # The worker died between the liveness check and the write, so
                # the job never started: fail over to a fresh child on it.
                slot.order = None
                try:
                    slots.remove(slot)
                except ValueError:
                    pass
                _reap_worker(slot.process, terminate=True)
                spawn(order)

        def commit(order: int) -> GeometryTileResult:
            record = staged[order]
            job = ordered[order]
            _validate_stage_record(record, job, recipes[order], store)
            tile_dir = _tile_directory(publication, order)
            tile_dir.mkdir(parents=True, exist_ok=True)
            _append_journal(journal, {"event": "begin", "order": order,
                                      "tile": job.tile})
            published = {
                "schema": PHASE_STAGE_SCHEMA,
                "publication_policy": PHASE_STAGE_POLICY,
                "order": order,
                "tile": job.tile,
                "recipe_digest": record["recipe_digest"],
                "native_key": record["native_key"],
                "store_root": record["store_root"],
                "store_manifest": record["store_manifest"],
                "cache_hit": record["cache_hit"],
                "native": record["native"],
            }
            published["record_sha256"] = _record_digest(published)
            # Single atomic rename = the tile's commit point (manifest last).
            _atomic_write_json(tile_dir / "result.json", published)
            _append_journal(journal, {"event": "commit", "order": order,
                                      "tile": job.tile})
            return GeometryTileResult(
                order=order,
                tile=job.tile,
                native_key=record["native_key"],
                recipe_digest=record["recipe_digest"],
                store_root=str(record["store_root"]),
                cache_hit=bool(record["cache_hit"]),
                published_record=str(tile_dir / "result.json"),
                native=dict(record["native"]),
            )

        def drain(limit: int | None = None) -> None:
            """Commit the contiguous staged prefix (never past ``limit``)."""
            nonlocal committed_upto
            while committed_upto + 1 in staged:
                if limit is not None and committed_upto + 1 >= limit:
                    break
                results[committed_upto + 1] = commit(committed_upto + 1)
                committed_upto += 1

        def stop_dispatch_and_reap() -> None:
            for slot in list(slots):
                _reap_worker(slot.process, terminate=True)
            slots.clear()

        def observe_failure(order: int, code: int) -> BaseException:
            """Numerically first recorded failure among started jobs (m3 §7.3).

            Started lower-index jobs get a bounded grace to report their own
            failures so the raised error is the one the serial order would
            have produced; then everything live is reaped with terminate=True,
            the contiguous staged prefix below that failure is committed, and
            the failure itself is returned for the caller to raise.
            """
            deadline = time.monotonic() + max(0.0, float(failure_grace_seconds))
            while time.monotonic() < deadline:
                started_lower = [earlier for earlier in range(order)]
                if any(
                    _read_done(root, earlier) not in (None, 0)
                    for earlier in started_lower
                ):
                    break  # a numerically lower failure already recorded
                if all(
                    _read_done(root, earlier) is not None
                    for earlier in started_lower
                ):
                    break  # no lower job is running any more
                time.sleep(0.01)
            stop_dispatch_and_reap()
            recorded = [
                earlier for earlier in range(order)
                if _read_done(root, earlier) not in (None, 0)
            ]
            first = recorded[0] if recorded else order
            # Harvest tiles that completed below the failure while the main
            # loop was busy elsewhere: the public set must be exactly the
            # contiguous staged prefix below the first failure (serial
            # behaviour), whatever order completions were observed in.
            for earlier in range(first):
                if earlier in staged:
                    continue
                if _read_done(root, earlier) == 0:
                    record = _stage_record(root, earlier)
                    if record is not None:
                        staged[earlier] = record
            drain(limit=first)  # public set = exactly the prefix below it
            first_code = _read_done(root, first)
            return _job_failure(root, first,
                                code if first_code is None else first_code)

        try:
            while next_order < width and next_order < len(ordered):
                spawn(next_order)
                next_order += 1
            while True:
                if failure is not None:
                    break
                if next_order >= len(ordered) and all(
                    slot.order is None for slot in slots
                ):
                    break
                progressed = False
                for slot in list(slots):
                    order = slot.order
                    if order is None:
                        if slot.process.poll() is not None:
                            # An idle persistent worker that exited anyway.
                            slots.remove(slot)
                            _reap_worker(slot.process, terminate=False)
                            progressed = True
                            break
                        continue
                    marker = _read_done(root, order)
                    if marker is not None:
                        slot.confirmed = True
                        slot.order = None
                        if marker != 0:
                            failure = observe_failure(order, marker)
                            break
                        record = _stage_record(root, order)
                        if record is None:
                            failure = observe_failure(order, 1)
                            break
                        staged[order] = record
                        drain()
                        progressed = True
                        break
                    code = slot.process.poll()
                    if code is not None:
                        failure = observe_failure(order, code)
                        break
                if not progressed:
                    if failure is None:
                        time.sleep(0.01)
                    continue
                while next_order < len(ordered) and failure is None:
                    idle = next(
                        (slot for slot in slots if slot.order is None
                         and slot.confirmed and slot.process.poll() is None),
                        None,
                    )
                    if idle is not None:
                        dispatch(idle, next_order)
                    elif len(slots) < width:
                        spawn(next_order)
                    else:
                        break
                    next_order += 1
            if failure is not None:
                stop_dispatch_and_reap()
                raise failure
            # Graceful drain: persistent workers exit on end-of-input.
            for slot in slots:
                _reap_worker(slot.process, terminate=False)
            slots.clear()
        except BaseException:
            # This includes KeyboardInterrupt/SystemExit.  Reap every child
            # before TemporaryDirectory removes the JSON/job log paths.
            stop_dispatch_and_reap()
            raise

    committed = [item for item in results if item is not None]
    if len(committed) != len(ordered):
        raise TileExecutionError(
            f"geometry phase ended with {len(committed)}/{len(ordered)} tiles "
            "published; the stage barrier did not complete"
        )
    phase_manifest = {
        "schema": PHASE_OUTCOME_SCHEMA,
        "publication_policy": PHASE_STAGE_POLICY,
        "tiles": [
            {"order": item.order,
             "tile": item.tile,
             "record_sha256": _record_digest(
                 json.loads(Path(item.published_record).read_text(encoding="utf-8"))),
             "native_key": item.native_key}
            for item in committed
        ],
        "plan": plan.to_dict(),
    }
    phase_manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in phase_manifest.items()},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    # Phase-level commit point, written last (m3 §2: manifest last).
    _atomic_write_json(publication / "PHASE_MANIFEST.json", phase_manifest)
    return GeometryPhaseOutcome(
        plan=plan,
        publication_root=str(publication),
        results=tuple(committed),
        phase_manifest=phase_manifest,
    )


def _close_quietly(stream: Any) -> None:
    if stream is not None:
        try:
            stream.close()
        except BaseException:
            pass
