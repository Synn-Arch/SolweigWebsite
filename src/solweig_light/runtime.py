"""Immutable execution configuration and bounded tile scheduling.

The runtime layer owns execution policy only.  It does not alter the model's
logical tile, timestep, or sky-patch domains.  In particular, memory
estimates are admission estimates; they are not a substitute for process-tree
measurements at release time.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import builtins
import json
import numbers
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


def _read_positive_number(path: str | os.PathLike[str]) -> float | None:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _unescape_mountinfo(value: str) -> str:
    return value.replace(r"\040", " ").replace(r"\011", "\t").replace(r"\134", "\\")


def _proc_cgroup_entries(proc_root: str | os.PathLike[str] = "/proc") -> list[tuple[str, str, str]]:
    path = Path(proc_root) / "self" / "cgroup"
    entries = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return entries
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3:
            entries.append((fields[0], fields[1], fields[2] or "/"))
    return entries


def _proc_cgroup_mounts(proc_root: str | os.PathLike[str] = "/proc") -> list[dict[str, str]]:
    path = Path(proc_root) / "self" / "mountinfo"
    mounts = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return mounts
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        tail = after.split()
        if len(fields) < 6 or len(tail) < 3 or tail[0] not in {"cgroup", "cgroup2"}:
            continue
        mounts.append({
            "fstype": tail[0],
            "root": _unescape_mountinfo(fields[3]),
            "mountpoint": _unescape_mountinfo(fields[4]),
            "mount_options": fields[5],
            "super_options": tail[2],
        })
    return mounts


def _cgroup_paths(resource: str, cgroup_root: str | os.PathLike[str] | None = None,
                  proc_root: str | os.PathLike[str] = "/proc") -> list[Path]:
    """Resolve the current resource cgroup and its ancestors.

    ``cgroup_root`` is an injectable direct mount root for unit fixtures.  In
    production, `/proc/self/cgroup` and `/proc/self/mountinfo` locate nested
    cgroup mounts instead of assuming `/sys/fs/cgroup` is the process root.
    """
    paths = []

    def add_ancestors(current: Path, mount_base: Path | None) -> None:
        while True:
            if current not in paths:
                paths.append(current)
            if current.parent == current or (mount_base is not None and current == mount_base):
                break
            current = current.parent

    if cgroup_root is not None:
        add_ancestors(Path(cgroup_root), None)
        return paths

    entries = _proc_cgroup_entries(proc_root)
    mounts = _proc_cgroup_mounts(proc_root)
    for _, controllers, group_path in entries:
        for mount in mounts:
            if mount["fstype"] == "cgroup2":
                # A cgroup2 mount belongs to the unified hierarchy, whose
                # /proc entry has an empty controller field.  Do not use it
                # for a v1 cpu/memory entry in a hybrid setup.
                applicable = controllers == ""
            else:
                mount_controllers = set(mount["mount_options"].split(",")) | set(mount["super_options"].split(","))
                # cpuacct alone does not impose a CPU quota; it is an
                # accounting controller.  Match the controlling hierarchy.
                requested = {resource}
                applicable = bool(set(controllers.split(",")) & mount_controllers & requested)
            if not applicable:
                continue
            mount_root = mount["root"].rstrip("/") or "/"
            suffix = group_path
            if mount_root != "/" and (group_path == mount_root or group_path.startswith(mount_root + "/")):
                suffix = group_path[len(mount_root):] or "/"
            candidate = Path(mount["mountpoint"]) / suffix.lstrip("/")
            if candidate.exists():
                add_ancestors(candidate, Path(mount["mountpoint"]))
    if not paths:
        add_ancestors(Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup"))
    return paths


def _cgroup_cpu_limit(cgroup_root: str | os.PathLike[str] | None = None, *,
                      proc_root: str | os.PathLike[str] = "/proc") -> int | None:
    """Return the minimum CPU quota across the current cgroup's ancestors."""
    limits = []
    for root in _cgroup_paths("cpu", cgroup_root, proc_root):
        try:
            value = (root / "cpu.max").read_text(encoding="utf-8").split()
            if len(value) >= 2 and value[0] != "max":
                quota, period = float(value[0]), float(value[1])
                if quota > 0 and period > 0:
                    limits.append(max(1, int(quota // period)))
        except (OSError, ValueError, IndexError):
            pass
        quota = _read_positive_number(root / "cpu.cfs_quota_us")
        period = _read_positive_number(root / "cpu.cfs_period_us")
        if quota is None or period is None:
            quota = _read_positive_number(root / "cpu/cpu.cfs_quota_us")
            period = _read_positive_number(root / "cpu/cpu.cfs_period_us")
        if quota is not None and period is not None:
            limits.append(max(1, int(quota // period)))
    return min(limits) if limits else None


def _total_cpus() -> int:
    limits = [max(1, int(os.cpu_count() or 1))]
    try:
        limits.append(max(1, len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    quota = _cgroup_cpu_limit()
    if quota is not None:
        limits.append(quota)
    return min(limits)


def _physical_memory_bytes() -> int:
    """Return physical RAM, without making psutil a core dependency."""
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (ValueError, OSError, TypeError):
            pass
    # Windows and unusual containers may not expose sysconf.  This fallback
    # intentionally remains conservative and is only used for admission.
    if platform.system() == "Darwin":
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            )
            if int(value.strip()) > 0:
                return int(value.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return 1024 * 1024 * 1024


def _available_host_memory_bytes() -> int | None:
    """Best-effort available host RAM for conservative default admission."""
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (ValueError, OSError, TypeError):
            pass
    if platform.system() == "Darwin":
        # vm_stat is available on macOS where SC_AVPHYS_PAGES may be absent.
        try:
            output = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
            page_size_match = re.search(r"page size of (\d+) bytes", output)
            if page_size_match:
                page_size = int(page_size_match.group(1))
                available = 0
                for label in ("Pages free", "Pages inactive", "Pages speculative"):
                    match = re.search(rf"{re.escape(label)}:\s+(\d+)\.", output)
                    if match:
                        available += int(match.group(1))
                if available > 0:
                    return available * page_size
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def _container_memory_limit_bytes(cgroup_root: str | os.PathLike[str] | None = None, *,
                                  proc_root: str | os.PathLike[str] = "/proc") -> int | None:
    """Return the minimum finite memory limit across cgroup ancestors."""
    limits = []
    for root in _cgroup_paths("memory", cgroup_root, proc_root):
        for path in (root / "memory.max", root / "memory.limit_in_bytes", root / "memory/memory.limit_in_bytes"):
            try:
                value = Path(path).read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if value == "max":
                continue
            try:
                limit = int(value)
            except ValueError:
                continue
            # v1 uses a very large sentinel when no limit is configured.
            if limit > 0 and limit < 1 << 60:
                limits.append(limit)
    return min(limits) if limits else None


def _container_memory_headroom_bytes(cgroup_root: str | os.PathLike[str] | None = None, *,
                                     proc_root: str | os.PathLike[str] = "/proc") -> int | None:
    """Return minimum finite cgroup memory headroom, including current use."""
    headrooms = []
    for root in _cgroup_paths("memory", cgroup_root, proc_root):
        for limit_path, usage_path in (
            (root / "memory.max", root / "memory.current"),
            (root / "memory.limit_in_bytes", root / "memory.usage_in_bytes"),
            (root / "memory/memory.limit_in_bytes", root / "memory/memory.usage_in_bytes"),
        ):
            try:
                raw_limit = limit_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if raw_limit == "max":
                continue
            try:
                limit = int(raw_limit)
            except ValueError:
                continue
            if limit < 0 or limit >= 1 << 60:
                continue
            try:
                usage = int(usage_path.read_text(encoding="utf-8").strip())
            except (OSError, UnicodeError, ValueError):
                # A finite limit remains useful when a platform omits the
                # usage counter; measured current use is preferred whenever it
                # is available.
                usage = 0
            headrooms.append(max(0, limit - usage))
            break
    return min(headrooms) if headrooms else None


DEFAULT_MEMORY_FRACTION = 0.50
DEFAULT_PATCHES = 153
DEFAULT_WIND_CHANNELS = 12
# Conservative inventory used for admission.  These are accounting units, not
# a measured RSS bound: each full-plane equivalent is rows*cols float32
# values.  The breakdown and rationale live in docs/p6_memory_inventory.md.
LIVE_FULL_PLANE_EQUIVALENTS = 192
DTYPE64_RESERVED_PLANES = 32


def default_memory_budget_bytes() -> int:
    """Use half of the most restrictive available/container memory view."""
    limits = [_physical_memory_bytes()]
    available = _available_host_memory_bytes()
    if available is not None:
        limits.append(available)
    container_headroom = _container_memory_headroom_bytes()
    if container_headroom is not None:
        if container_headroom <= 0:
            return 0
        limits.append(container_headroom)
    return max(1, int(min(limits) * DEFAULT_MEMORY_FRACTION))


def resolved_memory_budget_bytes(options: RuntimeOptions | None = None) -> int:
    """Resolve an explicit budget or the conservative physical-RAM default."""
    return (options or get_runtime_options()).resolved_memory_budget_bytes


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Frozen execution controls shared by one public workflow invocation."""

    cache_dir: str | None = None
    legacy_cache_policy: str = "recompute"
    cache_enabled: bool = True
    memory_budget_bytes: int | None = None
    cpu_budget: int = 1
    workers: int = 1
    threads_per_worker: int = 1
    block_pixels: int = 128
    checkpoint_interval: int = 1
    resume: bool = False

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, os.PathLike)):
                raise TypeError("cache_dir must be a string path or None")
            object.__setattr__(self, "cache_dir", os.fspath(self.cache_dir))
        if self.legacy_cache_policy not in {"recompute", "trust"}:
            raise ValueError("legacy_cache_policy must be 'recompute' or 'trust'")
        if not isinstance(self.cache_enabled, bool):
            raise TypeError("cache_enabled must be a bool")
        if self.memory_budget_bytes is not None:
            object.__setattr__(
                self,
                "memory_budget_bytes",
                _positive_int(self.memory_budget_bytes, "memory_budget_bytes"),
            )
        for name in (
            "cpu_budget",
            "workers",
            "threads_per_worker",
            "block_pixels",
            "checkpoint_interval",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a bool")
        cpus = _total_cpus()
        if self.threads_per_worker > cpus:
            raise ValueError(
                f"threads_per_worker={self.threads_per_worker} exceeds {cpus} available CPUs"
            )
        if self.cpu_budget > cpus:
            raise ValueError(
                f"cpu_budget={self.cpu_budget} exceeds {cpus} available CPUs"
            )
        if self.threads_per_worker > self.cpu_budget:
            raise ValueError(
                "threads_per_worker cannot exceed cpu_budget; increase cpu_budget "
                "or reduce threads_per_worker"
            )

    @property
    def resolved_memory_budget_bytes(self) -> int:
        return (
            self.memory_budget_bytes
            if self.memory_budget_bytes is not None
            else default_memory_budget_bytes()
        )

    @property
    def requested_native_threads(self) -> int:
        """Native threads requested if all configured workers run at once."""
        return self.workers * self.threads_per_worker

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_DEFAULT_OPTIONS = RuntimeOptions()
_OPTIONS: contextvars.ContextVar[RuntimeOptions] = contextvars.ContextVar(
    "solweig_runtime_options", default=_DEFAULT_OPTIONS
)


def get_runtime_options() -> RuntimeOptions:
    """Return the immutable options snapshot for the current context."""
    return _OPTIONS.get()


@contextlib.contextmanager
def runtime_options(
    options: RuntimeOptions | Mapping[str, Any] | None = None, **overrides: Any
) -> Iterator[RuntimeOptions]:
    """Temporarily bind runtime options to the current context.

    Nested scopes copy the current snapshot.  ContextVar reset in ``finally``
    gives concurrent threads and asyncio tasks independent configuration.
    """
    if options is None:
        base = get_runtime_options()
    elif isinstance(options, RuntimeOptions):
        base = options
    elif isinstance(options, Mapping):
        base = RuntimeOptions(**dict(options))
    else:
        raise TypeError("options must be RuntimeOptions, a mapping, or None")
    unknown = set(overrides) - {field.name for field in dataclasses.fields(RuntimeOptions)}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown runtime option(s): {names}")
    selected = dataclasses.replace(base, **overrides) if overrides else base
    token = _OPTIONS.set(selected)
    try:
        yield selected
    finally:
        _OPTIONS.reset(token)


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Auditable components of one tile's conservative live-memory estimate."""

    total_bytes: int
    pixels: int
    patches: int
    raw_visibility_bytes: int
    live_array_bytes: int
    decoded_block_bytes: int
    native_overhead_bytes: int
    full_plane_equivalents: int = LIVE_FULL_PLANE_EQUIVALENTS
    dtype64_reserved_planes: int = DTYPE64_RESERVED_PLANES
    windchannels: int = DEFAULT_WIND_CHANNELS

    @property
    def inventory(self) -> dict[str, int]:
        return {
            "pixels": self.pixels,
            "patches": self.patches,
            "full_plane_equivalents": self.full_plane_equivalents,
            "dtype64_reserved_planes": self.dtype64_reserved_planes,
            "windchannels": self.windchannels,
            "raw_visibility_bytes": self.raw_visibility_bytes,
            "live_array_bytes": self.live_array_bytes,
            "decoded_block_bytes": self.decoded_block_bytes,
            "native_overhead_bytes": self.native_overhead_bytes,
            "total_bytes": self.total_bytes,
        }


def estimate_memory(
    rows: int,
    cols: int,
    patches: int = DEFAULT_PATCHES,
    windchannels: int = DEFAULT_WIND_CHANNELS,
    block_pixels: int = 128,
) -> MemoryEstimate:
    """Estimate worst-case live memory for a tile before launching it.

    The estimate retains the three raw float32 visibility channels, full-raster
    live/state arrays, a decoded patch block, wind coefficient channels, and a
    fixed allowance for JIT/native allocations.  It is deliberately
    conservative and makes no claim to be a hard RSS bound.
    """
    rows = _positive_int(rows, "rows")
    cols = _positive_int(cols, "cols")
    patches = _positive_int(patches, "patches")
    windchannels = _positive_int(windchannels, "windchannels")
    block_pixels = _positive_int(block_pixels, "block_pixels")
    pixels = rows * cols
    float32 = 4
    raw_visibility = 3 * pixels * patches * float32
    # Scene/geometry, chronological state/forcing, engine outputs, and
    # radiation/ground-view scratch are accounted as 192 full planes.  Reserve
    # an additional 32 planes for arrays promoted to float64 by dependencies.
    live_arrays = (
        (LIVE_FULL_PLANE_EQUIVALENTS + DTYPE64_RESERVED_PLANES + windchannels)
        * pixels
        * float32
    )
    decoded = block_pixels * patches * (windchannels + 4) * float32
    # JIT code and native-library workspaces are not represented by NumPy nbytes.
    native = 256 * 1024 * 1024 + 64 * 1024 * 1024 * min(8, windchannels)
    total = raw_visibility + live_arrays + decoded + native
    return MemoryEstimate(
        total,
        pixels,
        patches,
        raw_visibility,
        live_arrays,
        decoded,
        native,
        LIVE_FULL_PLANE_EQUIVALENTS,
        DTYPE64_RESERVED_PLANES,
        windchannels,
    )


def estimate_tile_memory(
    rows: int,
    cols: int,
    patches: int = DEFAULT_PATCHES,
    windchannels: int = DEFAULT_WIND_CHANNELS,
    block_pixels: int = 128,
) -> int:
    """Return only the byte count for callers that need a scalar estimate."""
    return estimate_memory(rows, cols, patches, windchannels, block_pixels).total_bytes


# Descriptive aliases for callers that used the terminology in the P6 packet.
estimate_live_memory = estimate_tile_memory
conservative_memory_estimate = estimate_memory


class ResourceAdmissionError(MemoryError):
    """Raised before work starts when configured resource budgets cannot fit."""


@dataclass(frozen=True, slots=True)
class AdmissionPlan:
    active_workers: int
    estimates: tuple[int, ...]
    native_threads: int


def _job_estimate(job: Mapping[str, Any], options: RuntimeOptions, index: int) -> int:
    if "memory_estimate_bytes" in job:
        return _positive_int(job["memory_estimate_bytes"], "memory_estimate_bytes")
    rows, cols = job.get("rows"), job.get("cols")
    shape = job.get("shape")
    if (rows is None or cols is None) and isinstance(shape, (tuple, list)) and len(shape) >= 2:
        rows, cols = shape[:2]
    if rows is None or cols is None:
        dsm = job.get("paths", {}).get("Building_DSM") if isinstance(job.get("paths"), Mapping) else None
        if not dsm:
            raise ResourceAdmissionError(
                f"tile job {index} has no dimensions; provide rows/cols or shape, "
                "or a readable paths['Building_DSM'] before resource admission"
            )
        try:
            from osgeo import gdal

            dataset = gdal.Open(os.fspath(dsm), gdal.GA_ReadOnly)
            if dataset is None:
                raise RuntimeError("GDAL could not open the Building_DSM")
            rows, cols = dataset.RasterYSize, dataset.RasterXSize
            dataset = None
        except Exception as error:
            raise ResourceAdmissionError(
                f"tile job {index} dimensions could not be read from Building_DSM "
                f"{dsm!r}: {error}; provide verified rows/cols or shape"
            ) from error
    if rows is None or cols is None:
        raise ResourceAdmissionError(f"tile job {index} must provide both rows and cols")
    patches = int(job.get("patches", DEFAULT_PATCHES))
    windchannels = int(job.get("windchannels", DEFAULT_WIND_CHANNELS))
    return estimate_tile_memory(rows, cols, patches, windchannels, options.block_pixels)


def plan_admission(jobs: Sequence[Mapping[str, Any]], options: RuntimeOptions | None = None) -> AdmissionPlan:
    """Compute deterministic worker admission under memory and CPU budgets."""
    options = options or get_runtime_options()
    estimates = tuple(_job_estimate(job, options, index) for index, job in enumerate(jobs))
    budget = options.resolved_memory_budget_bytes
    if any(value > budget for value in estimates):
        index, value = next((i, v) for i, v in enumerate(estimates) if v > budget)
        raise ResourceAdmissionError(
            f"tile job {index} needs about {value:,} bytes but memory_budget_bytes "
            f"is {budget:,}; increase the memory budget or use supported disk-backed "
            "execution with an explicit live-array inventory"
        )
    cpu_workers = options.cpu_budget // options.threads_per_worker
    if not estimates:
        return AdmissionPlan(0, (), 0)
    requested = min(options.workers, max(1, cpu_workers), len(estimates))
    # Use the largest simultaneous jobs as the safe aggregate bound.  This is
    # deterministic even when tiles have different dimensions.
    active = 0
    for count in range(1, requested + 1):
        if sum(sorted(estimates, reverse=True)[:count]) <= budget:
            active = count
        else:
            break
    if estimates and active == 0:  # defensive; the per-job check above catches it
        raise ResourceAdmissionError("no tile fits the configured resource budgets")
    return AdmissionPlan(active, estimates, active * options.threads_per_worker)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"job contains non-JSON-safe value {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TileResult:
    index: int
    tile: str | None
    returncode: int


class TileExecutionError(RuntimeError):
    """A child failed; all siblings have been stopped before this is raised."""


_CHILD_BUILTIN_EXCEPTIONS = {
    name: getattr(builtins, name)
    for name in (
        "ArithmeticError", "AssertionError", "AttributeError", "EOFError",
        "FileExistsError", "FileNotFoundError", "ImportError", "IndexError",
        "IsADirectoryError", "KeyError", "LookupError", "MemoryError",
        "ModuleNotFoundError", "NameError", "NotADirectoryError", "OSError",
        "OverflowError", "PermissionError", "RuntimeError", "TimeoutError",
        "TypeError", "UnicodeError", "ValueError", "ZeroDivisionError",
    )
}


def _child_exception(payload: Any) -> BaseException | None:
    """Reconstruct only builtins and explicitly owned SOLWEIG exceptions."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        return None
    name, module = payload.get("exception_type"), payload.get("exception_module")
    cls = None
    if module == "builtins":
        cls = _CHILD_BUILTIN_EXCEPTIONS.get(name)
    elif (module, name) == ("solweig_light.identities", "InputChangedError"):
        from .identities import InputChangedError

        cls = InputChangedError
    elif (module, name) == ("solweig_light.persistence", "PersistenceError"):
        from .persistence import PersistenceError

        cls = PersistenceError
    elif (module, name) == ("solweig_light.runtime", "ResourceAdmissionError"):
        cls = ResourceAdmissionError
    if cls is None:
        return None
    args = payload.get("args")
    if not isinstance(args, list):
        args = [payload.get("message", "tile worker failed")]
    try:
        return cls(*args)
    except Exception:
        try:
            return cls(payload.get("message", "tile worker failed"))
        except Exception:
            return None


def _child_environment(options: RuntimeOptions) -> dict[str, str]:
    env = os.environ.copy()
    value = str(options.threads_per_worker)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = value
    # C6-42: cap the per-process GDAL block cache at the allowance the phase
    # admission calculator charges (5% of physical RAM when GDAL_CACHEMAX is
    # unset; M2 §5).  GDAL reads this at first use in each fresh child, the
    # same mechanism as the thread caps above.  Value is megabytes.
    from .runtime_memory import default_gdal_cache_bytes
    env['GDAL_CACHEMAX'] = str(max(1, default_gdal_cache_bytes() // (1024 * 1024)))
    return env


# Persistent-pool handshake.  The scheduler places its private pool directory
# in the child environment before the child's first import; a worker that
# understands the protocol reports each finished job with an atomic done
# marker and serves further job paths from stdin until end-of-input.  Worker
# modules without protocol support (the one-shot entry and test stubs) exit
# after their first job, and the scheduler respawns them per job, preserving
# the previous one-shot behavior.  Keep in sync with runtime_worker.main.
_WORKER_POOL_ENV = "SOLWEIG_LIGHT_WORKER_POOL"


def _text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return ""


def _read_done(root: Path, index: int) -> int | None:
    """Return a job's recorded return code, or ``None`` while still running.

    Workers publish done markers by atomic rename, so an existing marker is
    final.  A marker that exists but cannot be parsed cannot certify success
    and is reported as a failure instead of hanging the scheduler.
    """
    marker = root / f"job-{index}.done"
    if not marker.is_file():
        return None
    try:
        return int(json.loads(marker.read_text(encoding="utf-8"))["returncode"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 1


def _job_failure(root: Path, index: int, code: int) -> BaseException:
    """Rebuild the same observable failure the one-shot scheduler raised."""
    failure_payload = None
    failure_path = root / f"job-{index}.failure.json"
    if failure_path.is_file():
        try:
            failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            failure_payload = None
    error_text = _text_or_empty(root / f"job-{index}.stderr")[-4000:]
    child_error = _child_exception(failure_payload)
    if child_error is not None:
        note = f"tile job {index} failed with exit code {code}"
        if error_text.strip():
            note += f"; worker stderr: {error_text.strip()}"
        child_error.add_note(note)
        return child_error
    return TileExecutionError(
        f"tile job {index} failed with exit code {code}: {error_text.strip()}"
    )


def _close_quietly(stream: Any) -> None:
    if stream is not None:
        try:
            stream.close()
        except BaseException:
            # Preserve the scheduling or cancellation exception.
            pass


def _reap_worker(process: subprocess.Popen[bytes], *, terminate: bool) -> None:
    """Close the job channel and reap the child; never raises.

    Persistent workers exit on end-of-input; ``terminate=True`` is the
    cancellation path and stops the child immediately.  A child that survives
    termination is killed, so cancellation can never hang the batch.
    """
    _close_quietly(getattr(process, "stdin", None))
    if not terminate and process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        except BaseException:
            pass
    try:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait()
        except BaseException:
            pass
    except BaseException:
        pass


class _WorkerSlot:
    """One child process, its job channel, and its current job index."""

    __slots__ = ("process", "job_index", "confirmed")

    def __init__(self, process: subprocess.Popen[bytes], job_index: int) -> None:
        self.process = process
        self.job_index: int | None = job_index
        # Set once this slot has reported a done marker, proving it speaks the
        # persistent protocol and may receive further jobs on stdin.
        self.confirmed = False


def execute_tiles(
    jobs: Sequence[Mapping[str, Any]],
    options: RuntimeOptions | None = None,
    *,
    python_executable: str | None = None,
    worker_module: str = "solweig_light.runtime_worker",
) -> tuple[TileResult, ...]:
    """Run independent tile jobs on a bounded pool of persistent subprocesses.

    Exactly ``plan_admission(...).active_workers`` children are created per
    call, each with native thread limits in ``env=`` before its first import.
    A child that supports the persistent protocol serves tile jobs
    sequentially from the scheduler until the batch completes or is
    cancelled, releasing tile state between jobs; one-shot worker modules
    exit after their single job and are respawned, preserving the previous
    per-job behavior.  Per-tile artifacts, failure payloads, and checkpoint
    publication inside ``run_tile`` are unchanged: a failing job raises the
    child's rebuilt exception (or ``TileExecutionError`` for crashes), and
    every remaining job is stopped before it propagates.  The parent
    environment and process-global numerical settings are never changed.
    """
    options = options or get_runtime_options()
    normalized = [_json_safe(dict(job)) for job in jobs]
    if not normalized:
        return ()
    plan = plan_admission(normalized, options)
    executable = python_executable or sys.executable
    results: list[TileResult | None] = [None] * len(normalized)

    with tempfile.TemporaryDirectory(prefix="solweig-runtime-") as directory:
        root = Path(directory)
        env = _child_environment(options)
        env[_WORKER_POOL_ENV] = str(root)
        slots: list[_WorkerSlot] = []
        next_index = 0

        def write_job(index: int) -> Path:
            path = root / f"job-{index}.json"
            path.write_text(json.dumps(normalized[index], sort_keys=True), encoding="utf-8")
            return path

        def spawn(index: int) -> None:
            """Start a fresh child on ``index`` (first assignment or fallback)."""
            path = write_job(index)
            stdout = (root / f"job-{index}.stdout").open("wb")
            stderr = (root / f"job-{index}.stderr").open("wb")
            try:
                process = subprocess.Popen(
                    [executable, "-m", worker_module, "--job", os.fspath(path),
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
            # The child owns duplicated descriptors, and a persistent worker
            # reopens its own per-job streams, so the parent copies close now.
            _close_quietly(stdout)
            _close_quietly(stderr)
            slots.append(_WorkerSlot(process, index))

        def dispatch(slot: _WorkerSlot, index: int) -> None:
            """Hand a confirmed persistent worker its next job path."""
            path = write_job(index)
            slot.job_index = index
            try:
                stdin = slot.process.stdin
                if stdin is None:  # pragma: no cover - Popen(stdin=PIPE) sets it
                    raise OSError("worker job channel is missing")
                stdin.write(os.fspath(path).encode("utf-8") + b"\n")
                stdin.flush()
            except OSError:
                # The worker died between the liveness check and the write, so
                # the job never started: fail over to a fresh child on it.
                slot.job_index = None
                try:
                    slots.remove(slot)
                except ValueError:
                    pass
                _reap_worker(slot.process, terminate=True)
                spawn(index)

        def complete(index: int, code: int) -> None:
            if code != 0:
                raise _job_failure(root, index, code)
            results[index] = TileResult(index, normalized[index].get("tile"), code)

        def reap(slot: _WorkerSlot, code: int) -> None:
            """The slot's process exited on its own while holding a job."""
            index = slot.job_index
            try:
                slots.remove(slot)
            except ValueError:
                pass
            slot.job_index = None
            marker_code = _read_done(root, index) if index is not None else None
            if index is None:  # pragma: no cover - only reachable via dispatch
                return
            if marker_code is not None:
                # A persistent worker that exits after reporting its result.
                complete(index, marker_code)
            elif code == 0 and not slot.confirmed:
                # One-shot worker modules exit after their single job.
                complete(index, 0)
            elif code == 0:
                raise TileExecutionError(
                    f"tile worker exited before tile job {index} could start"
                )
            else:
                raise _job_failure(root, index, code)

        try:
            while next_index < len(normalized) and len(slots) < plan.active_workers:
                spawn(next_index)
                next_index += 1
            while True:
                if next_index >= len(normalized) and all(
                    slot.job_index is None for slot in slots
                ):
                    break
                progressed = False
                for slot in list(slots):
                    index = slot.job_index
                    if index is None:
                        if slot.process.poll() is not None:
                            # An idle persistent worker that exited anyway.
                            try:
                                slots.remove(slot)
                            except ValueError:
                                pass
                            _reap_worker(slot.process, terminate=False)
                            progressed = True
                            break
                        continue
                    marker_code = _read_done(root, index)
                    if marker_code is not None:
                        slot.confirmed = True
                        slot.job_index = None
                        complete(index, marker_code)
                        progressed = True
                        break
                    code = slot.process.poll()
                    if code is not None:
                        reap(slot, code)
                        progressed = True
                        break
                if not progressed:
                    time.sleep(0.01)
                    continue
                while next_index < len(normalized):
                    idle = next(
                        (slot for slot in slots if slot.job_index is None
                         and slot.confirmed and slot.process.poll() is None),
                        None,
                    )
                    if idle is not None:
                        dispatch(idle, next_index)
                    elif len(slots) < plan.active_workers:
                        spawn(next_index)
                    else:
                        break
                    next_index += 1
            # Graceful drain: persistent workers exit on end-of-input.
            for slot in slots:
                _reap_worker(slot.process, terminate=False)
            slots.clear()
        except BaseException:
            # This includes KeyboardInterrupt/SystemExit.  Reap every child
            # before TemporaryDirectory removes the JSON/job log paths.
            for slot in list(slots):
                _reap_worker(slot.process, terminate=True)
            slots.clear()
            raise
    return tuple(result for result in results if result is not None)


__all__ = [
    "AdmissionPlan",
    "MemoryEstimate",
    "ResourceAdmissionError",
    "RuntimeOptions",
    "TileExecutionError",
    "TileResult",
    "conservative_memory_estimate",
    "default_memory_budget_bytes",
    "estimate_live_memory",
    "estimate_memory",
    "estimate_tile_memory",
    "execute_tiles",
    "get_runtime_options",
    "plan_admission",
    "runtime_options",
    "resolved_memory_budget_bytes",
]
