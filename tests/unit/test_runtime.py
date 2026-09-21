"""Runtime policy and scheduler tests; no numerical pipeline is exercised here."""

from __future__ import annotations

import json
import os
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from solweig_light.runtime import (
    ResourceAdmissionError,
    RuntimeOptions,
    TileExecutionError,
    estimate_memory,
    estimate_tile_memory,
    execute_tiles,
    get_runtime_options,
    plan_admission,
    runtime_options,
)
from solweig_light import runtime as runtime_module


def test_options_are_frozen_and_scoped():
    before = get_runtime_options()
    with pytest.raises(Exception):
        before.workers = 2  # type: ignore[misc]
    with runtime_options(workers=2, block_pixels=9) as selected:
        assert selected.workers == 2
        assert get_runtime_options().block_pixels == 9
    assert get_runtime_options() is before


def test_contexts_are_isolated_between_threads():
    def read(value):
        with runtime_options(block_pixels=value):
            return get_runtime_options().block_pixels

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(read, (17, 31))) == [17, 31]
    assert get_runtime_options().block_pixels == 128


@pytest.mark.parametrize("field", ["workers", "threads_per_worker", "block_pixels", "checkpoint_interval"])
def test_positive_integer_validation(field):
    with pytest.raises(ValueError):
        RuntimeOptions(**{field: 0})


def test_cache_policy_and_cpu_validation():
    with pytest.raises(ValueError, match="legacy_cache_policy"):
        RuntimeOptions(legacy_cache_policy="use-it")
    with pytest.raises(ValueError, match="threads_per_worker"):
        RuntimeOptions(cpu_budget=1, threads_per_worker=2)


def test_cgroup_resource_readers_are_fixture_injectable(tmp_path):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.max").write_text("150000 100000\n")
    (cgroup / "memory.max").write_text(str(4 * 1024**3) + "\n")
    assert runtime_module._cgroup_cpu_limit(cgroup) == 1
    assert runtime_module._container_memory_limit_bytes(cgroup) == 4 * 1024**3


def test_cgroup_discovery_uses_nested_path_and_ancestor_minimum(tmp_path):
    mount = tmp_path / "mounted-cgroup"
    nested = mount / "leaf"
    nested.mkdir(parents=True)
    (mount / "cpu.max").write_text("400000 100000\n")
    (nested / "cpu.max").write_text("max 100000\n")
    (mount / "memory.max").write_text(str(8 * 1024**3) + "\n")
    (nested / "memory.max").write_text("max\n")
    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/tenant/leaf\n")
    (proc / "mountinfo").write_text(
        f"36 25 0:32 /tenant {mount} rw,relatime - cgroup2 cgroup rw\n"
    )
    assert runtime_module._cgroup_cpu_limit(proc_root=tmp_path / "proc") == 4
    assert runtime_module._container_memory_limit_bytes(proc_root=tmp_path / "proc") == 8 * 1024**3

    # A finite nested limit must win over a more generous ancestor.
    (nested / "cpu.max").write_text("200000 100000\n")
    (nested / "memory.max").write_text(str(2 * 1024**3) + "\n")
    assert runtime_module._cgroup_cpu_limit(proc_root=tmp_path / "proc") == 2
    assert runtime_module._container_memory_limit_bytes(proc_root=tmp_path / "proc") == 2 * 1024**3


def test_hybrid_mounts_match_only_the_controlling_v1_hierarchy(tmp_path):
    unified = tmp_path / "unified"
    cpu_mount = tmp_path / "cpu"
    memory_mount = tmp_path / "memory"
    for path in (unified / "leaf", cpu_mount / "leaf", memory_mount / "leaf"):
        path.mkdir(parents=True)
    (unified / "leaf" / "cpu.max").write_text("800000 100000\n")
    (cpu_mount / "leaf" / "cpu.cfs_quota_us").write_text("200000\n")
    (cpu_mount / "leaf" / "cpu.cfs_period_us").write_text("100000\n")
    (memory_mount / "leaf" / "memory.limit_in_bytes").write_text(str(4 * 1024**3) + "\n")
    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text(
        "0::/unified/leaf\n5:cpu,cpuacct:/leaf\n6:memory:/leaf\n"
    )
    (proc / "mountinfo").write_text(
        f"36 25 0:32 / {unified} rw - cgroup2 cgroup rw\n"
        f"37 25 0:33 / {cpu_mount} rw,cpu,cpuacct - cgroup cgroup rw,cpu,cpuacct\n"
        f"38 25 0:34 / {memory_mount} rw,memory - cgroup cgroup rw,memory\n"
    )
    assert runtime_module._cgroup_cpu_limit(proc_root=tmp_path / "proc") == 2
    assert runtime_module._container_memory_limit_bytes(proc_root=tmp_path / "proc") == 4 * 1024**3

    # A standalone cpuacct hierarchy is accounting-only and cannot establish
    # a CPU quota for admission.
    acct = tmp_path / "acct"
    (acct / "leaf").mkdir(parents=True)
    (acct / "leaf" / "cpu.cfs_quota_us").write_text("100000\n")
    (acct / "leaf" / "cpu.cfs_period_us").write_text("100000\n")
    (proc / "cgroup").write_text("5:cpuacct:/leaf\n")
    (proc / "mountinfo").write_text(
        f"37 25 0:33 / {acct} rw,cpuacct - cgroup cgroup rw,cpuacct\n"
    )
    assert runtime_module._cgroup_cpu_limit(proc_root=tmp_path / "proc") is None


def test_memory_headroom_uses_current_and_exhausted_ancestor(tmp_path):
    mount = tmp_path / "cgroup"
    nested = mount / "leaf"
    nested.mkdir(parents=True)
    (mount / "memory.max").write_text(str(8 * 1024**3) + "\n")
    (mount / "memory.current").write_text(str(6 * 1024**3) + "\n")
    (nested / "memory.max").write_text(str(4 * 1024**3) + "\n")
    (nested / "memory.current").write_text(str(3 * 1024**3) + "\n")
    assert runtime_module._container_memory_headroom_bytes(nested) == 1 * 1024**3
    (mount / "memory.current").write_text(str(8 * 1024**3) + "\n")
    assert runtime_module._container_memory_headroom_bytes(nested) == 0

    v1_mount = tmp_path / "v1-memory"
    v1_nested = v1_mount / "leaf"
    v1_nested.mkdir(parents=True)
    (v1_mount / "memory.limit_in_bytes").write_text(str(6 * 1024**3) + "\n")
    (v1_mount / "memory.usage_in_bytes").write_text(str(5 * 1024**3) + "\n")
    (v1_nested / "memory.limit_in_bytes").write_text(str(4 * 1024**3) + "\n")
    (v1_nested / "memory.usage_in_bytes").write_text(str(1 * 1024**3) + "\n")
    assert runtime_module._container_memory_headroom_bytes(v1_nested) == 1 * 1024**3


@pytest.mark.parametrize('limit_name,usage_name', [
    ('memory.max', 'memory.current'),
    ('memory.limit_in_bytes', 'memory.usage_in_bytes'),
])
def test_zero_memory_limit_is_exhausted(tmp_path, limit_name, usage_name):
    (tmp_path / limit_name).write_text('0\n')
    (tmp_path / usage_name).write_text('0\n')
    assert runtime_module._container_memory_headroom_bytes(tmp_path) == 0
    (tmp_path / usage_name).write_text('123\n')
    assert runtime_module._container_memory_headroom_bytes(tmp_path) == 0


def test_estimate_components_are_monotonic():
    small = estimate_memory(10, 10, patches=10, windchannels=2, block_pixels=8)
    large = estimate_memory(20, 10, patches=20, windchannels=4, block_pixels=16)
    assert large.total_bytes > small.total_bytes
    assert large.raw_visibility_bytes > small.raw_visibility_bytes
    assert estimate_tile_memory(10, 10) == estimate_memory(10, 10).total_bytes


def test_memory_inventory_accounts_full_planes_and_dtype64_reserve():
    estimate = estimate_memory(10, 20, patches=153, windchannels=12, block_pixels=128)
    assert estimate.full_plane_equivalents == 192
    assert estimate.dtype64_reserved_planes == 32
    expected_live = 10 * 20 * (192 + 32 + 12) * 4
    assert estimate.live_array_bytes == expected_live
    assert estimate.inventory["total_bytes"] == estimate.total_bytes


def test_admission_reduces_workers_and_rejects_oversized_jobs():
    jobs = [{"tile": str(i), "memory_estimate_bytes": 60} for i in range(4)]
    options = RuntimeOptions(memory_budget_bytes=130, cpu_budget=4, workers=4)
    plan = plan_admission(jobs, options)
    assert plan.active_workers == 2
    with pytest.raises(ResourceAdmissionError, match="increase the memory budget"):
        plan_admission([{"memory_estimate_bytes": 131}], options)
    with pytest.raises(ResourceAdmissionError, match="no dimensions"):
        plan_admission([{"tile": "missing-shape"}], options)


def _write_stub(path, marker):
    path.write_text(
        textwrap.dedent(
            f"""
            import argparse, json, os
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument('--job'); p.add_argument('--options')
            a = p.parse_args()
            job = json.loads(Path(a.job).read_text())
            Path({str(marker)!r}).open('a').write(job['tile'] + ':' + os.environ['OMP_NUM_THREADS'] + '\\n')
            if job.get('fail'): raise SystemExit(7)
            """
        ),
        encoding="utf-8",
    )


def test_subprocess_scheduler_sets_child_threads_and_cleans_failures(tmp_path, monkeypatch):
    marker = tmp_path / "marker.txt"
    module = tmp_path / "runtime_stub.py"
    _write_stub(module, marker)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", ""))))
    options = RuntimeOptions(memory_budget_bytes=1_000_000, cpu_budget=2, workers=2, threads_per_worker=1)
    jobs = [{"tile": "a", "memory_estimate_bytes": 10}, {"tile": "b", "memory_estimate_bytes": 10}]
    result = execute_tiles(jobs, options, worker_module="runtime_stub")
    assert [item.tile for item in result] == ["a", "b"]
    assert sorted(marker.read_text().splitlines()) == ["a:1", "b:1"]
    with pytest.raises(TileExecutionError, match="job 1"):
        execute_tiles(jobs[:1] + [{"tile": "bad", "fail": True, "memory_estimate_bytes": 10}], options,
                      worker_module="runtime_stub")


def test_keyboard_interrupt_reaps_live_children(tmp_path, monkeypatch):
    pid_file = tmp_path / "child.pid"
    module = tmp_path / "runtime_live_stub.py"
    module.write_text(
        textwrap.dedent(
            f"""
            import argparse, json, os, time
            from pathlib import Path
            p = argparse.ArgumentParser(); p.add_argument('--job'); p.add_argument('--options')
            a = p.parse_args()
            job = json.loads(Path(a.job).read_text())
            Path({str(pid_file)!r}).write_text(str(os.getpid()))
            while job.get('hold'):
                time.sleep(.05)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", ""))))
    options = RuntimeOptions(memory_budget_bytes=1_000_000, cpu_budget=1, workers=1)
    interrupted = True
    real_sleep = time.sleep

    def interrupt(_seconds):
        nonlocal interrupted
        if interrupted:
            deadline = time.monotonic() + 2
            while not pid_file.exists() and time.monotonic() < deadline:
                real_sleep(.01)
            interrupted = False
            raise KeyboardInterrupt
        real_sleep(_seconds)

    monkeypatch.setattr("solweig_light.runtime.time.sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute_tiles([{"tile": "live", "hold": True, "memory_estimate_bytes": 10}], options,
                      worker_module="runtime_live_stub")
    deadline = time.monotonic() + 2
    while not pid_file.exists() and time.monotonic() < deadline:
        real_sleep(.01)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_subprocess_known_failures_restore_types_and_crash_is_generic(tmp_path, monkeypatch):
    module = tmp_path / "runtime_error_stub.py"
    module.write_text(
        textwrap.dedent(
            """
            import argparse, json, os
            from pathlib import Path
            p = argparse.ArgumentParser(); p.add_argument('--job'); p.add_argument('--options')
            a = p.parse_args(); job = json.loads(Path(a.job).read_text())
            try:
                if job['kind'] == 'value': raise ValueError('bad value')
                if job['kind'] == 'file': raise FileNotFoundError(2, 'missing input', '/missing.tif')
                os._exit(9)
            except BaseException as error:
                Path(a.job).with_suffix('.failure.json').write_text(json.dumps({
                    'schema_version': 1, 'exception_type': type(error).__name__,
                    'exception_module': type(error).__module__, 'args': list(error.args),
                    'message': str(error)
                }))
                raise
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", ""))))
    options = RuntimeOptions(memory_budget_bytes=1_000_000)
    with pytest.raises(ValueError, match="bad value"):
        execute_tiles([{"kind": "value", "memory_estimate_bytes": 10}], options,
                      worker_module="runtime_error_stub")
    with pytest.raises(FileNotFoundError, match="missing input"):
        execute_tiles([{"kind": "file", "memory_estimate_bytes": 10}], options,
                      worker_module="runtime_error_stub")
    with pytest.raises(TileExecutionError, match="exit code 9"):
        execute_tiles([{"kind": "crash", "memory_estimate_bytes": 10}], options,
                      worker_module="runtime_error_stub")
