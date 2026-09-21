"""Execute an installed upstream oracle in isolation and retain failures and RSS.

Run with the oracle environment's Python; no candidate modules are imported.
The child records provenance before invoking the public workflow. The parent
samples summed process-tree RSS (not Python allocations) at 20 ms intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext

COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def child(run, kwargs, capture=False, threads=None):
    import torch
    import solweig_gpu
    from osgeo import gdal

    if torch.cuda.is_available():
        raise RuntimeError("CPU oracle unexpectedly sees CUDA")
    if threads is not None:
        torch.set_num_threads(threads)
    package = Path(solweig_gpu.__file__).parent
    source = Path(__file__).resolve().parents[1] / ".upstream/SOLWEIG-GPU"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only"], text=True).strip()
    if revision != COMMIT or dirty:
        raise RuntimeError(f"Unpinned or modified upstream: {revision}, {dirty}")
    hashes = {}
    for path in sorted((source / "solweig_gpu").iterdir()):
        if not path.is_file() or path.suffix not in {".py", ".txt"}:
            continue
        installed = package / path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(installed.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Installed oracle differs from source: {path.name}")
        hashes[path.name] = digest
    env = {
        "evidence_class": "original_upstream_cpu", "source_commit": revision,
        "source_package_sha256": hashes, "oracle_patch_hash": None,
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "package_path": str(package), "gdal_version": gdal.VersionInfo(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "cuda": "not available", "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "invocation_kwargs": kwargs,
        "harness_sha256": {name: hashlib.sha256((run / "harness" / name).read_bytes()).hexdigest()
                           for name in ("run_reference.py", "capture_reference.py")},
        "requested_native_threads": threads,
        "candidate_state": subprocess.run(["git", "-C", str(source.parent.parent), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip() or "uncommitted repository without HEAD",
    }
    write_json(run / "environment.json", env)
    start = time.perf_counter()
    try:
        context = nullcontext()
        if capture:
            from capture_reference import BoundaryCapture
            from solweig_gpu import solweig
            context = BoundaryCapture(solweig, run / "boundaries")
        with context:
            returned = solweig_gpu.thermal_comfort(**kwargs)
    except BaseException:
        write_json(run / "outcome.json", {"status": "failed", "traceback": traceback.format_exc(), "workflow_seconds": time.perf_counter() - start})
        raise
    write_json(run / "outcome.json", {"status": "executed_not_yet_verified", "return_repr": repr(returned), "workflow_seconds": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--kwargs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--capture", action="store_true", help="Record numerical boundaries; invalidates performance interpretation")
    parser.add_argument("--threads", type=int, help="Explicit native thread budget for CPU baseline tuning")
    args = parser.parse_args()
    run = args.run.resolve()
    kwargs = json.loads(args.kwargs.read_text())
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.child:
        child(run, kwargs, args.capture, args.threads)
        return
    import psutil

    run.mkdir(parents=True, exist_ok=False)
    (run / "harness").mkdir()
    for name in ("run_reference.py", "capture_reference.py"):
        shutil.copyfile(Path(__file__).resolve().parent / name, run / "harness" / name)
    if args.fixture is None:
        parser.error("--fixture required")
    scene = run / "scene"
    shutil.copytree(args.fixture, scene)
    kwargs["base_path"] = str(scene)
    kwargs["own_met_file"] = str(scene / Path(kwargs["own_met_file"]).name)
    write_json(run / "kwargs.json", kwargs)
    fixture_hashes = {str(p.relative_to(args.fixture)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.fixture.rglob("*")) if p.is_file()}
    write_json(run / "fixture_hashes.json", fixture_hashes)
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--run", str(run), "--kwargs", str(run / "kwargs.json")]
    if args.capture:
        command.append("--capture")
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    if args.threads is not None:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(args.threads)
    env.pop("PYTHONPATH", None)
    start = time.perf_counter()
    peak = 0
    samples = 0
    with (run / "stdout.log").open("w") as out, (run / "stderr.log").open("w") as err:
        process = subprocess.Popen(command, cwd=run, env=env, stdout=out, stderr=err)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            rss = 0
            try:
                members = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                members = []
            for member in members:
                try:
                    rss += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak = max(peak, rss)
            samples += 1
            time.sleep(0.02)
        code = process.wait()
    write_json(run / "measurement.json", {
        "command": command, "exit_code": code,
        "elapsed_seconds_including_imports": time.perf_counter() - start,
        "sampled_process_tree_peak_rss_bytes": peak, "sample_count": samples,
        "sample_interval_seconds": 0.02,
        "rss_caveat": "Summed resident sets; shared pages may be counted more than once; between-sample peaks may be missed.",
        "cache_state": "fresh copied input fixture; OS file cache uncontrolled",
        "measurement_class": "instrumented diagnostic; not a benchmark" if args.capture else "initial characterization, not release benchmark",
    })
    print(run)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
