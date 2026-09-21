"""Upstream CUDA geometry-warm correctness runner; never benchmark evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

LIMIT = 12 * 1024**3
COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".nbi", ".nbc")}


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def child(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    run, source = spec_path.parent, Path(spec["source"]).resolve()
    try:
        if spec["backend"] == "upstream_cuda":
            import torch
            import solweig_gpu
            from osgeo import gdal
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("CUDA admission requires exactly one visible GPU")
            torch.cuda.set_device(0)
            torch.set_num_threads(1)
            package = Path(solweig_gpu.__file__).resolve().parent
            revision = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            status = subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
            if revision != COMMIT or status:
                raise RuntimeError(f"Unpinned or dirty upstream: {revision}; {status}")
            source_files = {p.name: digest(p) for p in sorted((source / "solweig_gpu").iterdir())
                            if p.is_file() and p.suffix in (".py", ".txt")}
            for name, expected in source_files.items():
                if digest(package / name) != expected:
                    raise RuntimeError(f"Installed upstream differs: {name}")
            props = torch.cuda.get_device_properties(0)
            environment = dict(evidence_class="original_upstream_cuda", source_commit=revision,
                               source_package_sha256=source_files, package=str(package),
                               torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                               torch_interop_threads=torch.get_num_interop_threads(),
                               cuda_available=True, compiled_cuda=torch.version.cuda,
                               cuda_device_count=torch.cuda.device_count(), cuda_current_device=0,
                               cuda_name=props.name, cuda_total_memory=props.total_memory,
                               cuda_compute_capability=[props.major, props.minor],
                               gdal_version=gdal.VersionInfo())
            call = solweig_gpu.run_utci_tiles
            kwargs = {k: v for k, v in spec["kwargs"].items()
                      if k.startswith("save_") or k in ("base_path", "selected_date_str")}
            kwargs["preprocess_dir"] = str(Path(kwargs["base_path"]) / "processed_inputs")
            spec["kwargs"] = kwargs
            context = None
        else:
            import solweig_light
            package = Path(solweig_light.__file__).resolve()
            if not package.is_relative_to(source):
                raise RuntimeError(f"Candidate imported outside snapshot: {package}")
            environment = dict(evidence_class="candidate_cpu", package=str(package),
                               source_sha256=hashes(source))
            call = solweig_light.thermal_comfort
            context = solweig_light.runtime_options(solweig_light.RuntimeOptions(
                cpu_budget=1, workers=1, threads_per_worker=1,
                memory_budget_bytes=LIMIT, block_pixels=128,
                cache_enabled=True, cache_dir=str(run / "runtime_cache"),
                checkpoint_interval=1, legacy_cache_policy="recompute", resume=False))
        environment.update(python=sys.version, executable=sys.executable,
                           platform=platform.platform(), invocation=spec,
                           thread_environment={k: os.environ.get(k) for k in THREAD_ENV},
                           packages={d.metadata["Name"]: d.version
                                     for d in importlib.metadata.distributions()},
                           harness_sha256=digest(Path(__file__).resolve()))
        write(run / "environment.json", environment)
        start = time.perf_counter()
        torch.cuda.synchronize()
        if context is None:
            returned = call(**spec["kwargs"])
        else:
            with context:
                returned = call(**spec["kwargs"])
        torch.cuda.synchronize()
        write(run / "outcome.json", dict(status="executed_not_yet_compared",
                                           return_repr=repr(returned),
                                           workflow_seconds=time.perf_counter() - start,
                                           cuda_max_memory_allocated=torch.cuda.max_memory_allocated(0),
                                           cuda_max_memory_reserved=torch.cuda.max_memory_reserved(0)))
    except BaseException:
        write(run / "outcome.json", dict(status="failed", traceback=traceback.format_exc()))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("upstream_cuda",))
    parser.add_argument("--python")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--kwargs", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    if not all((args.backend, args.python, args.source, args.fixture, args.kwargs, args.run)):
        parser.error("all parent-mode arguments are required")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    scene = run / "scene"
    shutil.copytree(args.fixture, scene)
    output = scene / "output_folder"
    if output.exists():
        shutil.rmtree(output)
    kwargs = json.loads(args.kwargs.read_text())
    kwargs["base_path"] = str(scene)
    kwargs["own_met_file"] = str(scene / Path(kwargs["own_met_file"]).name)
    spec = dict(backend=args.backend, source=str(args.source.resolve()), kwargs=kwargs,
                fixture_sha256=hashes(args.fixture), cpu_threads=1,
                memory_limit_bytes=LIMIT, entrypoint="run_utci_tiles",
                cache_regime="geometry_warm_from_hash_verified_original_cpu_geometry",
                interpretation="correctness admission, not timing; does not replace first-use")
    write(run / "spec.json", spec)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0")
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    for key in THREAD_ENV:
        env[key] = "1"
    command = [args.python, str(Path(__file__).resolve()), "--child", str(run / "spec.json")]
    start, peak, samples, aborted = time.perf_counter(), 0, [], False
    import psutil
    with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=run, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        root = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    members = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    members = []
                rss = 0
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                elapsed = time.perf_counter() - start
                samples.append([elapsed, rss])
                peak = max(peak, rss)
                if rss > LIMIT:
                    aborted = True
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.02)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    write(run / "rss_samples.json", samples)
    write(run / "measurement.json", dict(command=command, exit_code=code,
          elapsed_seconds_including_imports_jit_io=time.perf_counter() - start,
          peak_process_tree_rss_bytes=peak, memory_limit_bytes=LIMIT,
          memory_limit_aborted=aborted, sample_interval_seconds=0.02,
          measurement_class="correctness admission diagnostic; not benchmark timing"))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
