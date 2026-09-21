#!/usr/bin/env python3
"""Capture canonical original-SLEEF inputs and classifier/large-scene oracles.

This tool must be run with ``.venv-oracle/bin/python``.  It imports only the
installed, unchanged ``solweig_gpu`` oracle.  The classifier corpus is frozen
before any candidate comparison and results are written one timestep at a time.
Large 1024 runs are merely prepared unless ``run-large`` is explicitly invoked.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/characterization/p8_portable_profile_v1/oracle"
UPSTREAM = ROOT / ".upstream/SOLWEIG-GPU"
COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
SOURCE_SHA256 = "1ac27bcd56edcd690d11fce756bedc17cc53d1259e0fe60313e868921a28095c"
CAP_BYTES = 12 * 1024**3
EXPECTED = {"torch": "2.14.0", "numpy": "2.4.6"}
SVF_SOURCES = {
    "dense1024": ROOT / "reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/dense_urban/raw/upstream/scene/processed_inputs/SVF/SkyViewFactor_0_0.tif",
    "vegetation1024": ROOT / "reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/vegetation_rich/raw/upstream/scene/processed_inputs/SVF/SkyViewFactor_0_0.tif",
}
LARGE_FIXTURES = {
    "dense1024": ROOT / "reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/mkl_pipeline_experiment/remote/dense_urban_retry/scene",
    "vegetation1024": ROOT / "reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/mkl_pipeline_experiment/remote/vegetation_rich/scene",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_oracle() -> dict:
    import torch
    import solweig_gpu
    from solweig_gpu import solweig

    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if revision != COMMIT or dirty:
        raise RuntimeError(f"upstream checkout is not pristine {COMMIT}: revision={revision}, dirty={dirty!r}")
    source = UPSTREAM / "solweig_gpu/solweig.py"
    installed = Path(solweig.__file__).resolve()
    if sha(source) != SOURCE_SHA256 or sha(installed) != SOURCE_SHA256:
        raise RuntimeError("solweig.py source or installed oracle hash mismatch")
    if not Path(solweig_gpu.__file__).resolve().is_relative_to((ROOT / ".venv-oracle").resolve()):
        raise RuntimeError(f"oracle package has unexpected origin: {solweig_gpu.__file__}")
    if torch.cuda.is_available():
        raise RuntimeError("canonical CPU oracle unexpectedly sees CUDA")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED or getattr(torch.version, "git_version", None) != "08187d9e0fba026dc8217405802ab5381dc88d90":
        raise RuntimeError(f"oracle dependency mismatch: {versions}, torch git={torch.version.git_version}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "evidence_class": "original_upstream_CPU_SLEEF_macOS_ARM64",
        "upstream_commit": revision,
        "solweig_source_sha256": SOURCE_SHA256,
        "solweig_package": str(Path(solweig_gpu.__file__).resolve()),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda": "not available",
    }


def read_tiff_unique_bits(path: Path) -> np.ndarray:
    from osgeo import gdal

    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    values = ds.GetRasterBand(1).ReadAsArray().astype(np.float32, copy=False)
    return np.unique(values[np.isfinite(values)].view(np.uint32))


def freeze_inputs() -> None:
    env = assert_oracle()
    old = ROOT / "reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/classifier_mkl_diagnostic/corpus.npz"
    frozen_dir = ROOT / "tests/reference/svf_original_cpu"
    source_records, arrays = {}, []
    with np.load(old) as data:
        boundary = np.asarray(data["boundary_bits"], dtype=np.uint32)
        patch_altitude = np.asarray(data["patch_altitude"])
        patch_azimuth = np.asarray(data["patch_azimuth"])
    source_records[str(old.relative_to(ROOT))] = sha(old)
    for path in sorted(frozen_dir.glob("*outputs.npz")):
        with np.load(path) as data:
            if "svf" not in data:
                continue
            arrays.append(np.asarray(data["svf"], dtype=np.float32).ravel().view(np.uint32))
        source_records[str(path.relative_to(ROOT))] = sha(path)
    existing = np.unique(np.concatenate(arrays))
    scene_arrays = []
    scene_counts = {}
    for name, path in SVF_SOURCES.items():
        bits = read_tiff_unique_bits(path)
        scene_arrays.append(bits)
        scene_counts[name] = int(bits.size)
        source_records[str(path.relative_to(ROOT))] = sha(path)
    scene_unique = np.unique(np.concatenate(scene_arrays))
    all_bits = np.unique(np.concatenate((boundary, existing, scene_unique)))
    corpus = OUT / "classifier_inputs.npz"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(corpus, input_bits=all_bits, boundary_bits=boundary, existing_bits=existing,
             scene_bits=scene_unique, patch_altitude=patch_altitude, patch_azimuth=patch_azimuth)
    manifest = {
        "schema": "portable-profile-original-classifier-inputs.v1",
        "status": "frozen_before_candidate_comparison",
        "environment": env,
        "counts": {"boundary": int(boundary.size), "existing_unique": int(existing.size),
                   "scene_unique": int(scene_unique.size), "all_unique": int(all_bits.size),
                   **scene_counts},
        "source_sha256": source_records,
        "corpus": corpus.name,
        "corpus_sha256": sha(corpus),
        "command": [sys.executable, str(Path(__file__).resolve()), "freeze-inputs"],
        "tool_sha256": sha(Path(__file__)),
    }
    write_json(OUT / "classifier_inputs_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def solar_events() -> list[dict]:
    boundary = ROOT / "tests/reference/small_original_cpu/boundaries"
    manifest = json.loads((boundary / "manifest.json").read_text())
    events = []
    for event in manifest["events"]:
        if event["boundary"] != "input":
            continue
        with np.load(boundary / event["path"]) as data:
            events.append({"timestep": int(event["timestep"]), "altitude": float(data["altitude"]),
                           "azimuth": float(data["azimuth"]), "source": event["path"]})
    if len(events) != 24 or sorted(e["timestep"] for e in events) != list(range(24)):
        raise RuntimeError("expected exactly 24 ordered solar input events")
    return sorted(events, key=lambda event: event["timestep"])


def capture_classifier() -> None:
    env = assert_oracle()
    import torch
    from solweig_gpu import solweig

    corpus = OUT / "classifier_inputs.npz"
    if not corpus.exists():
        raise RuntimeError("run freeze-inputs before capture-classifier")
    frozen_manifest = json.loads((OUT / "classifier_inputs_manifest.json").read_text())
    if sha(corpus) != frozen_manifest["corpus_sha256"]:
        raise RuntimeError("frozen classifier corpus hash mismatch")
    with np.load(corpus) as data:
        values = data["input_bits"].copy().view(np.float32)
        pa = torch.from_numpy(data["patch_altitude"].copy())
        pazi = torch.from_numpy(data["patch_azimuth"].copy())
    asvf = torch.acos(torch.sqrt(torch.from_numpy(values.copy())))[:, None]
    records = []
    for event in solar_events():
        sa = torch.tensor(event["altitude"], dtype=torch.float64)
        saz = torch.tensor(event["azimuth"], dtype=torch.float64)
        suns, shades, coefficient_bits, wrapped_bits = [], [], [], []
        for patch in range(153):
            # This is the original scalar call.  Tensor-origin float32 patch values
            # mix with float64 0-d solar values exactly as upstream executes them.
            sun, shade = solweig.shaded_or_sunlit(sa, saz, pa[patch], pazi[patch], asvf)
            xi = torch.cos(torch.abs(saz - pazi[patch]) * (torch.pi / 180.0))
            yi = 2 * xi * torch.tan(sa * (torch.pi / 180.0))
            coefficient_bits.append(np.asarray(yi.numpy(), dtype=np.float64).view(np.uint64))
            wrapped_bits.append(np.asarray(yi.to(torch.float32).numpy(), dtype=np.float32).view(np.uint32))
            suns.append(np.packbits(sun.numpy()[:, 0], bitorder="little"))
            shades.append(np.packbits(shade.numpy()[:, 0], bitorder="little"))
        output = OUT / f"classifier_timestep_{event['timestep']:02d}.npz"
        np.savez(output, sun_pack=np.stack(suns), shade_pack=np.stack(shades),
                 coefficient_bits=np.asarray(coefficient_bits, dtype=np.uint64),
                 coefficient_float32_bits=np.asarray(wrapped_bits, dtype=np.uint32))
        records.append({**event, "path": output.name, "sha256": sha(output)})
        print(f"captured timestep {event['timestep']:02d}: {output}", flush=True)
    write_json(OUT / "classifier_oracle_manifest.json", {
        "schema": "portable-profile-original-sleef-classifier.v1",
        "status": "captured",
        "environment": env,
        "input_manifest_sha256": sha(OUT / "classifier_inputs_manifest.json"),
        "input_corpus_sha256": sha(corpus),
        "actual_call_semantics": "153 original upstream scalar patch calls per timestep; float32 tensor patch scalars with float64 0-d solar tensors",
        "mask_storage": "numpy packbits little bitorder; shape [153, ceil(input_count/8)]",
        "timesteps": records,
        "command": [sys.executable, str(Path(__file__).resolve()), "capture-classifier"],
        "tool_sha256": sha(Path(__file__)),
    })


def capture_cura_primitives() -> None:
    """Capture the retained 189,220-value Cura corpus on canonical M1 SLEEF."""
    env = assert_oracle()
    import torch

    source = ROOT / "reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/asvf_torch_characterization/corpus.npz"
    source_manifest = source.with_name("corpus_manifest.json")
    with np.load(source) as data:
        bits = np.asarray(data["input_bits"], dtype=np.uint32)
    if bits.size != 189220 or np.unique(bits).size != bits.size:
        raise RuntimeError(f"unexpected Cura corpus cardinality: {bits.size}")
    values = torch.from_numpy(bits.copy().view(np.float32))
    sqrt = torch.sqrt(values)
    asvf = torch.acos(sqrt)
    output = OUT / "cura_189220_primitive_oracle.npz"
    np.savez(output, input_bits=bits, sqrt_bits=sqrt.numpy().view(np.uint32),
             asvf_bits=asvf.numpy().view(np.uint32))
    write_json(OUT / "cura_189220_primitive_manifest.json", {
        "schema": "portable-profile-original-sleef-cura-primitive.v1", "status": "captured",
        "environment": env, "input_count": int(bits.size), "source": str(source.relative_to(ROOT)),
        "source_sha256": sha(source), "source_manifest": str(source_manifest.relative_to(ROOT)),
        "source_manifest_sha256": sha(source_manifest), "output": output.name,
        "output_sha256": sha(output), "operations": ["torch.sqrt(float32)", "torch.acos(float32)"],
        "command": [sys.executable, str(Path(__file__).resolve()), "capture-cura-primitives"],
        "tool_sha256": sha(Path(__file__)),
    })


def prepare_large() -> None:
    env = assert_oracle()
    records = {}
    for name, fixture in LARGE_FIXTURES.items():
        if not fixture.is_dir():
            raise FileNotFoundError(fixture)
        kwargs = json.loads((fixture / "kwargs.json").read_text())
        # UTCI is an unconditional upstream output and has no save_utci argument.
        kwargs.update({f"save_{field}": True for field in ("tmrt", "wbgt", "kup", "kdown", "lup", "ldown", "shadow", "ta", "wind")})
        spec_dir = OUT / "large" / name
        spec_dir.mkdir(parents=True, exist_ok=True)
        clean_fixture = spec_dir / "fixture"
        clean_fixture.mkdir(exist_ok=True)
        # Never seed the canonical run from retained Cura/MKL outputs or caches.
        # Copy only immutable public-workflow inputs into the prepared fixture.
        input_names = ("Building_DSM.tif", "DEM.tif", "Trees.tif", "Landcover.tif",
                       "met.txt", "manifest.json")
        for input_name in input_names:
            source = fixture / input_name
            if source.is_file():
                destination = clean_fixture / input_name
                if destination.exists() and sha(destination) != sha(source):
                    raise RuntimeError(f"prepared fixture changed: {destination}")
                if not destination.exists():
                    shutil.copyfile(source, destination)
        kwargs_path = spec_dir / "kwargs.json"
        write_json(kwargs_path, kwargs)
        fixture_hashes = {str(p.relative_to(clean_fixture)): sha(p) for p in sorted(clean_fixture.iterdir()) if p.is_file()}
        command = [str(ROOT / ".venv-oracle/bin/python"), str(Path(__file__).resolve()),
                   "large-child", name]
        records[name] = {"retained_input_source": str(fixture.relative_to(ROOT)),
                         "fixture": str(clean_fixture.relative_to(ROOT)), "fixture_sha256": fixture_hashes,
                         "kwargs_sha256": sha(kwargs_path), "command": command, "status": "prepared_not_run"}
    write_json(OUT / "large_run_plan.json", {
        "schema": "portable-profile-original-sleef-large-plan.v1", "status": "prepared_not_run",
        "environment": env, "constraints": {"sequential": True, "process_tree_rss_cap_bytes": CAP_BYTES,
        "fields": 10, "bands": 24, "patches": 153, "no_candidate_import": True}, "cases": records,
        "preparation_command": [sys.executable, str(Path(__file__).resolve()), "prepare-large"],
        "tool_sha256": sha(Path(__file__)),
    })
    print(json.dumps(records, indent=2, sort_keys=True))


def run_large(case: str) -> None:
    """Run one prepared case with an enforced process-tree RSS cap."""
    assert_oracle()
    import psutil

    plan = json.loads((OUT / "large_run_plan.json").read_text())
    spec = plan["cases"][case]
    if spec["status"] != "prepared_not_run":
        raise RuntimeError("large plan is not in prepared_not_run state")
    command = spec["command"]
    log_dir = OUT / "large" / case
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    peak, samples, aborted = 0, [], False
    with (log_dir / "launcher_stdout.log").open("w") as out, (log_dir / "launcher_stderr.log").open("w") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            total = 0
            try:
                members = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                members = []
            for member in members:
                try:
                    total += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak = max(peak, total); samples.append(total)
            if total > CAP_BYTES:
                aborted = True; process.terminate()
                try: process.wait(10)
                except subprocess.TimeoutExpired: process.kill()
                break
            time.sleep(0.02)
        code = process.wait()
    np.save(log_dir / "launcher_rss_samples.npy", np.asarray(samples, dtype=np.uint64))
    write_json(log_dir / "launcher_measurement.json", {"command": command, "exit_code": code,
        "memory_limit_aborted": aborted, "peak_process_tree_rss_bytes": peak,
        "rss_cap_bytes": CAP_BYTES, "sample_interval_seconds": 0.02})
    if code or aborted:
        raise SystemExit(code or 70)


def large_child(case: str) -> None:
    """Execute one original workflow; called only by the monitored parent."""
    env_record = assert_oracle()
    import solweig_gpu

    plan = json.loads((OUT / "large_run_plan.json").read_text())
    spec = plan["cases"][case]
    fixture = ROOT / spec["fixture"]
    for name, digest in spec["fixture_sha256"].items():
        if sha(fixture / name) != digest:
            raise RuntimeError(f"fixture changed before execution: {name}")
    run = OUT / "large" / case / "run"
    if run.exists():
        raise RuntimeError(f"run destination already exists: {run}")
    run.mkdir()
    scene = run / "scene"
    shutil.copytree(fixture, scene)
    kwargs = json.loads((OUT / "large" / case / "kwargs.json").read_text())
    kwargs["base_path"] = str(scene)
    kwargs["own_met_file"] = str(scene / Path(kwargs["own_met_file"]).name)
    write_json(run / "kwargs.json", kwargs)
    env_record.update({"invocation_kwargs": kwargs, "fixture_sha256": spec["fixture_sha256"],
                       "command": [sys.executable, str(Path(__file__).resolve()), "large-child", case],
                       "tool_sha256": sha(Path(__file__))})
    write_json(run / "environment.json", env_record)
    started = time.perf_counter()
    try:
        returned = solweig_gpu.thermal_comfort(**kwargs)
    except BaseException:
        write_json(run / "outcome.json", {"status": "failed", "traceback": traceback.format_exc(),
                                           "workflow_seconds": time.perf_counter() - started})
        raise
    write_json(run / "outcome.json", {"status": "executed_not_yet_verified", "return_repr": repr(returned),
                                       "workflow_seconds": time.perf_counter() - started})


def verify_packet() -> None:
    """Validate packet structure and exact overlap with retained M1 captures."""
    assert_oracle()
    inputs_manifest = json.loads((OUT / "classifier_inputs_manifest.json").read_text())
    oracle_manifest = json.loads((OUT / "classifier_oracle_manifest.json").read_text())
    primitive_manifest = json.loads((OUT / "cura_189220_primitive_manifest.json").read_text())
    if sha(OUT / primitive_manifest["output"]) != primitive_manifest["output_sha256"]:
        raise RuntimeError("189220 primitive oracle hash mismatch")
    with np.load(OUT / "classifier_inputs.npz") as data:
        all_bits = data["input_bits"].copy()
        legacy_bits = np.concatenate((data["boundary_bits"], data["existing_bits"]))
    positions = np.searchsorted(all_bits, legacy_bits)
    if not np.array_equal(all_bits[positions], legacy_bits):
        raise RuntimeError("legacy corpus is not an exact subset of frozen inputs")
    overlap = []
    for record in oracle_manifest["timesteps"]:
        path = OUT / record["path"]
        if sha(path) != record["sha256"]:
            raise RuntimeError(f"oracle hash mismatch: {path}")
        with np.load(path) as data:
            sun = np.unpackbits(data["sun_pack"], axis=1, count=len(all_bits), bitorder="little")
            shade = np.unpackbits(data["shade_pack"], axis=1, count=len(all_bits), bitorder="little")
            if np.any(sun & shade):
                raise RuntimeError(f"sun/shade masks overlap: {path}")
        legacy = ROOT / "reports/characterization/p8_asvf_parity_review/m1_full_solar" / f"oracle_{record['timestep']:02d}.npz"
        with np.load(legacy) as expected:
            sun_mismatch = int(np.count_nonzero(sun[:, positions].T != expected["sun"]))
            shade_mismatch = int(np.count_nonzero(shade[:, positions].T != expected["shade"]))
        overlap.append({"timestep": record["timestep"], "retained_sha256": sha(legacy),
                        "sun_mismatches": sun_mismatch, "shade_mismatches": shade_mismatch})
    if any(row["sun_mismatches"] or row["shade_mismatches"] for row in overlap):
        raise RuntimeError("new canonical capture disagrees with retained original M1 capture")
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"packet_validation.json", "evidence.sha256"}:
            files[str(path.relative_to(OUT))] = sha(path)
    validation = {"schema": "portable-profile-original-oracle-validation.v1", "status": "passed",
                  "input_manifest_sha256": sha(OUT / "classifier_inputs_manifest.json"),
                  "oracle_manifest_sha256": sha(OUT / "classifier_oracle_manifest.json"),
                  "cura_189220_primitive_manifest_sha256": sha(OUT / "cura_189220_primitive_manifest.json"),
                  "large_plan_sha256": sha(OUT / "large_run_plan.json"),
                  "counts": inputs_manifest["counts"], "retained_m1_overlap": overlap,
                  "large_runs": "prepared_not_run", "files_sha256": files,
                  "command": [sys.executable, str(Path(__file__).resolve()), "verify-packet"],
                  "tool_sha256": sha(Path(__file__))}
    write_json(OUT / "packet_validation.json", validation)
    lines = [f"{digest}  {name}" for name, digest in sorted({**files, "packet_validation.json": sha(OUT / "packet_validation.json")}.items())]
    (OUT / "evidence.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "passed", "files": len(lines), "counts": inputs_manifest["counts"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-inputs")
    sub.add_parser("capture-classifier")
    sub.add_parser("capture-cura-primitives")
    sub.add_parser("prepare-large")
    sub.add_parser("verify-packet")
    run = sub.add_parser("run-large")
    run.add_argument("case", choices=sorted(LARGE_FIXTURES))
    child = sub.add_parser("large-child")
    child.add_argument("case", choices=sorted(LARGE_FIXTURES))
    args = parser.parse_args()
    {"freeze-inputs": freeze_inputs, "capture-classifier": capture_classifier,
     "capture-cura-primitives": capture_cura_primitives,
     "prepare-large": prepare_large, "verify-packet": verify_packet,
     "large-child": lambda: large_child(args.case)}.get(
         args.command, lambda: run_large(args.case))()


if __name__ == "__main__":
    main()
