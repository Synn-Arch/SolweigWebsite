"""Execute real upstream save-flag and cache-hit/miss contracts on a small scene."""
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"tmrt": "TMRT", "svf": "SVF", "kup": "Kup", "kdown": "Kdown",
         "lup": "Lup", "ldown": "Ldown", "shadow": "Shadow", "wbgt": "WBGT",
         "ta": "Ta", "wind": "Wind"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=False)
    if args.report.exists():
        raise FileExistsError(args.report)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from osgeo import gdal
    import solweig_gpu

    gdal.UseExceptions()
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    source = ROOT / ".upstream/SOLWEIG-GPU"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "--name-only"], text=True).strip()
    package = Path(solweig_gpu.__file__).parent
    source_hashes = {}
    for path in (source / "solweig_gpu").iterdir():
        if path.suffix in {".py", ".txt"}:
            assert sha(path) == sha(package / path.name)
            source_hashes[path.name] = sha(path)
    reference = ROOT / "tests/reference/small_original_cpu/scene"
    golden = reference / "output_folder/0_0"
    cases = [("all_off", set(), False), ("default", {"tmrt"}, False)]
    cases += [("only_" + flag, {flag}, False) for flag in FLAGS]
    cases += [("all_on", set(FLAGS), False), ("cold_all_on", set(FLAGS), True)]
    report = {"evidence_class": "original_upstream_cpu", "source_commit": revision,
              "oracle_patch_hash": None, "source_package_sha256": source_hashes,
              "script_sha256": sha(Path(__file__)), "invocation": sys.argv,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}},
              "input_files_sha256": {str(p.relative_to(reference)): sha(p) for p in reference.rglob("*") if p.is_file() and "output_folder" not in p.parts},
              "cases": [], "limitations": ["One small scene; not every combination or stale/corrupt cache behavior", "No performance claim; runs share a process", "No candidate output"]}
    for name, enabled, cold in cases:
        scene = args.run / name
        shutil.copytree(reference, scene, ignore=shutil.ignore_patterns("output_folder"))
        cache = scene / "processed_inputs/SVF"
        if cold:
            shutil.rmtree(cache)
        cache_before = {p.name: sha(p) for p in cache.glob("*") if p.is_file()}
        flags = {"save_" + flag: flag in enabled for flag in FLAGS}
        kwargs = {"base_path": str(scene.resolve()), "preprocess_dir": str((scene / "processed_inputs").resolve()),
                  "selected_date_str": "2020-07-18", **flags}
        if name == "default":
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("save_")}
        entry = {"name": name, "cache_state": "cold" if cold else "warm", "kwargs": kwargs}
        try:
            result = solweig_gpu.run_utci_tiles(**kwargs)
            actual = {p.name for p in (scene / "output_folder/0_0").glob("*.tif")}
            expected = {"UTCI_0_0.tif"} | {f"{FLAGS[f]}_0_0.tif" for f in enabled if f != "svf" or cold}
            checks = {"exact_names": actual == expected, "return_none": result is None}
            artifacts = {}
            for filename in sorted(actual):
                path = scene / "output_folder/0_0" / filename
                comparison = golden / filename if filename != "SVF_0_0.tif" else reference / "processed_inputs/SVF/SkyViewFactor_0_0.tif"
                ds, oracle = gdal.Open(str(path)), gdal.Open(str(comparison))
                values, expected_values = ds.ReadAsArray(), oracle.ReadAsArray()
                equal = np.array_equal(values, expected_values, equal_nan=True)
                schema = (ds.RasterCount == oracle.RasterCount and ds.GetGeoTransform() == oracle.GetGeoTransform()
                          and ds.GetProjection() == oracle.GetProjection()
                          and all(ds.GetRasterBand(i).GetMetadata() == oracle.GetRasterBand(i).GetMetadata()
                                  and ds.GetRasterBand(i).DataType == oracle.GetRasterBand(i).DataType
                                  and ds.GetRasterBand(i).GetNoDataValue() == oracle.GetRasterBand(i).GetNoDataValue()
                                  for i in range(1, ds.RasterCount + 1)))
                checks[filename] = equal and schema
                artifacts[filename] = {"sha256": sha(path), "array_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                                       "shape": list(values.shape), "dtype": str(values.dtype), "exact_values": equal, "exact_schema": schema}
            cache_after = {p.name: sha(p) for p in cache.glob("*") if p.is_file()}
            checks["cache_artifact_names"] = set(cache_after) == {"SkyViewFactor_0_0.tif", "svfs_0_0.zip", "shadowmats_0_0.npz"}
            if not cold:
                checks["warm_cache_unchanged"] = cache_after == cache_before
            entry.update(status="passed" if all(checks.values()) else "comparison_failed", checks=checks,
                         expected_names=sorted(expected), actual_names=sorted(actual), artifacts=artifacts)
        except BaseException:
            entry.update(status="execution_failed", traceback=traceback.format_exc())
        report["cases"].append(entry)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(name, entry["status"], flush=True)
    if not all(case["status"] == "passed" for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
