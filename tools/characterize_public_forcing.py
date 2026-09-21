"""Original public preprocess integration; WRF repair is separately labeled."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from osgeo import gdal, osr
    from solweig_gpu import preprocessor, solweig_gpu as public
    checkout = ROOT / ".upstream/SOLWEIG-GPU"
    if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise RuntimeError("upstream pin mismatch")
    if subprocess.check_output(["git", "-C", str(checkout), "diff", "--name-only"], text=True).strip():
        raise RuntimeError("upstream checkout modified")
    for module, name in ((preprocessor, "preprocessor.py"), (public, "solweig_gpu.py")):
        if sha(module.__file__) != sha(checkout / "solweig_gpu" / name):
            raise RuntimeError("original installed source mismatch")
    if "solweig_light" in sys.modules:
        raise RuntimeError("candidate import in oracle")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "raw"
    raw.mkdir()
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    grid = np.arange(16, dtype=np.float32).reshape(4, 4)
    for name, array in (("DEM", grid * .1), ("Building_DSM", grid * .1 + (grid % 3 == 0) * 5),
                        ("Trees", (grid % 4 == 0) * np.float32(3)), ("Landcover", np.full((4, 4), 2, np.float32))):
        ds = gdal.GetDriverByName("GTiff").Create(str(raw / f"{name}.tif"), 4, 4, 1, gdal.GDT_Float32)
        ds.SetProjection(srs.ExportToWkt()); ds.SetGeoTransform((-74.3, .175, 0, 41.0, 0, -.175))
        ds.GetRasterBand(1).WriteArray(array); ds.GetRasterBand(1).SetNoDataValue(-9999); ds = None
    report = {"upstream_commit": COMMIT, "source_sha256": {name: sha(module.__file__) for module, name in ((preprocessor, "preprocessor.py"), (public, "solweig_gpu.py"))},
              "collector_sha256": sha(__file__), "invocation": sys.argv,
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
              "cases": []}
    def run(label, source, folder, date, start, end, uhi, evidence):
        kwargs = dict(base_path=str(raw), selected_date_str=date, landcover_filename="Landcover.tif",
                      tile_size=2, overlap=1, use_own_met=False, start_time=start, end_time=end,
                      data_source_type=source, data_folder=str(folder), preprocess_dir=str(args.output / label), use_uhi=uhi)
        entry = {"label": label, "evidence_class": evidence, "arguments": kwargs}
        try:
            returned = public.preprocess(**kwargs)
            entry.update(status="success", returned_preprocess_dir=returned)
        except Exception as error:
            entry.update(status="failure", exception_type=type(error).__name__, exception_message=str(error))
        entry["arguments"]["base_path"] = {"packet_path": "raw"}
        entry["arguments"]["preprocess_dir"] = {"packet_path": label}
        entry["arguments"]["data_folder"] = {"repository_path": str(folder.relative_to(ROOT))}
        report["cases"].append(entry)
    for coord in ("time", "valid_time"):
        for uhi in (False, True):
            run(f"{coord}-{'uhi' if uhi else 'standard'}", "ERA5", ROOT / "tests/reference/forcing_original_cpu" / coord,
                "2024-11-03", "2024-11-03 00:00:00", "2024-11-04 12:00:00", uhi, "original_upstream_cpu_reference")
    wrf = ROOT / "tests/reference/wrf_patched_cpu"
    run("wrf-original-failure", "wrfout", wrf / "three-hour-rotation", "2024-06-21", "2024-06-21 12:00:00", "2024-06-21 14:00:00", True, "original_upstream_cpu_reference")
    # Public wrapper resolves this isolated, complete source copy via its normal
    # relative import. No numerical functions or inputs are replaced by mocks.
    source = Path(preprocessor.__file__).read_text()
    before = "start_time <= extract_datetime_strict(f) <= end_time"
    after = "start_time <= extract_datetime_strict(f)[0] <= end_time"
    if source.count(before) != 1:
        raise RuntimeError("timestamp repair not unique")
    patched_path = args.output / "preprocessor_wrf_timestamp_v1.py"
    patched_path.write_text(source.replace(before, after))
    patched = types.ModuleType("solweig_gpu.preprocessor")
    patched.__file__ = str(patched_path)
    exec(compile(patched_path.read_text(), str(patched_path), "exec"), patched.__dict__)
    sys.modules["solweig_gpu.preprocessor"] = patched
    report["wrf_repair"] = {"policy": "wrf_timestamp_v1", "patched_source_sha256": sha(patched_path),
                            "patch": "docs/upstream_patches/wrf_timestamp_v1.patch", "patch_sha256": sha(ROOT / "docs/upstream_patches/wrf_timestamp_v1.patch")}
    run("wrf-patched-three-hour", "wrfout", wrf / "three-hour-rotation", "2024-06-21", "2024-06-21 12:00:00", "2024-06-21 14:00:00", True, "patched_upstream_cpu_reference")
    run("wrf-patched-domain-shape-failure", "wrfout", wrf / "duplicate-domain-shape-failure", "2024-06-21", "2024-06-21 12:00:00", "2024-06-21 12:00:00", True, "patched_upstream_cpu_reference")
    report["files"] = {str(path.relative_to(args.output)): sha(path) for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Captured {len(report['cases'])} public preprocessing calls")


if __name__ == "__main__":
    main()
