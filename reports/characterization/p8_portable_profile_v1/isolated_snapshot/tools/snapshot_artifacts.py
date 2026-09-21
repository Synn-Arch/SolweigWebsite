#!/usr/bin/env python3
"""Snapshot and assert the pinned upstream scene's on-disk artifact contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

EXPECTED_OUTPUTS = 10
EXPECTED_BANDS = 24
EXPECTED_ROWS = 32
EXPECTED_COLS = 35
EXPECTED_UTCI_NAN_PER_BAND = 56
EXPECTED_ZIP_MEMBERS = 15
EXPECTED_NPZ_SHAPE = [32, 35, 153]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _number(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    return value if isinstance(value, (int, float)) and math.isfinite(value) else (str(value) if value is not None else None)


def tiff_snapshot(path: Path, display_path: str | None = None, open_path: str | None = None) -> dict[str, Any]:
    target = open_path or str(path)
    ds = gdal.Open(target, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL could not open {target}")
    bands = []
    for index in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(index)
        array = band.ReadAsArray()
        finite = np.isfinite(array)
        bands.append({
            "index": index,
            "dtype": gdal.GetDataTypeName(band.DataType),
            "nodata_present": band.GetNoDataValue() is not None,
            "nodata": _number(band.GetNoDataValue()),
            "metadata": dict(sorted(band.GetMetadata().items())),
            "numeric": {
                "finite": int(finite.sum()),
                "nan": int(np.isnan(array).sum()),
                "inf": int(np.isinf(array).sum()),
                "min": _number(np.nanmin(array)) if finite.any() else None,
                "max": _number(np.nanmax(array)) if finite.any() else None,
            },
            "nan_mask_sha256": sha256_bytes(np.ascontiguousarray(np.isnan(array)).tobytes()),
            "content_sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
        })
    return {
        "path": display_path or str(path),
        "dimensions": {"columns": ds.RasterXSize, "rows": ds.RasterYSize},
        "band_count": ds.RasterCount,
        "dtype": bands[0]["dtype"] if bands else None,
        "affine": list(ds.GetGeoTransform()),
        "crs_wkt": ds.GetProjectionRef(),
        "crs_authority": ds.GetSpatialRef().GetAuthorityCode(None) if ds.GetSpatialRef() else None,
        "bands": bands,
        "file_sha256": sha256_bytes(path.read_bytes()) if path.exists() and open_path is None else None,
    }


def npz_snapshot(path: Path, unique_cap: int = 64) -> dict[str, Any]:
    members = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            values, counts = np.unique(array, return_counts=True)
            values_out = [_number(v) for v in values[:unique_cap]]
            counts_out = [int(v) for v in counts[:unique_cap]]
            members[name] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "content_sha256": sha256_bytes(array.tobytes()),
                "numeric": {"finite": int(np.isfinite(array).sum()), "nan": int(np.isnan(array).sum()), "inf": int(np.isinf(array).sum())},
                "unique_count": int(values.size),
                "unique_values_capped": values_out,
                "unique_counts_capped": counts_out,
                "unique_values_truncated": bool(values.size > unique_cap),
            }
    return {"path": str(path), "file_sha256": sha256_bytes(path.read_bytes()), "members": members}


def zip_snapshot(path: Path, unique_cap: int = 64) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
    schemas = {}
    for name in names:
        schemas[name] = tiff_snapshot(path, display_path=name, open_path=f"/vsizip/{path.resolve()}/{name}")
    return {"path": str(path), "file_sha256": sha256_bytes(path.read_bytes()), "member_names": names, "member_count": len(names), "tiff_schemas": schemas}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("reports/runs/dependencies_cpu/scene"))
    parser.add_argument("--output", type=Path, default=Path("reports/initial_artifact_snapshot.json"))
    args = parser.parse_args()
    gdal.UseExceptions()
    root = args.run.resolve()
    output_dir = root / "output_folder" / "0_0"
    output_paths = sorted(output_dir.glob("*.tif"))
    outputs = {p.name: tiff_snapshot(p, str(p.relative_to(root))) for p in output_paths}
    utci = outputs.get("UTCI_0_0.tif")
    assertions = {
        "output_tiff_count": {"expected": EXPECTED_OUTPUTS, "actual": len(outputs), "passed": len(outputs) == EXPECTED_OUTPUTS},
        "output_tiff_schema": {"expected": {"bands": EXPECTED_BANDS, "rows": EXPECTED_ROWS, "columns": EXPECTED_COLS, "dtype": "Float32"}, "actual": {"bands": utci["band_count"] if utci else None, "rows": utci["dimensions"]["rows"] if utci else None, "columns": utci["dimensions"]["columns"] if utci else None, "dtype": utci["dtype"] if utci else None}, "passed": bool(utci and utci["band_count"] == EXPECTED_BANDS and utci["dimensions"] == {"rows": EXPECTED_ROWS, "columns": EXPECTED_COLS} and utci["dtype"] == "Float32")},
        "utci_nan_mask": {"expected_nan_per_band": EXPECTED_UTCI_NAN_PER_BAND, "actual_nan_per_band": sorted({b["numeric"]["nan"] for b in utci["bands"]}) if utci else [], "mask_sha256": sorted({b["nan_mask_sha256"] for b in utci["bands"]}) if utci else [], "passed": bool(utci and {b["numeric"]["nan"] for b in utci["bands"]} == {EXPECTED_UTCI_NAN_PER_BAND} and len({b["nan_mask_sha256"] for b in utci["bands"]}) == 1)},
    }
    expected_names = {f"{name}_0_0.tif" for name in
                      ("UTCI", "TMRT", "Kup", "Kdown", "Lup", "Ldown", "Shadow", "WBGT", "Ta", "Wind")}
    assertions["exact_output_names"] = {"passed": set(outputs) == expected_names}
    expected_times = [f"2020-07-18T{hour:02d}:00:00" for hour in range(24)]
    template = gdal.Open(str(root / "Building_DSM.tif"))
    assertions["all_output_schemas_and_timestamps"] = {"passed": bool(outputs) and all(
        item["band_count"] == EXPECTED_BANDS
        and item["dimensions"] == {"rows": EXPECTED_ROWS, "columns": EXPECTED_COLS}
        and item["affine"] == list(template.GetGeoTransform())
        and item["crs_wkt"] == template.GetProjection()
        and all(b["dtype"] == "Float32" and not b["nodata_present"] for b in item["bands"])
        and [b["metadata"].get("Time") for b in item["bands"]] == expected_times
        for item in outputs.values())}
    expected_roofs = np.zeros((EXPECTED_ROWS, EXPECTED_COLS), dtype=bool)
    expected_roofs[12:19, 14:22] = True
    roof_hash = sha256_bytes(expected_roofs.tobytes())
    assertions["utci_exact_roof_locations"] = {"passed": bool(utci) and all(
        b["nan_mask_sha256"] == roof_hash for b in utci["bands"])}
    svf = root / "processed_inputs" / "SVF"
    archive = svf / "svfs_0_0.zip"
    matrices = svf / "shadowmats_0_0.npz"
    zip_data = zip_snapshot(archive) if archive.exists() else None
    npz_data = npz_snapshot(matrices) if matrices.exists() else None
    assertions["svf_zip_members"] = {"expected": EXPECTED_ZIP_MEMBERS, "actual": zip_data["member_count"] if zip_data else None, "passed": bool(zip_data and zip_data["member_count"] == EXPECTED_ZIP_MEMBERS)}
    expected_zip = {f"svf{direction}{kind}.tif" for direction in ("", "E", "S", "W", "N") for kind in ("", "veg", "aveg")}
    assertions["exact_svf_zip_names_and_schemas"] = {"passed": bool(zip_data)
        and set(zip_data["member_names"]) == expected_zip
        and all(item["band_count"] == 1
                and item["dimensions"] == {"rows": EXPECTED_ROWS, "columns": EXPECTED_COLS}
                and item["dtype"] == "Float32"
                and item["affine"] == list(template.GetGeoTransform())
                and item["crs_wkt"] == template.GetProjection()
                for item in zip_data["tiff_schemas"].values())}
    assertions["npz_channels"] = {"expected": {"names": ["shadowmat", "vegshadowmat", "vbshmat"], "dtype": "float32", "shape": EXPECTED_NPZ_SHAPE}, "actual": {"names": sorted(npz_data["members"]) if npz_data else None, "schemas": {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in npz_data["members"].items()} if npz_data else None}, "passed": bool(npz_data and sorted(npz_data["members"]) == ["shadowmat", "vbshmat", "vegshadowmat"] and all(v["dtype"] == "float32" and v["shape"] == EXPECTED_NPZ_SHAPE for v in npz_data["members"].values()))}
    report = {"evidence_class": "original_upstream_cpu_artifact_snapshot", "run": str(root), "assertions": assertions, "outputs": outputs, "svf": {"standalone": tiff_snapshot(svf / "SkyViewFactor_0_0.tif", str((svf / "SkyViewFactor_0_0.tif").relative_to(root))) if (svf / "SkyViewFactor_0_0.tif").exists() else None, "zip": zip_data, "npz": npz_data}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} (all assertions passed: {all(v['passed'] for v in assertions.values())})")
    return 0 if all(v["passed"] for v in assertions.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
