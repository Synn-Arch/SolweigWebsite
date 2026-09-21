#!/usr/bin/env python3
"""Compare current candidate outputs to canonical original M1 SLEEF outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from osgeo import gdal


FIELDS = ("Kdown", "Kup", "Ldown", "Lup", "Shadow", "TMRT", "Ta", "UTCI", "WBGT", "Wind")
TOLERANCES = {
    "Kdown": (0.05, 1e-5), "Kup": (0.05, 1e-5), "Ldown": (0.05, 1e-5), "Lup": (0.05, 1e-5),
    "Shadow": (0.0, 0.0), "TMRT": (0.01, 0.0), "Ta": (0.0, 0.0), "UTCI": (0.02, 0.0),
    "WBGT": (0.02, 0.0), "Wind": (0.0, 0.0),
}
GEOMETRY_NAMES = ("svf.tif", "svfE.tif", "svfEaveg.tif", "svfEveg.tif", "svfN.tif", "svfNaveg.tif",
                  "svfNveg.tif", "svfS.tif", "svfSaveg.tif", "svfSveg.tif", "svfW.tif", "svfWaveg.tif",
                  "svfWveg.tif", "svfaveg.tif", "svfveg.tif")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def raster_compare(reference: Path, candidate: Path, atol: float, rtol: float) -> dict:
    fields = []
    passed = True
    ref, cand = gdal.Open(str(reference)), gdal.Open(str(candidate))
    if ref is None or cand is None:
        raise FileNotFoundError(f"cannot open {reference} or {candidate}")
    schema = (ref.RasterXSize == cand.RasterXSize and ref.RasterYSize == cand.RasterYSize
              and ref.RasterCount == cand.RasterCount and ref.GetGeoTransform() == cand.GetGeoTransform()
              and ref.GetProjection() == cand.GetProjection() and ref.GetMetadata() == cand.GetMetadata())
    for band in range(1, ref.RasterCount + 1):
            rb, cb = ref.GetRasterBand(band), cand.GetRasterBand(band)
            x, y = rb.ReadAsArray(), cb.ReadAsArray()
            masks = (np.array_equal(np.isfinite(x), np.isfinite(y))
                     and np.array_equal(np.isnan(x), np.isnan(y))
                     and np.array_equal(np.isposinf(x), np.isposinf(y))
                     and np.array_equal(np.isneginf(x), np.isneginf(y)))
            finite = np.isfinite(x) & np.isfinite(y)
            max_abs = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
            values = bool(np.allclose(x[finite], y[finite], atol=atol, rtol=rtol))
            metadata = (rb.GetMetadata() == cb.GetMetadata() and rb.GetNoDataValue() == cb.GetNoDataValue()
                        and rb.GetScale() == cb.GetScale() and rb.GetOffset() == cb.GetOffset()
                        and rb.DataType == cb.DataType)
            row_passed = schema and masks and values and metadata
            passed &= row_passed
            fields.append({"band": band, "max_abs": max_abs, "masks_equal": masks,
                           "metadata_equal": metadata, "passed": row_passed})
    ref, cand = None, None
    return {"schema_equal": schema, "bands": fields, "passed": passed}


def geometry_compare(reference_dir: Path, candidate_dir: Path) -> dict:
    rows = []
    passed = True
    for name in GEOMETRY_NAMES:
        with zipfile.ZipFile(reference_dir / "svfs_0_0.zip") as zr, zipfile.ZipFile(candidate_dir / "svfs_0_0.zip") as zc:
            rb, cb = zr.read(name), zc.read(name)
        rpath, cpath = f"/vsimem/e0-ref-{name}", f"/vsimem/e0-cand-{name}"
        gdal.FileFromMemBuffer(rpath, rb); gdal.FileFromMemBuffer(cpath, cb)
        ref, cand = gdal.Open(rpath), gdal.Open(cpath)
        x, y = ref.GetRasterBand(1).ReadAsArray(), cand.GetRasterBand(1).ReadAsArray()
        schema = (ref.RasterXSize == cand.RasterXSize and ref.RasterYSize == cand.RasterYSize
                  and ref.GetProjection() == cand.GetProjection()
                  and ref.GetGeoTransform() == cand.GetGeoTransform()
                  and ref.GetMetadata() == cand.GetMetadata()
                  and ref.GetRasterBand(1).DataType == cand.GetRasterBand(1).DataType)
        masks = np.array_equal(np.isfinite(x), np.isfinite(y)) and np.array_equal(np.isnan(x), np.isnan(y))
        equal = np.array_equal(x, y)
        ref, cand = None, None
        gdal.Unlink(rpath); gdal.Unlink(cpath)
        row_passed = schema and masks and equal
        passed &= row_passed
        rows.append({"path": name, "schema_equal": schema, "masks_equal": masks,
                     "values_equal": equal, "passed": row_passed,
                     "reference_sha256": hashlib.sha256(rb).hexdigest(),
                     "candidate_sha256": hashlib.sha256(cb).hexdigest()})
    with np.load(reference_dir / "shadowmats_0_0.npz") as ref, np.load(candidate_dir / "shadowmats_0_0.npz") as cand:
        if sorted(ref.files) != sorted(cand.files):
            raise RuntimeError("shadowmat key sets differ")
        for name in sorted(ref.files):
            x, y = ref[name], cand[name]
            masks = np.array_equal(np.isfinite(x), np.isfinite(y)) and np.array_equal(np.isnan(x), np.isnan(y))
            equal = np.array_equal(x, y)
            row_passed = masks and equal
            passed &= row_passed
            rows.append({"path": f"shadowmats/{name}", "schema_equal": x.shape == y.shape and x.dtype == y.dtype,
                         "masks_equal": masks, "values_equal": equal, "passed": row_passed,
                         "reference_sha256": sha(reference_dir / "shadowmats_0_0.npz"),
                         "candidate_sha256": sha(candidate_dir / "shadowmats_0_0.npz")})
    return {"count": len(rows), "passed": passed, "artifacts": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "local-cpu-optimization-e0-comparison.v1", "case": args.case,
              "reference": "canonical_original_upstream_CPU_SLEEF_macOS_ARM64",
              "candidate": "current_source_bound_installed_wheel",
              "performance_claim": None, "fields": [], "reference_output_manifest": [],
              "candidate_output_manifest": []}
    all_passed = True
    for field in FIELDS:
        result = raster_compare(args.reference / f"{field}_0_0.tif", args.candidate / f"{field}_0_0.tif", *TOLERANCES[field])
        report["fields"].append({"field": field, "atol": TOLERANCES[field][0], "rtol": TOLERANCES[field][1], **result})
        all_passed &= result["passed"]
        report["reference_output_manifest"].append({"field": field, "sha256": sha(args.reference / f"{field}_0_0.tif")})
        report["candidate_output_manifest"].append({"field": field, "sha256": sha(args.candidate / f"{field}_0_0.tif")})
    geometry = geometry_compare(args.reference.parents[1] / "processed_inputs" / "SVF",
                                args.candidate.parents[1] / "processed_inputs" / "SVF")
    report["geometry"] = geometry
    all_passed &= geometry["passed"]
    report["output_bands"] = len(FIELDS) * 24
    report["status"] = "passed" if all_passed else "failed"
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not all_passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
