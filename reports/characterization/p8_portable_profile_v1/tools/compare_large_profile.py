#!/usr/bin/env python3
"""Streaming comparison of a portable-profile run to a frozen original oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schema(ds):
    return {
        "size": [ds.RasterXSize, ds.RasterYSize, ds.RasterCount],
        "projection": ds.GetProjection(),
        "geotransform": list(ds.GetGeoTransform()),
        "metadata": ds.GetMetadata(),
        "bands": [
            {
                "dtype": ds.GetRasterBand(i).DataType,
                "nodata": ds.GetRasterBand(i).GetNoDataValue(),
                "description": ds.GetRasterBand(i).GetDescription(),
                "metadata": ds.GetRasterBand(i).GetMetadata(),
            }
            for i in range(1, ds.RasterCount + 1)
        ],
    }


def rule(name):
    if name == "TMRT":
        return "max", 0.01, 0.0
    if name in ("UTCI", "WBGT"):
        return "max", 0.02, 0.0
    if name in ("Shadow", "Ta", "Wind"):
        return "exact", 0.0, 0.0
    return "close", 0.05, 1e-5


def compare_raster(a: Path, b: Path, name: str, block=256, override=None):
    da, db = gdal.Open(str(a)), gdal.Open(str(b))
    if da is None or db is None:
        return {"path": a.name, "passed": False, "error": "missing_or_unreadable"}
    schema_equal = schema(da) == schema(db)
    typ, atol, rtol = override or rule(name)
    bands = []
    if da.RasterCount != db.RasterCount or da.RasterXSize != db.RasterXSize or da.RasterYSize != db.RasterYSize:
        return {"path": a.name, "schema_equal": schema_equal, "passed": False, "error": "shape_mismatch"}
    for band_index in range(1, da.RasterCount + 1):
        ba, bb = da.GetRasterBand(band_index), db.GetRasterBand(band_index)
        masks_equal = True
        values_equal = True
        max_abs = 0.0
        worst = [0, 0]
        for yoff in range(0, da.RasterYSize, block):
            ny = min(block, da.RasterYSize - yoff)
            for xoff in range(0, da.RasterXSize, block):
                nx = min(block, da.RasterXSize - xoff)
                x = ba.ReadAsArray(xoff, yoff, nx, ny)
                y = bb.ReadAsArray(xoff, yoff, nx, ny)
                finite = np.isfinite(x) & np.isfinite(y)
                same_masks = (
                    np.array_equal(np.isfinite(x), np.isfinite(y))
                    and np.array_equal(np.isnan(x), np.isnan(y))
                    and np.array_equal(np.isposinf(x), np.isposinf(y))
                    and np.array_equal(np.isneginf(x), np.isneginf(y))
                )
                masks_equal &= same_masks
                if typ == "exact":
                    values_equal &= np.array_equal(x, y)
                elif typ == "max":
                    values_equal &= bool(np.all(np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64)) <= atol))
                else:
                    values_equal &= bool(np.allclose(x[finite], y[finite], atol=atol, rtol=rtol))
                if finite.any():
                    d = np.zeros(x.shape, dtype=np.float64)
                    d[finite] = np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64))
                    idx = np.unravel_index(int(np.argmax(d)), d.shape)
                    if float(d[idx]) > max_abs:
                        max_abs = float(d[idx])
                        worst = [int(yoff + idx[0]), int(xoff + idx[1])]
        passed = masks_equal and values_equal
        bands.append({"band": band_index, "passed": passed, "max_abs": max_abs, "worst_coordinate": worst, "masks_equal": masks_equal})
    return {"path": a.name, "schema_equal": schema_equal, "passed": schema_equal and all(x["passed"] for x in bands), "rule": {"kind": typ, "atol": atol, "rtol": rtol}, "bands": bands}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--comparison-protocol", type=Path, required=True)
    p.add_argument("--reference-manifest", type=Path, required=True)
    p.add_argument("--reference-class", default="original upstream CPU SLEEF on M1")
    p.add_argument("--candidate-class", default="installed Linux portable-profile wheel")
    args = p.parse_args()
    ref, cand = args.reference, args.candidate
    artifacts = []
    expected = sorted(x.name for x in (ref / "output_folder/0_0").glob("*.tif"))
    actual = sorted(x.name for x in (cand / "output_folder/0_0").glob("*.tif"))
    for filename in expected:
        artifacts.append(compare_raster(ref / "output_folder/0_0" / filename, cand / "output_folder/0_0" / filename, filename.split("_")[0]))
    geometry = []
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        with zipfile.ZipFile(ref / "processed_inputs/SVF/svfs_0_0.zip") as z:
            z.extractall(ta)
        with zipfile.ZipFile(cand / "processed_inputs/SVF/svfs_0_0.zip") as z:
            z.extractall(tb)
        za, zb = Path(ta), Path(tb)
        ref_names = sorted(x.name for x in za.glob("*.tif"))
        cand_names = sorted(x.name for x in zb.glob("*.tif"))
        for filename in ref_names:
            geometry.append(compare_raster(za / filename, zb / filename, "SVF", override=("max", 1e-6, 0.0)))
    with np.load(ref / "processed_inputs/SVF/shadowmats_0_0.npz") as ar, np.load(cand / "processed_inputs/SVF/shadowmats_0_0.npz") as ac:
        ref_keys, cand_keys = sorted(ar.files), sorted(ac.files)
        for key in ref_keys:
            x, y = ar[key], ac[key]
            schema_equal = x.shape == y.shape and x.dtype == y.dtype
            equal = schema_equal and np.array_equal(x, y)
            max_abs = None
            if schema_equal and x.size:
                max_abs = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            geometry.append({"path": "shadowmats/" + key, "schema_equal": schema_equal, "passed": bool(equal), "max_abs": max_abs})
    inventory_ok = expected == actual and len(expected) == 10 and len(geometry) == 18
    passed = inventory_ok and all(x["passed"] for x in artifacts) and all(x["passed"] for x in geometry)
    report = {
        "schema_version": 1,
        "evidence_classes": {"reference": args.reference_class, "candidate": args.candidate_class},
        "comparison_protocol": {"path": str(args.comparison_protocol), "sha256": sha256(args.comparison_protocol)},
        "reference_manifest": {"path": str(args.reference_manifest), "sha256": sha256(args.reference_manifest)},
        "output_inventory": {"reference": expected, "candidate": actual, "equal": expected == actual},
        "geometry_inventory": {"zip_reference": ref_names, "zip_candidate": cand_names, "npz_reference": ref_keys, "npz_candidate": cand_keys},
        "passed": passed,
        "artifacts": artifacts,
        "geometry": geometry,
        "summary": {"output_count": len(artifacts), "output_band_count": sum(len(x.get("bands", [])) for x in artifacts), "failed_outputs": [x["path"] for x in artifacts if not x["passed"]], "geometry_count": len(geometry), "failed_geometry": [x["path"] for x in geometry if not x["passed"]], "inventory_ok": inventory_ok},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"] | {"passed": passed}, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
