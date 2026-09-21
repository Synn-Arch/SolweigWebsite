"""Check exact TIFF data/metadata repeatability for two original oracle runs."""
import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    gdal.UseExceptions()
    left = {p.relative_to(args.left) for p in args.left.rglob("*.tif")}
    right = {p.relative_to(args.right) for p in args.right.rglob("*.tif")}
    if left != right or not left:
        raise AssertionError(f"Empty or mismatched TIFF paths: {left ^ right}")
    results = []
    for path in sorted(left):
        a = gdal.Open(str(args.left / path))
        b = gdal.Open(str(args.right / path))
        xa, xb = a.ReadAsArray(), b.ReadAsArray()
        data_equal = xa.dtype == xb.dtype and np.array_equal(xa, xb, equal_nan=True)
        metadata_equal = (a.GetGeoTransform() == b.GetGeoTransform()
                          and a.GetProjection() == b.GetProjection()
                          and a.RasterCount == b.RasterCount
                          and a.GetMetadata() == b.GetMetadata())
        if metadata_equal:
            for i in range(1, a.RasterCount + 1):
                ba, bb = a.GetRasterBand(i), b.GetRasterBand(i)
                metadata_equal &= (ba.GetMetadata() == bb.GetMetadata()
                                   and ba.GetNoDataValue() == bb.GetNoDataValue()
                                   and ba.DataType == bb.DataType
                                   and np.array_equal(ba.GetMaskBand().ReadAsArray(), bb.GetMaskBand().ReadAsArray()))
        results.append({"path": str(path), "shape": list(xa.shape),
                        "values_exact_equal": bool(data_equal),
                        "metadata_equal": bool(metadata_equal)})
    passed = all(r["values_exact_equal"] and r["metadata_equal"] for r in results)
    report = {
        "evidence_class": "original_upstream_cpu_repeatability",
        "runs": [str(args.left), str(args.right)],
        "status": "passed_for_listed_tiffs_only" if passed else "failed",
        "results": results,
        "limitations": ["Two same-host runs, not cross-platform tolerance characterization",
                        "ZIP/NPZ repeatability and intermediate state not checked",
                        "No candidate comparison"],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise AssertionError("Reference TIFF repeatability failed; inspect report")
    print(f"{len(results)} TIFFs exactly equal including NaNs and checked metadata")


if __name__ == "__main__":
    main()
