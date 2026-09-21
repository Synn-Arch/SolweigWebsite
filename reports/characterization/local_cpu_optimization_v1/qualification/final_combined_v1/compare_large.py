#!/usr/bin/env python3
"""Bind final large outputs to E0 candidates and frozen original gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from osgeo import gdal

ROOT = Path("/Users/alansynn/Workspace/solweig-light")
sys.path.insert(0, str(ROOT / "tools"))
from run_exact_cpu_pairs import compare_scenes  # noqa: E402

FIELDS = ("Kdown", "Kup", "Ldown", "Lup", "Shadow", "TMRT", "Ta", "UTCI", "WBGT", "Wind")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nodata_equal(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))


def compare_tolerant(candidate: Path, reference: Path, rtol: float, atol: float) -> dict:
    a, b = gdal.Open(str(candidate)), gdal.Open(str(reference))
    signature_a = (a.RasterXSize, a.RasterYSize, a.RasterCount, a.GetGeoTransform(),
                  a.GetProjection(), a.GetMetadata())
    signature_b = (b.RasterXSize, b.RasterYSize, b.RasterCount, b.GetGeoTransform(),
                  b.GetProjection(), b.GetMetadata())
    schema = signature_a == signature_b
    bands = []
    passed = schema
    if schema:
        for index in range(1, a.RasterCount + 1):
            x, y = a.GetRasterBand(index), b.GetRasterBand(index)
            band_schema = (x.DataType == y.DataType and x.GetMetadata() == y.GetMetadata()
                           and x.GetDescription() == y.GetDescription()
                           and nodata_equal(x.GetNoDataValue(), y.GetNoDataValue())
                           and x.GetMaskFlags() == y.GetMaskFlags())
            max_abs = 0.0
            values_passed = band_schema
            for row in range(0, a.RasterYSize, 256):
                for col in range(0, a.RasterXSize, 256):
                    width, height = min(256, a.RasterXSize - col), min(256, a.RasterYSize - row)
                    left = x.ReadAsArray(col, row, width, height)
                    right = y.ReadAsArray(col, row, width, height)
                    left_mask = x.GetMaskBand().ReadAsArray(col, row, width, height)
                    right_mask = y.GetMaskBand().ReadAsArray(col, row, width, height)
                    masks_equal = np.array_equal(left_mask, right_mask)
                    finite = np.isfinite(left) & np.isfinite(right)
                    if np.any(finite):
                        max_abs = max(max_abs, float(np.max(np.abs(left[finite] - right[finite]))))
                    special_equal = (np.array_equal(np.isnan(left), np.isnan(right))
                                     and np.array_equal(np.isposinf(left), np.isposinf(right))
                                     and np.array_equal(np.isneginf(left), np.isneginf(right)))
                    values = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
                    values_passed &= masks_equal and special_equal and values
            record_passed = band_schema and values_passed
            bands.append({"band": index, "metadata_equal": band_schema,
                          "masks_and_values_passed": values_passed,
                          "max_abs": max_abs, "passed": record_passed})
            passed &= record_passed
    a = b = None
    return {"candidate": str(candidate), "reference": str(reference), "rtol": rtol,
            "atol": atol, "schema_equal": schema, "bands": bands, "passed": bool(passed)}


def main() -> int:
    root = ROOT / "reports/characterization/local_cpu_optimization_v1/qualification/final_combined_v1"
    records = {}
    all_passed = True
    for case in ("dense1024", "vegetation1024"):
        candidate_scene = root / "large_harness/large_runs_v1" / case / "scene"
        e0_scene = ROOT / "reports/characterization/local_cpu_optimization_v1/e0/admission" / case
        oracle_scene = ROOT / "reports/characterization/p8_portable_profile_v1/oracle/large" / case / "run/scene"
        exact = compare_scenes(candidate_scene, e0_scene, [1024, 1024], timesteps=24)
        reference = {}
        for field in FIELDS:
            name = f"{field}_0_0.tif"
            old = json.loads((e0_scene / "comparison_postverify.json").read_text())
            policy = next(item for item in old["fields"] if item["field"] == field)
            reference[field] = compare_tolerant(candidate_scene / "output_folder/0_0" / name,
                                                 oracle_scene / "output_folder/0_0" / name,
                                                 policy["rtol"], policy["atol"])
        old = json.loads((e0_scene / "comparison_postverify.json").read_text())
        candidate_hashes = {item["field"]: item["sha256"] for item in old["candidate_output_manifest"]}
        candidate_hashes_actual = {field: digest(candidate_scene / "output_folder/0_0" / f"{field}_0_0.tif") for field in FIELDS}
        original_fields_passed = all(item["passed"] for item in reference.values())
        output_hashes_match_e0 = candidate_hashes_actual == candidate_hashes
        geometry_original_gate = bool(old["geometry"]["passed"])
        case_record = {
            "case": case,
            "candidate_vs_e0_exact": exact,
            "original_reference_fields": reference,
            "original_reference_fields_passed": original_fields_passed,
            "candidate_output_hashes_match_e0": output_hashes_match_e0,
            "candidate_output_hashes": candidate_hashes_actual,
            "e0_candidate_output_hashes": candidate_hashes,
            "original_reference_geometry_18_passed_in_e0_record": geometry_original_gate,
            "geometry_18_evidence_binding": old["geometry"],
            "original_reference_verification_sha256": digest(e0_scene / "reference_verification.json"),
            "e0_comparison_postverify_sha256": digest(e0_scene / "comparison_postverify.json"),
            "passed": bool(exact["passed"] and original_fields_passed and output_hashes_match_e0
                           and geometry_original_gate),
        }
        records[case] = case_record
        all_passed &= case_record["passed"]
    output = {"schema": "local-cpu-optimization-final-combined-large-comparison.v1",
              "performance_claim": False, "cases": records, "passed": bool(all_passed),
              "limitation": "No every-timestep large trace was captured; small 72-event trace evidence remains the trace gate."}
    path = root / "large_comparison_v1.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": all_passed, "path": str(path)}))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
