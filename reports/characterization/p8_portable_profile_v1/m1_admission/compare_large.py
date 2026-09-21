"""Streaming comparison of one installed-candidate case to original M1 SLEEF."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

tolerances = {
    "Kdown": (0.05, 1e-5), "Kup": (0.05, 1e-5),
    "Ldown": (0.05, 1e-5), "Lup": (0.05, 1e-5),
    "Shadow": (0.0, 0.0), "TMRT": (0.01, 0.0),
    "Ta": (0.0, 0.0), "UTCI": (0.02, 0.0),
    "WBGT": (0.02, 0.0), "Wind": (0.0, 0.0),
}
report = {
    "schema": "portable-profile-m1-large-comparison.v1",
    "case": a.case,
    "reference": "original_upstream_M1_SLEEF",
    "candidate": "installed_wheel_macos_ARM64",
    "fields": [],
    "performance_claim": None,
}
all_passed = True
for field, (atol, rtol) in tolerances.items():
    rp = a.reference / f"{field}_0_0.tif"
    cp = a.candidate / f"{field}_0_0.tif"
    field_result = {"field": field, "atol": atol, "rtol": rtol, "bands": []}
    with rasterio.open(rp) as ref, rasterio.open(cp) as cand:
        schema_equal = (
            ref.profile == cand.profile
            and ref.tags() == cand.tags()
            and ref.descriptions == cand.descriptions
            and ref.units == cand.units
            and ref.scales == cand.scales
            and ref.offsets == cand.offsets
        )
        field_result["schema_equal"] = schema_equal
        for band in range(1, ref.count + 1):
            x, y = ref.read(band), cand.read(band)
            finite_equal = np.array_equal(np.isfinite(x), np.isfinite(y))
            nan_equal = np.array_equal(np.isnan(x), np.isnan(y))
            posinf_equal = np.array_equal(np.isposinf(x), np.isposinf(y))
            neginf_equal = np.array_equal(np.isneginf(x), np.isneginf(y))
            masks_equal = finite_equal and nan_equal and posinf_equal and neginf_equal
            finite = np.isfinite(x) & np.isfinite(y)
            max_abs = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
            values_equal = bool(np.allclose(x[finite], y[finite], atol=atol, rtol=rtol))
            metadata_equal = ref.tags(band) == cand.tags(band)
            passed = schema_equal and masks_equal and values_equal and metadata_equal
            field_result["bands"].append({
                "band": band, "max_abs": max_abs, "masks_equal": masks_equal,
                "metadata_equal": metadata_equal, "passed": passed,
            })
            all_passed &= passed
    report["fields"].append(field_result)
report["status"] = "passed" if all_passed else "failed"
a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not all_passed:
    raise SystemExit(1)
