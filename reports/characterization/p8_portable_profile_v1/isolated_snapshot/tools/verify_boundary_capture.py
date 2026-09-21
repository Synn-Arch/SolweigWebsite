"""Verify completeness, provenance hashes, output linkage and carried state."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

STATE = ("firstdaytime", "timeadd", "timestepdec", "Tgmap1", "Tgmap1E",
         "Tgmap1S", "Tgmap1W", "Tgmap1N", "TgOut1")
OUTPUTS = {"Tmrt": "TMRT", "Kdown": "Kdown", "Kup": "Kup", "Ldown": "Ldown",
           "Lup": "Lup", "shadow": "Shadow"}


def verify(run):
    gdal.UseExceptions()
    root = run / "boundaries"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "captured":
        raise AssertionError("Capture did not finish")
    for name, expected in manifest.get("provenance_files", {}).items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected, ("provenance", name)
    inputs, outputs = {}, {}
    counts = Counter()
    dtype_sets = {}
    for event in manifest["events"]:
        path = root / event["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != event["sha256"]:
            raise AssertionError(f"Corrupt capture: {path}")
        counts[event["function"] + "/" + event["boundary"]] += 1
        with np.load(path, allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        expected_fields = {key for key, value in event["fields"].items() if value["kind"] in {"array", "scalar"}}
        assert set(fields) == expected_fields, ("missing/extra array fields", path)
        for key, array in fields.items():
            schema = event["fields"][key]
            assert str(array.dtype) == schema["dtype"], (path, key)
            if schema["kind"] == "array":
                assert list(array.shape) == schema["shape"], (path, key)
        if event["function"] == "Solweig_2022a_calc":
            target = inputs if event["boundary"] == "input" else outputs
            assert event["timestep"] not in target, "Duplicate timestep boundary"
            target[event["timestep"]] = fields
            if event["boundary"] == "output":
                for key, array in fields.items():
                    dtype_sets.setdefault(key, set()).add(str(array.dtype))
    n = np.loadtxt(run / "scene/met.txt", skiprows=1).shape[0]
    assert set(inputs) == set(outputs) == set(range(n)), "Missing chronological boundaries"
    links = []
    for step in range(n):
        for field, filename in OUTPUTS.items():
            ds = gdal.Open(str(run / f"scene/output_folder/0_0/{filename}_0_0.tif"))
            raster = ds.GetRasterBand(step + 1).ReadAsArray()
            assert np.array_equal(outputs[step][field].astype(np.float32), raster, equal_nan=True), (step, field)
        if step:
            for field in STATE:
                assert np.array_equal(inputs[step][field], outputs[step - 1][field], equal_nan=True), (step, field)
            # CI has a separate driver reset at midnight; test the branch explicitly.
            decimal = float(inputs[step]["dectime"])
            if decimal % 1 != 0:
                assert np.array_equal(inputs[step]["CI"], outputs[step - 1]["CI"], equal_nan=True), (step, "CI")
            links.append(step)
    altitudes = [float(inputs[i]["altitude"]) for i in range(n)]
    assert any(a > 0 for a in altitudes) and altitudes[0] < 0 and altitudes[-1] < 0
    return {"status": "passed", "evidence_class": "original_upstream_cpu_capture_verification",
            "run": str(run), "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "events": len(manifest["events"]), "call_counts": dict(counts),
            "timesteps": n, "state_links_checked": links, "state_fields": [*STATE, "CI except midnight reset"],
            "raster_links_checked": list(OUTPUTS),
            "output_dtype_sets": {k: sorted(v) for k, v in dtype_sets.items()},
            "limitations": ["One synthetic day; not restart verification or scientific validation",
                            "No candidate comparison or tolerance calibration implied",
                            "Diagnostic timings include capture overhead"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {result['events']} events, {result['timesteps']} timesteps and {len(result['state_links_checked'])} state handoffs")


if __name__ == "__main__":
    main()
