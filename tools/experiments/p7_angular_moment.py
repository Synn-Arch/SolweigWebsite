#!/usr/bin/env python3
"""Falsification harness for the reviewed six-moment longwave experiment."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numba
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from solweig_light.radiation import engine, patch_radiation as candidate  # noqa:E402

PACKET_ROOT = ROOT / "tests/reference/patch_radiation_original_cpu"
DESIGN = ROOT / "reports/characterization/p7_angular_moment_design.json"
OUT = ROOT / "reports/characterization/p7_angular_moment_experiment"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@njit(cache=False, fastmath=False)
def moment_longwave(sh, vs, vb, sun, shade, solid, sine, cosine, directions,
                    gate, solar_gate, sky_down, sky_side, surface_sun,
                    surface_sh, lup, reflection_factor):
    pixels, patches = sh.shape
    output = np.zeros((pixels, 11), dtype=np.float32)
    for pixel in range(pixels):
        accum = np.zeros(14, dtype=np.float32)
        for patch in range(patches):
            sky = sh[pixel, patch] == 1 and vs[pixel, patch] == 1
            veg = vs[pixel, patch] == 0 or vb[pixel, patch] == 0
            building = np.float32(np.float32(1)-sh[pixel, patch])*vb[pixel, patch] == 1
            accum[0] = np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5] = np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side = ((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down = ((surface_sh*solid[patch])*sine[patch])*veg
            accum[6] = np.float32(accum[6]+vegetation_side)
            accum[1] = np.float32(accum[1]+vegetation_down)
            for direction in range(4):
                if gate[patch, direction]:
                    sky_term = np.float32(np.float32(sky*sky_side[patch])*directions[patch, direction])
                    accum[10+direction] = np.float32(accum[10+direction]+sky_term)
                    vegetation_term = vegetation_side*directions[patch, direction]
                    accum[10+direction] = np.float32(accum[10+direction]+vegetation_term)
            if solar_gate[patch]:
                sun_side = ((((surface_sun*sun[pixel, patch])*solid[patch])*cosine[patch])*building)
                shade_side = ((((surface_sh*shade[pixel, patch])*solid[patch])*cosine[patch])*building)
                sun_down = ((((surface_sun*sun[pixel, patch])*solid[patch])*sine[patch])*building)
                shade_down = ((((surface_sh*shade[pixel, patch])*solid[patch])*sine[patch])*building)
                accum[8] = np.float32(accum[8]+sun_side)
                accum[7] = np.float32(accum[7]+shade_side)
                accum[3] = np.float32(accum[3]+sun_down)
                accum[2] = np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch, direction]:
                        accum[10+direction] = np.float32(accum[10+direction]+sun_side*directions[patch, direction])
                        accum[10+direction] = np.float32(accum[10+direction]+shade_side*directions[patch, direction])
            else:
                shade_side = (((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down = (((surface_sh*solid[patch])*sine[patch])*building)
                accum[7] = np.float32(accum[7]+shade_side)
                accum[2] = np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch, direction]:
                        accum[10+direction] = np.float32(accum[10+direction]+shade_side*directions[patch, direction])

        reflected = np.float32(np.float32(np.float32(np.float32(accum[0]+lup[pixel])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        moments = np.zeros(6, dtype=np.float64)
        for patch in range(patches):
            mask = sh[pixel, patch] == 0 or vs[pixel, patch] == 0 or vb[pixel, patch] == 0
            if mask:
                fs = np.float64(solid[patch])
                fc = np.float64(cosine[patch])
                fn = np.float64(sine[patch])
                moments[0] += fs*fc
                moments[1] += fs*fn
                for direction in range(4):
                    if gate[patch, direction]:
                        moments[2+direction] += fs*fc*np.float64(directions[patch, direction])
        accum[9] = np.float32(accum[9]+np.float32(np.float64(reflected)*moments[0]))
        accum[4] = np.float32(accum[4]+np.float32(np.float64(reflected)*moments[1]))
        for direction in range(4):
            accum[10+direction] = np.float32(accum[10+direction]+np.float32(np.float64(reflected)*moments[2+direction]))
        output[pixel, 0] = np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[pixel, 1] = np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[pixel, 2:7] = accum[5:10]
        output[pixel, 7] = accum[10]
        output[pixel, 8] = accum[12]
        output[pixel, 9] = accum[13]
        output[pixel, 10] = accum[11]
    return output


def load_packet(case, manifest):
    path = PACKET_ROOT / case["input"]
    assert sha(path) == case["input_sha256"]
    with np.load(path) as data:
        values = {}
        for key, spec in case["fields"].items():
            if spec["kind"] == "none": values[key] = None
            elif spec["kind"] in ("tensor", "ndarray"): values[key] = data[key].copy()
            elif spec["kind"] == "numpy_scalar": values[key] = data[key][()]
            else: values[key] = data[key].item()
    return values


def compare_tuple(actual, expected_path):
    worst = 0.0
    with np.load(expected_path) as expected:
        for index, value in enumerate(actual):
            oracle = expected[f"output_{index}"]
            for mask in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(mask(value), mask(oracle))
            finite = np.isfinite(oracle)
            if finite.any(): worst = max(worst, float(np.max(np.abs(value[finite]-oracle[finite]))))
            np.testing.assert_allclose(value, oracle, atol=.05, rtol=1e-5, equal_nan=True)
    return worst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    design = json.loads(DESIGN.read_text())
    provenance_drift = []
    critical_prefixes = ("src/", ".upstream/", "tests/", "benchmarks/protocols/", "SOLWEIG_LIGHT_IMPLEMENTATION_PLAN.md")
    for name, digest in design["source_provenance_sha256"].items():
        path = ROOT / name
        if path.exists() and sha(path) != digest:
            if name.startswith(critical_prefixes):
                raise RuntimeError(f"reviewed source changed: {name}")
            provenance_drift.append({"path": name, "reviewed_sha256": digest, "actual_sha256": sha(path)})
    manifest = json.loads((PACKET_ROOT / "manifest.json").read_text())
    selected = [c for c in manifest["cases"] if c["status"] == "captured" and c["function"] in ("Lcyl_v2022a", "define_patch_characteristics")]
    assert len(selected) == 18
    original_parallel, original_serial = candidate._longwave, candidate._longwave_serial
    candidate._longwave = candidate._longwave_serial = moment_longwave
    component = []
    failure = None
    try:
        for case in selected:
            values = load_packet(case, manifest)
            try:
                with np.errstate(all="ignore"):
                    actual = getattr(candidate, case["function"])(**values, parallel=False, block_pixels=7)
                worst = compare_tuple(actual, PACKET_ROOT / case["output"])
                component.append({"label": case["label"], "function": case["function"], "status": "pass", "max_abs": worst})
            except Exception as error:
                failure = {"phase": "component", "label": case["label"], "function": case["function"],
                           "type": type(error).__name__, "message": str(error)}
                component.append({"label": case["label"], "function": case["function"], "status": "fail", "error": failure})
                break

        boundaries = []
        if failure is None:
            helper_path = ROOT / "tests/differential/test_radiation_reference.py"
            spec = importlib.util.spec_from_file_location("p7_angular_reference", helper_path)
            helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
            events = [e for e in helpers.MANIFEST["events"] if e["boundary"] == "input"]
            assert len(events) == 24
            original_engine = engine.Lcyl_v2022a
            engine.Lcyl_v2022a_reference = original_engine
            engine.define_patch_characteristics_reference = engine.define_patch_characteristics
            def call(*args, **kwargs):
                import inspect
                values = inspect.signature(original_engine).bind(*args, **kwargs).arguments
                return candidate.Lcyl_v2022a(**values, parallel=False, block_pixels=17)
            engine.Lcyl_v2022a = call
            try:
                for event in events:
                    expected = next(x for x in helpers.MANIFEST["events"] if x["function"] == "Lcyl_v2022a" and x["timestep"] == event["timestep"])
                    captured = {}
                    def checked(*args, **kwargs):
                        value = call(*args, **kwargs)
                        helpers.compare(expected, value)
                        captured["done"] = True
                        return value
                    engine.Lcyl_v2022a = checked
                    with np.errstate(all="ignore"):
                        engine.Solweig_2022a_calc(**helpers.load_input(event))
                    assert captured.get("done")
                    boundaries.append({"timestep": event["timestep"], "status": "pass"})
                    engine.Lcyl_v2022a = call
            except Exception as error:
                failure = {"phase": "24_step_boundaries", "timestep": event["timestep"],
                           "type": type(error).__name__, "message": str(error)}
                boundaries.append({"timestep": event["timestep"], "status": "fail", "error": failure})
            finally:
                engine.Lcyl_v2022a = original_engine
        else:
            boundaries = []
    finally:
        candidate._longwave, candidate._longwave_serial = original_parallel, original_serial

    result = {
        "schema": "p7_angular_moment_experiment_v1", "design_sha256": sha(DESIGN),
        "experiment_sha256": sha(__file__), "component_cases": component,
        "non_executable_report_drift": provenance_drift,
        "boundary_cases": boundaries, "failure": failure,
        "decision": "reject_before_performance" if failure else "component_gate_passed_pipeline_stage_not_executed",
        "performance_executed": False,
        "rounding_contract": design["rounding_contract"]["candidate_fixed_before_execution"],
        "limits": "Isolated six-moment reflected-longwave experiment only; no source promotion or full-workload claim."
    }
    result_path = OUT / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    report = OUT / "report.md"
    if failure:
        summary = f"Rejected at `{failure['phase']}`: `{failure['type']}: {failure['message']}`. Performance was not run."
    else:
        summary = "The 18 component cases and 24 chronological boundaries passed. Full-pipeline admission remains required before performance."
    report.write_text("# P7 angular-moment experiment\n\n"+summary+"\n\nNo production source was changed. The preserved float32 counterexample means this is tolerance-based evidence, never an algebraic equivalence claim.\n")
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in (result_path, report)}
    (OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"decision": result["decision"], "component": len(component), "boundaries": len(boundaries), "failure": failure}, indent=2))
    return 2 if failure else 0


if __name__ == "__main__": raise SystemExit(main())
