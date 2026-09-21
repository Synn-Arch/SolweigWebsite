#!/usr/bin/env python3
"""Isolated exact-coefficient UTCI Horner experiment (no production imports)."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COEFFICIENTS = ROOT / "src/solweig_light/comfort/utci_coefficients.json"
UTCI_SOURCE = ROOT / "src/solweig_light/comfort/utci.py"
REFERENCE = ROOT / "tests/reference/utci_original_cpu"
GATE_C = 0.02


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_explicit():
    # Bind the checked-in evaluator itself through its real package context.
    sys.path.insert(0, str(ROOT / "src"))
    module = importlib.import_module("solweig_light.comfort.utci")
    return module.utci_polynomial


def coefficient_tensor(document: dict) -> np.ndarray:
    tensor = np.zeros((7, 7, 7, 7), dtype=np.float32)  # D, va, Ta, Pa
    occupied: set[tuple[int, int, int, int]] = set()
    for term in document["ordered_terms"]:
        p = term["powers"]
        key = (p["D_Tmrt"], p["va"], p["Ta"], p["Pa"])
        occupied.add(key)
        tensor[key] = np.float32(tensor[key] + np.float32(term["coefficient"]))
    if len(occupied) != 210:
        raise RuntimeError(f"expected 210 unique exponent tuples from 211 ordered terms, got {len(occupied)}")
    return tensor


def horner(x: np.ndarray | np.float32, coefficients: list[np.ndarray | np.float32]):
    result = np.zeros_like(x, dtype=np.float32) if isinstance(x, np.ndarray) else np.float32(0)
    for coefficient in reversed(coefficients):
        result = np.asarray(result * x + coefficient, dtype=np.float32)
    return result


def evaluate_horner(tensor: np.ndarray, d, ta, va, pa) -> np.ndarray:
    d, ta, va, pa = np.broadcast_arrays(*(np.asarray(x, np.float32) for x in (d, ta, va, pa)))
    by_d = []
    for di in range(7):
        by_va = []
        for vi in range(7):
            by_ta = [horner(pa, [tensor[di, vi, ti, pi] for pi in range(7)]) for ti in range(7)]
            by_va.append(horner(ta, by_ta))
        by_d.append(horner(va, by_va))
    return horner(d, by_d)


def specialize_uniform(tensor: np.ndarray, ta: np.float32, pa: np.float32) -> np.ndarray:
    table = np.empty((7, 7), np.float32)
    for di in range(7):
        for vi in range(7):
            inner = [horner(pa, [tensor[di, vi, ti, pi] for pi in range(7)]) for ti in range(7)]
            table[di, vi] = horner(ta, inner)
    return table


def evaluate_specialized(table: np.ndarray, d, va) -> np.ndarray:
    d, va = np.broadcast_arrays(np.asarray(d, np.float32), np.asarray(va, np.float32))
    return horner(d, [horner(va, [table[di, vi] for vi in range(7)]) for di in range(7)])


def error(actual, expected) -> dict:
    delta = np.asarray(actual, np.float64) - np.asarray(expected, np.float64)
    index = int(np.argmax(np.abs(delta)))
    return {
        "count": int(delta.size),
        "max_abs_c": float(np.max(np.abs(delta))),
        "rms_c": float(np.sqrt(np.mean(delta * delta))),
        "worst_flat_index": index,
        "actual_at_worst": float(np.ravel(actual)[index]),
        "expected_at_worst": float(np.ravel(expected)[index]),
    }


def cancellation_ratio(document: dict, d, ta, va, pa) -> np.ndarray:
    total = np.zeros_like(d, dtype=np.float64)
    magnitude = np.zeros_like(d, dtype=np.float64)
    values = {"D_Tmrt": d.astype(np.float64), "Ta": ta.astype(np.float64),
              "va": va.astype(np.float64), "Pa": pa.astype(np.float64)}
    for term in document["ordered_terms"]:
        value = np.full_like(total, float(term["coefficient"]))
        for name, power in term["powers"].items():
            if power:
                value *= values[name] ** power
        total += value
        magnitude += np.abs(value)
    return magnitude / np.maximum(np.abs(total), np.finfo(np.float64).tiny)


def trials(callable_, count=7) -> list[int]:
    result = []
    for _ in range(count):
        started = time.perf_counter_ns()
        callable_()
        result.append(time.perf_counter_ns() - started)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/characterization/p7_utci_horner_experiment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    document = json.loads(COEFFICIENTS.read_text())
    manifest = json.loads((REFERENCE / "manifest.json").read_text())
    if sha256(REFERENCE / "cases.npz") != manifest["fixture_sha256"]:
        raise RuntimeError("frozen original-CPU fixture hash mismatch")
    if document["source_sha256"] != manifest["source_package_sha256"]["calculate_utci.py"]:
        raise RuntimeError("coefficient inventory is not bound to frozen upstream source")
    if not document["reconstruction_ast_equal"] or document["term_count"] != 211:
        raise RuntimeError("coefficient inventory reconstruction is not exact")
    tensor = coefficient_tensor(document)
    explicit = load_explicit()

    with np.load(REFERENCE / "cases.npz", allow_pickle=False) as loaded:
        cases = {key: loaded[key] for key in loaded.files}
    upstream_accepted = ((cases["Ta"] >= -50) & (cases["Ta"] <= 50) & (cases["RH"] >= 0) &
                         (cases["RH"] <= 100) & (cases["Tmrt"] >= -50) & (cases["Tmrt"] <= 100) &
                         (cases["wind"] >= 0) & (cases["wind"] <= 17))
    d = np.asarray(cases["Tmrt"][upstream_accepted] - cases["Ta"][upstream_accepted], np.float32)
    ta, va, pa = (np.asarray(cases[k][upstream_accepted], np.float32) for k in ("Ta", "wind", "Pa"))
    original = cases["polynomial"][upstream_accepted]
    with np.errstate(all="ignore"):
        actual_explicit = np.asarray(explicit(d, ta, va, pa), np.float32)
        actual_horner = evaluate_horner(tensor, d, ta, va, pa)

    ratios = cancellation_ratio(document, d, ta, va, pa)
    selected = np.argsort(ratios, kind="stable")[-min(1024, ratios.size):]
    # One-ULP neighbours around high-cancellation, upstream-accepted frozen inputs
    # exercise numerical boundaries without inventing an independent golden. This
    # filter is executable upstream behavior, not a published UTCI applicability claim.
    directions = [np.float32(-np.inf), np.float32(np.inf)]
    sensitive_parts = []
    for axis in range(4):
        base = [d[selected], ta[selected], va[selected], pa[selected]]
        for direction in directions:
            changed = list(base)
            changed[axis] = np.nextafter(changed[axis], direction, dtype=np.float32)
            sensitive_parts.append(changed)
    sensitive = [np.concatenate([part[i] for part in sensitive_parts]) for i in range(4)]
    sd, sta, sva, spa = sensitive
    # Keep perturbations within the upstream-accepted ranges inherited from their source
    # cases (D follows Tmrt-Ta, Pa stays nonnegative). This is not a scientific-domain gate.
    keep = ((sta >= -50) & (sta <= 50) & (sva >= 0) & (sva <= 17) & (spa >= 0) &
            ((sd + sta) >= -50) & ((sd + sta) <= 100))
    sd, sta, sva, spa = (x[keep] for x in sensitive)
    with np.errstate(all="ignore"):
        sensitive_explicit = np.asarray(explicit(sd, sta, sva, spa), np.float32)
        sensitive_horner = evaluate_horner(tensor, sd, sta, sva, spa)

    uniform_ta, uniform_pa = np.float32(25), np.float32(1.584)
    ud = np.linspace(-75, 75, 32768, dtype=np.float32)
    uva = np.linspace(0, 17, 32768, dtype=np.float32)
    construction_trials = trials(lambda: specialize_uniform(tensor, uniform_ta, uniform_pa))
    table = specialize_uniform(tensor, uniform_ta, uniform_pa)
    uniform_generic = evaluate_horner(tensor, ud, uniform_ta, uva, uniform_pa)
    uniform_specialized = evaluate_specialized(table, ud, uva)
    uniform_bits_equal = bool(np.array_equal(uniform_generic.view(np.uint32), uniform_specialized.view(np.uint32)))

    timing_inputs = (d, ta, va, pa)
    runtime = {
        "explicit_ns": trials(lambda: explicit(*timing_inputs)),
        "horner_ns": trials(lambda: evaluate_horner(tensor, *timing_inputs)),
        "uniform_generic_ns": trials(lambda: evaluate_horner(tensor, ud, uniform_ta, uva, uniform_pa)),
        "uniform_specialized_ns": trials(lambda: evaluate_specialized(table, ud, uva)),
        "specialization_construction_ns": construction_trials,
    }
    runtime["medians_ns"] = {key: int(statistics.median(value)) for key, value in runtime.items() if key.endswith("_ns")}

    comparisons = {
        "explicit_vs_original_cpu": error(actual_explicit, original),
        "horner_vs_original_cpu": error(actual_horner, original),
        "horner_vs_actual_explicit": error(actual_horner, actual_explicit),
        "sensitive_horner_vs_actual_explicit": error(sensitive_horner, sensitive_explicit),
    }
    accepted = (comparisons["horner_vs_original_cpu"]["max_abs_c"] <= GATE_C and
                comparisons["sensitive_horner_vs_actual_explicit"]["max_abs_c"] <= GATE_C and
                uniform_bits_equal)
    result = {
        "schema": "p7_utci_horner_experiment_v1",
        "decision": "admit_component_followup" if accepted else "reject_horner_evaluator",
        "scope": "UTCI polynomial component diagnostic; no end-to-end or workload speed claim",
        "invocation": "uv run python tools/experiments/p7_utci_horner.py",
        "gate_c_unchanged": GATE_C,
        "full_inventory": {"term_count": 211, "unique_exponent_tuples": 210,
                           "dense_shape": list(tensor.shape), "nonzero": int(np.count_nonzero(tensor)),
                           "nesting_inner_to_outer": ["Pa", "Ta", "va", "D_Tmrt"]},
        "reference_binding": {
            "evidence_class": manifest["evidence_class"], "source_commit": manifest["source_commit"],
            "fixture": str((REFERENCE / "cases.npz").relative_to(ROOT)), "fixture_sha256": manifest["fixture_sha256"],
            "utci_source": str(UTCI_SOURCE.relative_to(ROOT)), "utci_source_sha256": sha256(UTCI_SOURCE),
            "coefficient_inventory_sha256": sha256(COEFFICIENTS),
            "experiment_source_sha256": sha256(Path(__file__)),
            "coefficient_source_sha256": document["source_sha256"],
            "reconstruction_ast_equal": document["reconstruction_ast_equal"],
        },
        "fixture": {"frozen_upstream_accepted_count": int(upstream_accepted.sum()),
                    "range_semantics": "executable upstream acceptance filter; not published UTCI applicability domain",
                    "cancellation_seed_count": int(selected.size),
                    "neighbour_count": int(sd.size), "max_cancellation_ratio": float(ratios[selected[-1]])},
        "comparisons": comparisons,
        "uniform_specialization": {"ta": float(uniform_ta), "pa": float(uniform_pa), "count": int(ud.size),
                                   "bitwise_equal_to_generic_horner": uniform_bits_equal},
        "storage_bytes": {"dense_float32_tensor": int(tensor.nbytes), "specialized_float32_table": int(table.nbytes),
                          "sparse_coefficients_only": int(211 * np.dtype(np.float32).itemsize)},
        "runtime_diagnostics": runtime,
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = [
        "# P7 exact-coefficient UTCI Horner experiment", "",
        f"Decision: **{result['decision']}**. The frozen numerical gate remained `{GATE_C} °C`.", "",
        "The experiment loaded all 211 mechanically extracted coefficient/exponent terms, called the checked-in explicit evaluator, and compared against the hash-verified original-upstream CPU fixture. It is a polynomial-component diagnostic, not an end-to-end performance claim.", "",
        f"Frozen upstream-accepted cases: {int(upstream_accepted.sum())}; numerical cancellation-neighbour cases: {int(sd.size)}. These labels describe executable source-test filtering, not the published UTCI applicability domain.",
        f"Horner vs original CPU max/RMS: {comparisons['horner_vs_original_cpu']['max_abs_c']:.9g} / {comparisons['horner_vs_original_cpu']['rms_c']:.9g} °C.",
        f"Horner vs explicit on cancellation neighbours max/RMS: {comparisons['sensitive_horner_vs_actual_explicit']['max_abs_c']:.9g} / {comparisons['sensitive_horner_vs_actual_explicit']['rms_c']:.9g} °C.",
        f"Uniform Ta/Pa specialization was bitwise equal to generic Horner: `{uniform_bits_equal}`.", "",
        "Raw construction/runtime trials and provenance are in `result.json`. Timings are diagnostics on this host and do not establish a full-workload benefit.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    manifest_out = {"files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                               for p in (result_path, args.output / "report.md")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "comparisons": comparisons,
                      "uniform_bits_equal": uniform_bits_equal}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
