"""Compare isolated original-Torch and candidate ground-view artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

report = {"evidence_class": "original_upstream_cpu_vs_candidate_diagnostic", "cases": []}
passed = True
with np.load(args.upstream) as upstream, np.load(args.candidate) as candidate:
    assert upstream.files == candidate.files
    for landcover in (0, 1):
        fields = []
        for index in range(17):
            key = f"land{landcover}_field{index}"
            left, right = upstream[key], candidate[key]
            finite = np.isfinite(left) & np.isfinite(right)
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            longwave = index in (0, 3, 6, 9, 12)
            tolerance = 0.05 + 1e-5 * np.abs(left) if longwave else np.full(left.shape, 1e-6)
            valid = bool(np.array_equal(np.isnan(left), np.isnan(right)) and
                         np.array_equal(np.isposinf(left), np.isposinf(right)) and
                         np.array_equal(np.isneginf(left), np.isneginf(right)) and
                         np.all(delta[finite] <= tolerance[finite]))
            passed &= valid
            fields.append({"index": index, "name": "gvfSum" if index == 15 else ("gvfNorm" if index == 16 else None),
                           "upstream_dtype": str(left.dtype), "candidate_dtype": str(right.dtype),
                           "max_abs": float(delta[finite].max(initial=0)),
                           "count_over_1e_6": int(np.sum(delta[finite] > 1e-6)), "passed": valid})
        mutation_equal = bool(np.array_equal(upstream[f"land{landcover}_tg_after"], candidate[f"land{landcover}_tg_after"]))
        passed &= mutation_equal
        report["cases"].append({"landcover": landcover, "fields": fields, "tg_mutation_equal": mutation_equal})
report["passed"] = passed
args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
