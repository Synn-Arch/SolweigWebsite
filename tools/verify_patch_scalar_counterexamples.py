"""Reproduce scalar-profile successes using only pinned original CPU inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "0d7fe742abeeddd890dd58fc76ed7f78bd47faec"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/p4_scalar_counterexamples.json")
    args = parser.parse_args()
    import numpy as np
    import torch
    from solweig_gpu import solweig as original

    checkout = ROOT / ".upstream/SOLWEIG-GPU"
    packet = ROOT / "tests/reference/patch_radiation_original_cpu"
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "diff", "--name-only"], text=True).strip()
    if revision != COMMIT or dirty:
        raise RuntimeError("upstream pin/cleanliness failed")
    source_hash = sha(original.__file__)
    if source_hash != sha(checkout / "solweig_gpu/solweig.py") or source_hash != manifest["source_sha256"]:
        raise RuntimeError("original source hash failed")
    if torch.cuda.is_available() or "solweig_light" in sys.modules:
        raise RuntimeError("original-only CPU isolation failed")
    torch.set_num_threads(1)
    report = {
        "evidence_class": "original_upstream_cpu_reference",
        "upstream_commit": COMMIT, "upstream_patch_hash": None,
        "source_sha256": source_hash, "script_sha256": sha(__file__),
        "input_manifest": str(manifest_path.relative_to(ROOT)),
        "input_manifest_sha256": sha(manifest_path), "invocation": sys.argv,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "original_module": original.__file__,
                        "numpy": np.__version__, "torch": torch.__version__,
                        "torch_threads": torch.get_num_threads(), "cuda_available": False},
        "cases": [],
    }

    def load(case):
        if sha(packet / case["input"]) != case["input_sha256"]:
            raise RuntimeError("input packet hash failed")
        values = {}
        with np.load(packet / case["input"]) as data:
            for key, field in case["fields"].items():
                kind = field["kind"]
                if kind == "tensor":
                    values[key] = torch.from_numpy(data[key].copy())
                elif kind == "ndarray":
                    values[key] = data[key].copy()
                elif kind == "numpy_scalar":
                    values[key] = data[key][()]
                elif kind == "none":
                    values[key] = None
                else:
                    values[key] = data[key].item()
        return values

    def execute(case, label, values, overrides, counterpart=False):
        entry = {"label": label, "function": case["function"],
                 "input": case["input"], "input_sha256": case["input_sha256"],
                 "packet_label": case["label"], "overrides": overrides}
        try:
            returned = getattr(original, case["function"])(**values)
            arrays = [value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                      else np.asarray(value) for value in returned]
            entry.update(status="success", outputs=[{
                "field": f"output_{i}", "dtype": str(array.dtype), "shape": list(array.shape),
                "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            } for i, array in enumerate(arrays)])
            if counterpart:
                path = packet / case["output"]
                if sha(path) != case["output_sha256"]:
                    raise RuntimeError("output packet hash failed")
                with np.load(path) as golden:
                    comparisons = []
                    for i, array in enumerate(arrays):
                        expected = golden[f"output_{i}"]
                        exact = bool(np.array_equal(array, expected, equal_nan=True))
                        finite = np.isfinite(array) & np.isfinite(expected)
                        error = float(np.max(np.abs(array[finite].astype(np.float64) - expected[finite].astype(np.float64)))) if finite.any() else 0.0
                        comparisons.append({"field": f"output_{i}", "exact_equal": exact, "max_absolute_error": error})
                entry["captured_counterpart"] = {"output": case["output"], "output_sha256": case["output_sha256"], "comparisons": comparisons}
        except Exception as error:
            entry.update(status="failure", exception_type=type(error).__name__, exception_message=str(error))
        report["cases"].append(entry)

    for name in ("Lcyl_v2022a", "define_patch_characteristics"):
        case = next(c for c in manifest["cases"] if c["function"] == name and c["label"] == "option2-binary-longwave-tensor-solar")
        for altitude in (0.0, -1.0):
            values = load(case)
            values.update(solar_altitude=altitude, solar_azimuth=180.0)
            execute(case, f"{name}-native-night-{altitude}", values,
                    {"solar_altitude": {"kind": "float", "value": altitude}, "solar_azimuth": {"kind": "float", "value": 180.0}})
        values = load(case)
        values.update(solar_altitude=35.0, solar_azimuth=0.0)
        sliced = ["shmat", "vegshmat", "vbshvegshmat"]
        for key in sliced:
            values[key] = values[key][:, :, -1:]
        sliced_geometry = ["sky_patches"] if name == "Lcyl_v2022a" else ["patch_altitude", "patch_azimuth", "steradian", "Lsky_down", "Lsky_side", "Lsky"]
        for key in sliced_geometry:
            values[key] = values[key][-1:]
        execute(case, f"{name}-native-positive-no-active-patch", values,
                {"solar_altitude": {"kind": "float", "value": 35.0}, "solar_azimuth": {"kind": "float", "value": 0.0},
                 "last_axis_last_patch": sliced, "first_axis_last_patch": sliced_geometry})
    for label, tensor_t in (("option2-binary-cylinder", False), ("option2-binary-box-tensor-azimuth", True)):
        case = next(c for c in manifest["cases"] if c["function"] == "Kside_veg_v2022a" and c["label"] == label)
        values = load(case)
        values["azimuth"] = 180.0
        overrides = {"azimuth": {"kind": "float", "value": 180.0}}
        if tensor_t:
            values["t"] = torch.tensor(0.0, dtype=torch.float32)
            overrides["t"] = {"kind": "tensor", "dtype": "float32", "shape": [], "value": 0.0}
        execute(case, f"Kside-native-azimuth-{'tensor-t-box' if tensor_t else 'cylinder'}", values, overrides, counterpart=True)
    report["all_successful"] = all(c["status"] == "success" for c in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report['cases'])} original cases; all_successful={report['all_successful']}; {args.output}")
    if not report["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
