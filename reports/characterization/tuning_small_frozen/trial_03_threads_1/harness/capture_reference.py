"""Read-only profiling of original upstream call boundaries, for small oracles.

This diagnostic collector changes timing and is never performance evidence.
Arrays are copied to non-pickled NPZ files synchronously at each boundary.
No source, arguments, returned values, or numerical operation is replaced.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

FUNCTIONS = {
    "Solweig_2022a_calc", "gvf_2018a", "TsWaveDelay_2015a",
    "Kside_veg_v2022a", "Lcyl_v2022a", "define_patch_characteristics",
    "shadowingfunction_wallheight_13", "shadowingfunction_wallheight_23",
}


class BoundaryCapture:
    def __init__(self, module, destination):
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=False)
        self.codes = {getattr(module, name).__code__: name for name in FUNCTIONS}
        self.returns = {}
        tree = ast.parse(Path(module.__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                value = max(returns, key=lambda n: n.lineno).value
                self.returns[node.name] = [ast.unparse(n) for n in value.elts] if isinstance(value, ast.Tuple) else [ast.unparse(value)]
        self.events = []
        self.timestep = None
        self.previous_profile = None

    def save(self, function, boundary, values):
        arrays = {}
        descriptions = {}

        def encode(value, key):
            # Torch is already loaded by the oracle. Conversion only reads data.
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise TypeError(f"Object data is not an admissible reference: {key}")
                arrays[key] = np.array(value, copy=True)
                descriptions[key] = {"kind": "array", "dtype": str(value.dtype), "shape": list(value.shape)}
            elif isinstance(value, dict):
                descriptions[key] = {"kind": "dict", "keys": list(value)}
                for child, item in value.items():
                    encode(item, f"{key}/{child}")
            elif isinstance(value, (tuple, list)):
                descriptions[key] = {"kind": type(value).__name__, "length": len(value)}
                for index, item in enumerate(value):
                    encode(item, f"{key}/{index}")
            elif value is None:
                descriptions[key] = {"kind": "none"}
            elif isinstance(value, (int, float, bool, np.number)):
                arrays[key] = np.asarray(value)
                descriptions[key] = {"kind": "scalar", "dtype": str(arrays[key].dtype)}
            else:
                raise TypeError(f"Unsupported reference value {key}: {type(value)}")

        for name, value in values.items():
            encode(value, name)
        name = f"{len(self.events):05d}_{function}_{boundary}.npz"
        path = self.destination / name
        np.savez_compressed(path, **arrays)
        self.events.append({"function": function, "boundary": boundary,
                            "timestep": self.timestep, "path": name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "fields": descriptions})

    def profile(self, frame, event, arg):
        function = self.codes.get(frame.f_code)
        if function is None:
            return
        if event == "call" and function == "Solweig_2022a_calc":
            self.timestep = int(frame.f_locals["i"])
            self.save(function, "input", dict(frame.f_locals))
        elif event == "return":
            # A Python frame unwinding through an exception reports return None.
            # Let the original exception propagate to the oracle failure log.
            if arg is None:
                return
            names = self.returns[function]
            values = arg if isinstance(arg, tuple) else (arg,)
            if len(names) != len(values):
                raise ValueError(f"Return schema changed in {function}")
            self.save(function, "output", dict(zip(names, values)))

    def __enter__(self):
        self.previous_profile = sys.getprofile()
        if self.previous_profile is not None:
            raise RuntimeError("Refusing to replace an existing profiler")
        sys.setprofile(self.profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(self.previous_profile)
        report = {"evidence_class": "original_upstream_cpu_instrumented",
                  "instrumentation": "read-only Python call/return profiler; timings invalid for benchmarking",
                  "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "status": "captured" if exc_type is None else "failed",
                  "events": self.events}
        report["provenance_files"] = {
            name: hashlib.sha256((self.destination.parent / name).read_bytes()).hexdigest()
            for name in ("environment.json", "fixture_hashes.json", "kwargs.json")
        }
        (self.destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
