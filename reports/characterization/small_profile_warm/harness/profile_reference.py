"""Read-only stage timing for diagnostic upstream runs (no array capture)."""
import hashlib
import json
from pathlib import Path
import sys
import time


class StageProfile:
    def __init__(self, package, destination):
        self.package = Path(package).resolve()
        self.destination = Path(destination)
        self.stack = []
        self.rows = {}
        self.selected = {
            "solweig_gpu.py": {"thermal_comfort", "preprocess", "run_walls_aspect", "calculate_svf", "run_utci_tiles"},
            "utci_process.py": {"compute_utci", "load_cached_svf_outputs"},
            "shadow.py": {"svf_calculator", "save_svf_zip_npz_outputs"},
            "solweig.py": {"Solweig_2022a_calc", "gvf_2018a", "Kside_veg_v2022a", "Lcyl_v2022a",
                           "shadowingfunction_wallheight_23", "TsWaveDelay_2015a"},
            "calculate_utci.py": {"utci_calculator"},
            "calculate_wbgt.py": {"isobaric_wet_bulb_temperature_from_rh", "black_globe_temperature"},
        }
        self.codes = {}

    def profile(self, frame, event, arg):
        if event not in {"call", "return"}:
            return
        code = frame.f_code
        if code not in self.codes:
            path = Path(code.co_filename)
            self.codes[code] = (path.name + ":" + code.co_name
                if code.co_name in self.selected.get(path.name, set()) and path.parent.resolve() == self.package else None)
        key = self.codes[code]
        if key is None:
            return
        if event == "call":
            self.stack.append([id(frame), key, time.perf_counter(), 0.0])
        elif self.stack:
            frame_id, key, start, nested = self.stack.pop()
            if frame_id != id(frame):
                raise RuntimeError("Unbalanced stage profiling stack")
            elapsed = time.perf_counter() - start
            row = self.rows.setdefault(key, {"calls": 0, "inclusive_seconds": 0.0, "excluding_selected_children_seconds": 0.0})
            row["calls"] += 1
            row["inclusive_seconds"] += elapsed
            row["excluding_selected_children_seconds"] += elapsed - nested
            if self.stack:
                self.stack[-1][3] += elapsed

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError("Refusing to overwrite another profiler")
        self.start = time.perf_counter()
        sys.setprofile(self.profile)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(None)
        elapsed = time.perf_counter() - self.start
        report = {"status": "profiled" if exc_type is None else "failed", "stages": self.rows,
                  "elapsed_profiled_region_seconds": elapsed,
                  "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "measurement_class": "diagnostic read-only call profiler; includes profiler overhead",
                  "limitations": ["Do not sum overlapping inclusive stage durations",
                                  "Only selected boundaries; exclusive time includes unselected helpers and I/O",
                                  "Not a paired speedup benchmark; child workers are included as wait time, not separately traced",
                                  "Use separate non-profiled fresh-process timings for performance comparisons"]}
        self.destination.write_text(json.dumps(report, indent=2) + "\n")
