"""Compute the baseline (no added trees) once; also warms the Numba cache.

Run at Docker build time or before the first local start:
    python -m app.precompute
"""
from __future__ import annotations

import shutil
import sys
import time

from . import config, render, runner, scene


def main(force: bool = False) -> int:
    marker = config.BASELINE_DIR / "result.json"
    if marker.exists() and not force:
        print(f"[baseline] already present at {config.BASELINE_DIR}")
        return 0
    work = config.DATA_DIR / "baseline_scene"
    if work.exists():
        shutil.rmtree(work)
    scene.copy_inputs(config.SCENE_DIR, work)
    print(f"[baseline] running SOLWEIG on {work} ...")
    elapsed = runner.run_model(work)
    print(f"[baseline] model finished in {elapsed:.0f}s; rendering")
    if config.BASELINE_DIR.exists():
        shutil.rmtree(config.BASELINE_DIR)
    runner.postprocess(work, config.BASELINE_DIR, None, keep_arrays=True)
    render.write_json(config.BASELINE_DIR / "meta.json", {"model_seconds": elapsed, "computed": time.time(),
                                                         "date": config.SIM_DATE, "threads": config.CPU_THREADS})
    shutil.rmtree(work, ignore_errors=True)
    print(f"[baseline] done -> {config.BASELINE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
