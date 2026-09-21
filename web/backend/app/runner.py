"""Execute solweig_light on a scene folder and post-process the outputs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config, render

OUTPUT_TILE = "0_0"


def run_model(scene_dir: Path) -> float:
    """Run the pinned own-met workflow on one logical tile; returns elapsed seconds."""
    from solweig_light import RuntimeOptions, runtime_options, thermal_comfort

    started = time.perf_counter()
    options = RuntimeOptions(
        memory_budget_bytes=config.MEMORY_BUDGET_BYTES,
        cpu_budget=config.CPU_THREADS,
        workers=1,
        threads_per_worker=config.CPU_THREADS,
        block_pixels=config.BLOCK_PIXELS,
    )
    from osgeo import gdal

    ds = gdal.Open(str(scene_dir / "Building_DSM.tif"))
    tile_size = max(ds.RasterXSize, ds.RasterYSize)
    ds = None
    with runtime_options(options):
        thermal_comfort(
            base_path=str(scene_dir),
            selected_date_str=config.SIM_DATE,
            landcover_filename=config.LANDCOVER_FILE,
            own_met_file=str(scene_dir / config.MET_FILE),
            ERA_5_z0_find=False,
            use_uhi=False,
            tile_size=tile_size,
            overlap=0,
            save_tmrt=True,
        )
    return time.perf_counter() - started


def output_path(scene_dir: Path, var: str) -> Path:
    return scene_dir / "output_folder" / OUTPUT_TILE / f"{var.upper()}_{OUTPUT_TILE}.tif"


CHANGE_THRESHOLD_C = 0.5  # |scenario - baseline| above this counts as a changed pixel


def postprocess(scene_dir: Path, result_dir: Path, baseline_dir: Path | None, *, keep_arrays: bool) -> dict:
    """Render PNGs (and differences against a baseline) and write result.json.

    ``keep_arrays`` stores the raw bands as .npy so later runs can diff against them.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"variables": {}, "hours": None}
    for var in render.VARIABLES:
        data, times = render.read_bands(output_path(scene_dir, var))
        summary["hours"] = times
        if keep_arrays:
            np.save(result_dir / f"{var}.npy", data)
        render.render_variable(data, var, result_dir)
        entry = {"stats": render.hourly_stats(data)}
        if baseline_dir is not None:
            base = np.load(baseline_dir / f"{var}.npy")
            render.render_difference(data, base, var, result_dir)
            diff = data - base
            entry["baseline_stats"] = render.hourly_stats(base)
            entry["diff_stats"] = render.hourly_stats(diff)
            entry["changed_pixels"] = [int(np.sum(np.abs(diff[..., h]) > CHANGE_THRESHOLD_C)) for h in range(diff.shape[-1])]
        summary["variables"][var] = entry
    render.write_json(result_dir / "result.json", summary)
    return summary


def committed_timesteps(scene_dir: Path) -> int:
    """Progress hint: number of timesteps committed by the output transaction."""
    best = 0
    for record in (scene_dir / ".solweig-light" / "transactions").glob("*/transaction.json"):
        try:
            hashes = json.loads(record.read_text())["checkpoint"]["band_hashes"]
            best = max(best, max((len(v) for v in hashes.values()), default=0))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return best


def geometry_done(scene_dir: Path) -> bool:
    return (scene_dir / "processed_inputs" / "SVF" / f"shadowmats_{OUTPUT_TILE}.npz").exists()
