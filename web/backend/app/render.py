"""Turn 24-band output rasters into hourly PNG overlays plus summary stats."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image

gdal.UseExceptions()

# Fixed colour ranges so baseline and scenario overlays are directly comparable.
VARIABLES = {
    "tmrt": {"label": "Mean radiant temperature (°C)", "vmin": 20.0, "vmax": 75.0},
    "utci": {"label": "UTCI (°C)", "vmin": 10.0, "vmax": 50.0},
}
DIFF_RANGE = 15.0  # ± °C for scenario minus baseline

# Sequential ramp (cool blue -> warm red), 5 stops.
_SEQ = np.array([[59, 76, 192], [131, 178, 251], [221, 220, 219], [240, 140, 97], [180, 4, 38]], float)
# Diverging ramp for differences: green (cooler) -> white -> purple (warmer).
_DIV = np.array([[0, 109, 44], [161, 217, 155], [247, 247, 247], [194, 165, 207], [118, 42, 131]], float)


def _ramp(stops: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def _write_png(rgb: np.ndarray, valid: np.ndarray, path: Path) -> None:
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path, optimize=False, compress_level=6)


def read_bands(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return (rows, cols, bands) float32 array and per-band Time metadata."""
    ds = gdal.Open(str(path))
    try:
        data = ds.ReadAsArray().astype(np.float32)  # (bands, rows, cols)
        times = [ds.GetRasterBand(i + 1).GetMetadataItem("Time") or f"band {i + 1}" for i in range(ds.RasterCount)]
        return np.moveaxis(data, 0, -1), times
    finally:
        ds = None


def render_variable(data: np.ndarray, var: str, out_dir: Path) -> None:
    spec = VARIABLES[var]
    out_dir.mkdir(parents=True, exist_ok=True)
    for h in range(data.shape[-1]):
        band = data[..., h]
        valid = np.isfinite(band)
        t = (band - spec["vmin"]) / (spec["vmax"] - spec["vmin"])
        _write_png(_ramp(_SEQ, np.nan_to_num(t)), valid, out_dir / f"{var}_{h:02d}.png")


def render_difference(scenario: np.ndarray, baseline: np.ndarray, var: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    diff = scenario - baseline
    for h in range(diff.shape[-1]):
        band = diff[..., h]
        valid = np.isfinite(band)
        t = (band + DIFF_RANGE) / (2 * DIFF_RANGE)
        _write_png(_ramp(_DIV, np.nan_to_num(t, nan=0.5)), valid, out_dir / f"{var}_diff_{h:02d}.png")


def hourly_stats(data: np.ndarray) -> dict:
    """Scene mean/min/max per hour, ignoring NaN."""
    return {
        "mean": [float(np.nanmean(data[..., h])) for h in range(data.shape[-1])],
        "min": [float(np.nanmin(data[..., h])) for h in range(data.shape[-1])],
        "max": [float(np.nanmax(data[..., h])) for h in range(data.shape[-1])],
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1))
