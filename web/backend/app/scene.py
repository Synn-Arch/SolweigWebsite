"""Scene georeferencing and tree painting on the canopy-height raster."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

from . import config

gdal.UseExceptions()
osr.UseExceptions()

INPUT_FILES = ("Building_DSM.tif", "DEM.tif", "Trees.tif", config.LANDCOVER_FILE, config.MET_FILE)


@dataclass(frozen=True)
class SceneGeometry:
    rows: int
    cols: int
    geotransform: tuple[float, ...]
    wkt: str

    @property
    def pixel_size(self) -> float:
        return abs(self.geotransform[1])

    def _transformer(self, to_wgs84: bool) -> osr.CoordinateTransformation:
        native = osr.SpatialReference()
        native.ImportFromWkt(self.wkt)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        native.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return osr.CoordinateTransformation(native, wgs84) if to_wgs84 else osr.CoordinateTransformation(wgs84, native)

    def pixel_to_lonlat(self, col: float, row: float) -> tuple[float, float]:
        gt = self.geotransform
        x = gt[0] + col * gt[1] + row * gt[2]
        y = gt[3] + col * gt[4] + row * gt[5]
        lon, lat, _ = self._transformer(True).TransformPoint(x, y)
        return lon, lat

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        x, y, _ = self._transformer(False).TransformPoint(lon, lat)
        gt = self.geotransform
        det = gt[1] * gt[5] - gt[2] * gt[4]
        col = ((x - gt[0]) * gt[5] - (y - gt[3]) * gt[2]) / det
        row = ((y - gt[3]) * gt[1] - (x - gt[0]) * gt[4]) / det
        return col, row

    def corners_lonlat(self) -> list[list[float]]:
        """Mapbox image-source order: top-left, top-right, bottom-right, bottom-left."""
        return [list(self.pixel_to_lonlat(c, r)) for c, r in ((0, 0), (self.cols, 0), (self.cols, self.rows), (0, self.rows))]

    def contains(self, lon: float, lat: float) -> bool:
        try:
            col, row = self.lonlat_to_pixel(lon, lat)
        except RuntimeError:  # outside the projection domain entirely
            return False
        return 0 <= col < self.cols and 0 <= row < self.rows


def read_geometry(scene_dir: Path = config.SCENE_DIR) -> SceneGeometry:
    ds = gdal.Open(str(scene_dir / "Building_DSM.tif"))
    try:
        return SceneGeometry(ds.RasterYSize, ds.RasterXSize, tuple(ds.GetGeoTransform()), ds.GetProjection())
    finally:
        ds = None


def copy_inputs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        shutil.copy2(src / name, dst / name)


def paint_trees(trees_tif: Path, geometry: SceneGeometry, trees: list[dict]) -> int:
    """Raise canopy height inside a disc for each tree. Returns pixels changed."""
    ds = gdal.Open(str(trees_tif), gdal.GA_Update)
    try:
        band = ds.GetRasterBand(1)
        canopy = band.ReadAsArray()
        rows, cols = canopy.shape
        yy, xx = np.ogrid[:rows, :cols]
        before = canopy.copy()
        for tree in trees:
            col, row = geometry.lonlat_to_pixel(tree["lon"], tree["lat"])
            radius_px = float(tree["crown_radius"]) / geometry.pixel_size
            mask = (yy - row + 0.5) ** 2 + (xx - col + 0.5) ** 2 <= radius_px ** 2
            canopy[mask] = np.maximum(canopy[mask], np.float32(tree["height"]))
        band.WriteArray(canopy)
        band.FlushCache()
        return int((canopy != before).sum())
    finally:
        ds = None
