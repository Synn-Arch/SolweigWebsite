"""Single-band GeoTIFF input and bounded-lifetime multiband output."""
from dataclasses import dataclass
import datetime
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


@dataclass(frozen=True)
class RasterMetadata:
    rows: int
    cols: int
    transform: tuple
    projection: str


def read_raster(path):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(path)
    values = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    metadata = RasterMetadata(dataset.RasterYSize, dataset.RasterXSize,
                              dataset.GetGeoTransform(), dataset.GetProjection())
    dataset = None
    return values, metadata


def write_single(path, values, metadata):
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), metadata.cols, metadata.rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(metadata.transform)
    dataset.SetProjection(metadata.projection)
    dataset.GetRasterBand(1).WriteArray(np.asarray(values, dtype=np.float32))
    dataset.FlushCache()
    dataset = None


class StreamingOutputs:
    """One dataset owner; each live raster band is consumed before reuse.

    No nodata tag is invented. NaN masks and timestamp metadata follow the
    original writer. Atomic completion/restart belongs to the later P6 gate.
    """
    def __init__(self, directory, tile, metadata, met, selected_date, fields):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.metadata = metadata
        self.met = met
        self.date = datetime.datetime.strptime(selected_date, "%Y-%m-%d")
        self.next_band = 0
        try:
            for field in fields:
                path = self.directory / f"{field}_{tile}.tif"
                dataset = gdal.GetDriverByName("GTiff").Create(str(path), metadata.cols, metadata.rows,
                                                               len(met), gdal.GDT_Float32)
                dataset.SetGeoTransform(metadata.transform)
                dataset.SetProjection(metadata.projection)
                self.datasets[field] = dataset
        except BaseException:
            self.close()
            raise

    def write(self, timestep, fields):
        if timestep != self.next_band:
            raise ValueError("Output timesteps must be written in chronological order")
        timestamp = self.date.replace(hour=int(self.met[timestep, 2]), minute=int(self.met[timestep, 3])).isoformat()
        for name, dataset in self.datasets.items():
            values = np.asarray(fields[name], dtype=np.float32)
            if values.shape != (self.metadata.rows, self.metadata.cols):
                raise ValueError(f"Unexpected output shape for {name}: {values.shape}")
            band = dataset.GetRasterBand(timestep + 1)
            band.WriteArray(values)
            band.SetMetadata({"Time": timestamp})
            band.FlushCache()
        self.next_band += 1

    def close(self):
        for dataset in self.datasets.values():
            dataset.FlushCache()
        self.datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        if exc_type is None and self.next_band != len(self.met):
            raise RuntimeError("Simulation ended before every requested output band was written")
