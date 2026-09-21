"""Strict, read-only validation and explicit trust of identity-less legacy SVF."""
import logging
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .geometry import SVF_FIELDS, content_fingerprint
from ..geometry.visibility import DEFAULT_WORKSPACE_BYTES, LEGACY_MEMBERS, _workspace, import_visibility_npz


def _same_nodata(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return actual == expected or (np.isnan(actual) and np.isnan(expected))


def _check_tiff(path, *, shape, geotransform, projection, nodata, budget):
    from osgeo import gdal
    ds = gdal.Open(str(path))
    if ds is None:
        raise ValueError(f'Cannot open legacy SVF raster {path}')
    try:
        if ds.RasterCount != 1 or (ds.RasterYSize, ds.RasterXSize) != tuple(shape):
            raise ValueError('Legacy SVF raster shape/band count mismatch')
        band = ds.GetRasterBand(1)
        if band.DataType != gdal.GDT_Float32:
            raise ValueError('Legacy SVF raster must have float32 dtype')
        if tuple(ds.GetGeoTransform()) != tuple(geotransform) or ds.GetProjection() != projection:
            raise ValueError('Legacy SVF raster spatial metadata mismatch')
        if not _same_nodata(band.GetNoDataValue(), nodata):
            raise ValueError('Legacy SVF raster NoData mismatch')
        # Opening a TIFF header does not prove its strips/tiles are readable.
        # Verify pixels in bounded windows without retaining any raster history.
        width = min(shape[1], max(1, budget // 4))
        height = max(1, budget // (width * 4))
        for y in range(0, shape[0], height):
            rows = min(height, shape[0] - y)
            for x in range(0, shape[1], width):
                cols = min(width, shape[1] - x)
                data = band.ReadRaster(x, y, cols, rows, buf_type=gdal.GDT_Float32)
                if data is None or len(data) != rows * cols * 4:
                    raise ValueError('Legacy SVF raster pixel payload is unreadable')
    finally:
        ds = None


def validate_legacy_geometry(zip_path, npz_path, total_path, *, shape, patch_count,
                             geotransform, projection, nodata=None,
                             max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Check exact ZIP/NPZ schema, full ZIP CRC, and every raster's metadata.

    Validation proves format compatibility only: these files carry no dependency
    identity. It never establishes that they correspond to current scene inputs.
    All three artifacts are required; total SVF is not recomputed or substituted.
    """
    budget = _workspace(max_workspace_bytes)
    shape = tuple(shape)
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape) or type(patch_count) is not int or patch_count <= 0:
        raise ValueError('Expected legacy geometry dimensions must be positive integers')
    expected = {name + '.tif' for name in SVF_FIELDS}
    metadata = dict(shape=shape, geotransform=geotransform, projection=projection, nodata=nodata, budget=budget)
    # Extract one raster at a time in bounded chunks. No original is modified.
    with tempfile.TemporaryDirectory(prefix='solweig-legacy-validate-') as temporary:
        with zipfile.ZipFile(zip_path) as archive:
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
                raise ValueError('Legacy SVF ZIP must contain exactly the fifteen named TIFFs')
            for name in sorted(expected):
                path = Path(temporary) / name
                with archive.open(name) as source, path.open('wb') as target:
                    while chunk := source.read(budget):
                        target.write(chunk)
                _check_tiff(path, **metadata)
                path.unlink()
    _check_tiff(total_path, **metadata)
    with zipfile.ZipFile(npz_path) as archive:
        expected_members = {name + '.npy' for name in LEGACY_MEMBERS}
        if len(archive.namelist()) != 3 or set(archive.namelist()) != expected_members:
            raise ValueError('Legacy visibility NPZ must contain exactly three named channels')
        for name in sorted(expected_members):
            info = archive.getinfo(name)
            with archive.open(name) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    array_shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    array_shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError('Unsupported legacy visibility NPY version')
                if array_shape != (*shape, patch_count) or dtype.kind != 'f' or dtype.itemsize != 4:
                    raise ValueError('Legacy visibility shape/patch count/dtype mismatch')
                if info.file_size != source.tell() + int(np.prod(array_shape)) * 4:
                    raise ValueError('Legacy visibility payload length mismatch')
                # Consume full member to verify CRC without materializing a cube.
                while source.read(budget):
                    pass
    return {'identity_present': False, 'shape': list(shape), 'patch_count': patch_count,
            'artifacts': {name: content_fingerprint(path, max_workspace_bytes=budget)
                          for name, path in (('svfs', zip_path), ('shadowmats', npz_path), ('svftotal', total_path))}}


def load_legacy_geometry(zip_path, npz_path, total_path, *, trust=False, logger=None, **validation):
    """Return validated fields only with explicit trust; default requests recompute.

    Schema failures raise recoverable ValueError/OSError/BadZipFile to the caller;
    the caller may ignore an invalid old artifact and run the real producer.
    """
    report = validate_legacy_geometry(zip_path, npz_path, total_path, **validation)
    if not trust:
        return None
    (logger or logging.getLogger(__name__)).warning(
        'Trusted legacy geometry without dependency identity: %s', report['artifacts'])
    from osgeo import gdal
    fields = {}
    for name in SVF_FIELDS:
        path = f'/vsizip/{Path(zip_path).resolve()}/{name}.tif'
        dataset = gdal.Open(path)
        try:
            fields[name] = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
        finally:
            dataset = None
    dataset = gdal.Open(str(total_path))
    try:
        fields['svftotal'] = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    finally:
        dataset = None
    channels = import_visibility_npz(npz_path, max_workspace_bytes=validation.get('max_workspace_bytes', DEFAULT_WORKSPACE_BYTES))
    fields.update({name: channels[legacy] for name, legacy in (
        ('shmat', 'shadowmat'), ('vegshmat', 'vegshadowmat'), ('vbshvegshmat', 'vbshmat'))})
    return fields
