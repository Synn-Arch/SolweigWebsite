"""Dependency-addressed immutable geometry generations with explicit ownership.

A manifest is the only commit point. Generations are never removed here: an old
reader may still own mapped payloads. Cleanup, if added, must prove no readers.
Only geometry belongs in this store; dynamic thermal/radiation state does not.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from ..geometry.visibility import PackedVisibility, DEFAULT_WORKSPACE_BYTES, _workspace
from ..geometry.visibility_native import save_native_visibility, open_native_visibility

SVF_FIELDS = tuple('svf svfE svfS svfW svfN svfveg svfEveg svfSveg svfWveg svfNveg svfaveg svfEaveg svfSaveg svfWaveg svfNaveg'.split())
ARRAY_FIELDS = SVF_FIELDS + ('svftotal',)
VISIBILITY_FIELDS = ('shmat', 'vegshmat', 'vbshvegshmat')
FORMAT = 'solweig-light-geometry-cache'
VERSION = 1
MODEL_VERSION = 'p6-geometry-v1'
MAX_MANIFEST_BYTES = 1024 * 1024


def canonical_json(value):
    """Strict JSON canonicalization: reject non-JSON keys and nonfinite numbers."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError('Dependency identity keys must be strings')
            for value in item.values():
                check(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                check(value)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise TypeError('Dependency identity must be JSON-safe')
    check(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _hash_file(path, budget):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while data := stream.read(budget):
            digest.update(data)
    return digest.hexdigest()


def content_fingerprint(path, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Fingerprint actual file bytes; size/mtime never stand in for content."""
    budget = _workspace(max_workspace_bytes)
    path = Path(path)
    return {'size_bytes': path.stat().st_size, 'sha256': _hash_file(path, budget)}


def array_fingerprint(array, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Hash dtype, shape and C-order exact bytes in bounded numeric chunks."""
    budget = _workspace(max_workspace_bytes)
    array = np.asarray(array)
    if array.dtype.hasobject or not array.dtype.itemsize:
        raise TypeError('Object or zero-width arrays cannot identify geometry')
    digest = hashlib.sha256()
    for chunk in np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                          op_flags=['readonly'], order='C', buffersize=max(1, budget // array.dtype.itemsize)):
        digest.update(chunk.tobytes(order='C'))
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'sha256': digest.hexdigest()}


def raster_fingerprint(path, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """File content plus all GDAL raster/band metadata relevant to reuse."""
    from osgeo import gdal
    def metadata_domains(obj):
        return {domain: obj.GetMetadata(domain) for domain in obj.GetMetadataDomainList() or ['']}
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f'Cannot open raster {path}')
    try:
        bands = []
        for index in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(index)
            nodata = band.GetNoDataValue()
            # JSON cannot represent NaN; this token preserves that metadata.
            if nodata is not None and not np.isfinite(nodata):
                nodata = {'nonfinite': str(nodata)}
            bands.append({'dtype': gdal.GetDataTypeName(band.DataType), 'nodata': nodata,
                          'description': band.GetDescription(), 'metadata': metadata_domains(band),
                          'scale': band.GetScale(), 'offset': band.GetOffset(),
                          'unit': band.GetUnitType(), 'color_interpretation': band.GetColorInterpretation(),
                          'mask_flags': band.GetMaskFlags(), 'categories': band.GetCategoryNames()})
        return {**content_fingerprint(path, max_workspace_bytes=max_workspace_bytes),
                'shape': [dataset.RasterYSize, dataset.RasterXSize], 'bands': bands,
                'geotransform': list(dataset.GetGeoTransform()), 'projection': dataset.GetProjection(),
                'metadata': metadata_domains(dataset), 'gcp_projection': dataset.GetGCPProjection(),
                'gcps': [{'id': point.Id, 'info': point.Info, 'pixel': point.GCPPixel,
                          'line': point.GCPLine, 'x': point.GCPX, 'y': point.GCPY, 'z': point.GCPZ}
                         for point in dataset.GetGCPs()]}
    finally:
        dataset = None


@contextmanager
def _lock(path):
    # Separate file descriptions also coordinate threads in a single process.
    with open(path, 'a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _manifest_digest(value):
    return hashlib.sha256(canonical_json({k: v for k, v in value.items() if k != 'manifest_sha256'})).hexdigest()


def _read_json(path):
    with open(path, 'rb') as stream:
        data = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError('Geometry manifest exceeds limit')
    return json.loads(data)


def _safe_path(directory, name):
    if not isinstance(name, str):
        raise ValueError('Invalid cache payload name')
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 2 or any(p in ('', '.', '..') for p in relative.parts):
        raise ValueError('Invalid cache payload path')
    path = directory / relative
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Cache payload cannot be a symlink')
    return path


@dataclass
class GeometryHandle:
    fields: dict
    hit: bool
    key: str
    closed: bool = False

    def close(self):
        if not self.closed:
            self.closed = True
            for value in self.fields.values():
                if isinstance(value, PackedVisibility) and hasattr(value, 'close'):
                    value.close()
                elif isinstance(value, np.memmap):
                    value._mmap.close()
            self.fields.clear()

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Geometry handle is closed')
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class GeometryStore:
    def __init__(self, root, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES, model_version=MODEL_VERSION):
        self.root = Path(root)
        self.max_workspace_bytes = _workspace(max_workspace_bytes)
        self.model_version = str(model_version)

    def key_for(self, identity):
        if not isinstance(identity, dict):
            raise TypeError('Geometry identity must be a mapping')
        return hashlib.sha256(canonical_json({'identity': identity, 'model_version': self.model_version,
                                            'format': FORMAT, 'version': VERSION})).hexdigest()

    def _open(self, directory, identity, key, hit):
        manifest = _read_json(directory / 'manifest.json')
        if (not isinstance(manifest, dict) or manifest.get('manifest_sha256') != _manifest_digest(manifest)
                or manifest.get('format') != FORMAT or manifest.get('version') != VERSION
                or manifest.get('model_version') != self.model_version or manifest.get('key') != key
                or canonical_json(manifest.get('identity')) != canonical_json(identity)):
            raise ValueError('Geometry manifest identity/schema/integrity mismatch')
        arrays, visibility = manifest.get('arrays'), manifest.get('visibility')
        if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_FIELDS) or not isinstance(visibility, dict) or set(visibility) != set(VISIBILITY_FIELDS):
            raise ValueError('Geometry manifest field schema mismatch')
        fields = {}
        try:
            for name, info in arrays.items():
                path = _safe_path(directory, info['name'])
                if content_fingerprint(path, max_workspace_bytes=self.max_workspace_bytes) != info['fingerprint']:
                    raise ValueError('Geometry array integrity mismatch')
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                fields[name] = value
                if value.dtype != np.dtype('<f4') or value.ndim != 2 or list(value.shape) != info['shape'] or path.stat().st_size != value.offset + value.nbytes:
                    raise ValueError('Geometry array layout mismatch')
            shapes = {value.shape for value in fields.values()}
            if len(shapes) != 1:
                raise ValueError('Geometry field shapes differ')
            for name, info in visibility.items():
                path = _safe_path(directory, info['name'])
                if content_fingerprint(path, max_workspace_bytes=self.max_workspace_bytes) != info['fingerprint']:
                    raise ValueError('Geometry visibility manifest integrity mismatch')
                if _read_json(path).get('payload') != info['payload']:
                    raise ValueError('Geometry visibility payload manifest mismatch')
                channel = open_native_visibility(path, max_workspace_bytes=self.max_workspace_bytes)
                fields[name] = channel
                if channel.shape[:2] not in shapes or list(channel.shape) != info['shape']:
                    raise ValueError('Geometry visibility shape mismatch')
            if len({fields[name].shape for name in VISIBILITY_FIELDS}) != 1:
                raise ValueError('Geometry visibility patch counts differ')
            return GeometryHandle(fields, hit, key)
        except BaseException:
            GeometryHandle(fields, hit, key).close()
            raise

    def _publish(self, directory, identity, key, fields):
        if set(fields) != set(ARRAY_FIELDS + VISIBILITY_FIELDS):
            raise ValueError('Producer must return sixteen SVF arrays and three packed channels')
        shape = None
        for name in ARRAY_FIELDS:
            value = fields[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2:
                raise TypeError('Geometry fields must be two-dimensional float32 arrays')
            shape = value.shape if shape is None else shape
            if value.shape != shape:
                raise ValueError('Geometry array shapes differ')
        channels = [fields[name] for name in VISIBILITY_FIELDS]
        if any(not isinstance(v, PackedVisibility) or v.shape[:2] != shape for v in channels) or len({v.shape for v in channels}) != 1:
            raise ValueError('Producer visibility must be matching PackedVisibility channels')
        # Immutable unique generations keep previous mapped readers valid.
        generation = directory / ('generation-' + uuid.uuid4().hex)
        generation.mkdir()
        manifest = {'format': FORMAT, 'version': VERSION, 'model_version': self.model_version,
                    'key': key, 'identity': identity, 'arrays': {}, 'visibility': {}}
        for name in ARRAY_FIELDS:
            path = generation / (name + '.npy')
            value = fields[name]
            with open(path, 'wb') as stream:
                np.lib.format.write_array_header_1_0(stream, {'descr': '<f4', 'fortran_order': False, 'shape': shape})
                for chunk in np.nditer(value, flags=['external_loop', 'buffered', 'zerosize_ok'], op_flags=['readonly'],
                                      op_dtypes=[np.dtype('<f4')], order='C', buffersize=max(1, self.max_workspace_bytes // 4)):
                    stream.write(chunk.tobytes(order='C'))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(path, 0o444)
            manifest['arrays'][name] = {'name': f'{generation.name}/{path.name}', 'shape': list(shape),
                                       'fingerprint': content_fingerprint(path, max_workspace_bytes=self.max_workspace_bytes)}
        for name in VISIBILITY_FIELDS:
            path = generation / (name + '.json')
            native = save_native_visibility(path, fields[name], max_workspace_bytes=self.max_workspace_bytes)
            manifest['visibility'][name] = {'name': f'{generation.name}/{path.name}', 'shape': list(fields[name].shape),
                                           'fingerprint': content_fingerprint(path, max_workspace_bytes=self.max_workspace_bytes),
                                           'payload': native['payload']}
        manifest['manifest_sha256'] = _manifest_digest(manifest)
        data = canonical_json(manifest)
        if len(data) > MAX_MANIFEST_BYTES:
            raise ValueError('Geometry manifest exceeds limit')
        generation_fd = os.open(generation, os.O_RDONLY)
        try:
            os.fsync(generation_fd)
        finally:
            os.close(generation_fd)
        fd, temporary = tempfile.mkstemp(prefix='.manifest-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / 'manifest.json')
            # Flush directory entries after publishing the commit point.
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def get_or_create(self, identity, producer):
        # Freeze caller data so in-place mutation cannot alter an in-flight key.
        identity = json.loads(canonical_json(identity))
        key = self.key_for(identity)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        with _lock(directory / '.lock'):
            try:
                return self._open(directory, identity, key, True)
            except (OSError, ValueError, TypeError, KeyError, EOFError, AttributeError, OverflowError):
                pass  # Invalid or incomplete cache: producer is the only oracle.
            fields = producer()
            self._publish(directory, identity, key, fields)
            return self._open(directory, identity, key, False)
