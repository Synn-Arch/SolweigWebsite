"""Version-one persistent encoded visibility, with explicit read-only mmap ownership.

The JSON manifest is the atomic publication point. Its unique uint8 NPY payload
contains codec-v1 patches concatenated in sky order. Offsets, modes, shape,
float32 codebook and whole-file SHA256 are recorded. Old payloads are retained
when a manifest is replaced: dependency validation and garbage collection belong
to the geometry-cache layer, not this storage primitive. Verification reads are
bounded by max_workspace_bytes; decoded planes own their memory independently.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from .visibility import (PackedVisibility, CODEC_VERSION, CODEBOOK_BITS,
                         DEFAULT_WORKSPACE_BYTES, _shape, _workspace)

FORMAT = 'solweig-light-visibility-native'
NATIVE_VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _digest_manifest(value):
    return hashlib.sha256(_canonical({k: v for k, v in value.items() if k != 'manifest_sha256'})).hexdigest()


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Native visibility metadata requires nonnegative integers')
    return value


def _hash_file(path, chunk):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


@dataclass(frozen=True)
class _MappedPatch:
    mode: str
    offset: int
    length: int
    owner: object

    @property
    def payload(self):
        # No view is retained: caller must hold owner's decode lock.
        return self.owner._mapping[self.offset:self.offset + self.length]


class MappedVisibility(PackedVisibility):
    """PackedVisibility interface backed by readonly NPY; close is explicit.

    Data reads after close raise RuntimeError. Decoding holds a lock through the
    copy, so concurrent close cannot invalidate a read. Returned planes survive
    close; no public operation returns a view into the mapped payload.
    """
    def __init__(self, shape, patches, mapping):
        # Deliberately bypass PackedVisibility.__post_init__, which copies bytes.
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, '_mapping', mapping)
        object.__setattr__(self, '_lock', threading.RLock())
        object.__setattr__(self, '_closed', False)
        object.__setattr__(self, '_patches', tuple(_MappedPatch(p['mode'], p['offset'], p['length'], self) for p in patches))

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Native visibility is closed')

    @property
    def closed(self):
        return self._closed

    @property
    def nbytes(self):
        return sum(p.length for p in self._patches)

    encoded_nbytes = nbytes
    storage_nbytes = nbytes

    def decode_pixels(self, patch, start, stop):
        with self._lock:
            self._check_open()
            return super().decode_pixels(patch, start, stop)

    def decode_patch(self, patch):
        with self._lock:
            self._check_open()
            return super().decode_patch(patch)

    def __getitem__(self, index):
        with self._lock:
            self._check_open()
            return super().__getitem__(index)

    def to_dense(self):
        with self._lock:
            self._check_open()
            return super().to_dense()

    def close(self):
        with self._lock:
            if not self._closed:
                object.__setattr__(self, '_closed', True)
                object.__setattr__(self, '_block_descriptor', None)
                self._mapping._mmap.close()
                object.__setattr__(self, '_mapping', None)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()


def save_native_visibility(manifest_path, channel, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Publish immutable encoded NPY plus checksummed manifest; return manifest."""
    if not isinstance(channel, PackedVisibility):
        raise TypeError('Native storage requires PackedVisibility')
    budget = _workspace(max_workspace_bytes)
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, payload_name = tempfile.mkstemp(prefix=target.name + '.', suffix='.npy', dir=target.parent)
    payload = Path(payload_name)
    manifest_temp = None
    published = False
    fd_owned = True
    guard = channel._lock if isinstance(channel, MappedVisibility) else nullcontext()
    try:
        with guard:
            if isinstance(channel, MappedVisibility):
                channel._check_open()
            patches = []
            offset = 0
            stream = os.fdopen(fd, 'wb')
            fd_owned = False
            with stream:
                np.lib.format.write_array_header_1_0(stream, {'descr': '|u1', 'fortran_order': False, 'shape': (channel.nbytes,)})
                for patch in channel._patches:
                    data = memoryview(patch.payload).cast('B')
                    length = len(data)
                    patches.append({'mode': patch.mode, 'offset': offset, 'length': length})
                    try:
                        for start in range(0, length, budget):
                            stream.write(data[start:start + budget])
                    finally:
                        data.release()
                    offset += length
                stream.flush()
                os.fsync(stream.fileno())
        os.chmod(payload, 0o444)
        manifest = {'format': FORMAT, 'native_version': NATIVE_VERSION,
                    'codec_version': CODEC_VERSION, 'shape': list(channel.shape),
                    'dtype': '<f4', 'codebook_bits': list(CODEBOOK_BITS), 'patches': patches,
                    'payload': {'name': payload.name, 'dtype': '|u1', 'shape': [offset],
                                'size_bytes': payload.stat().st_size,
                                'sha256': _hash_file(payload, budget)}}
        manifest['manifest_sha256'] = _digest_manifest(manifest)
        fd, manifest_temp = tempfile.mkstemp(prefix=target.name + '.', suffix='.json.tmp', dir=target.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_canonical(manifest) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_temp, target)
        published = True
        return manifest
    finally:
        if fd_owned:
            os.close(fd)
        if not published:
            payload.unlink(missing_ok=True)
        if manifest_temp is not None:
            Path(manifest_temp).unlink(missing_ok=True)


def open_native_visibility(manifest_path, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Verify bounded chunks before mapping; reject incomplete/corrupt storage."""
    budget = _workspace(max_workspace_bytes)
    target = Path(manifest_path)
    with open(target, 'rb') as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get('manifest_sha256') != _digest_manifest(manifest):
        raise ValueError('Native visibility manifest integrity mismatch')
    if (manifest.get('format') != FORMAT or type(manifest.get('native_version')) is not int or manifest.get('native_version') != NATIVE_VERSION
            or type(manifest.get('codec_version')) is not int or manifest.get('codec_version') != CODEC_VERSION or manifest.get('dtype') != '<f4'
            or manifest.get('codebook_bits') != list(CODEBOOK_BITS)):
        raise ValueError('Unsupported native visibility schema or codebook')
    shape_data = manifest.get('shape')
    if not isinstance(shape_data, list) or len(shape_data) != 3:
        raise ValueError('Invalid native visibility shape')
    shape = _shape(tuple(_integer(v) for v in shape_data))
    pixels = shape[0] * shape[1]
    patches = manifest.get('patches')
    if not isinstance(patches, list) or len(patches) != shape[2]:
        raise ValueError('Native patch count mismatch')
    offset = 0
    for patch in patches:
        if not isinstance(patch, dict) or patch.get('mode') not in ('binary', 'ternary', 'raw'):
            raise ValueError('Invalid native patch encoding')
        width = 1 if patch['mode'] == 'binary' else 2
        length = pixels * 4 if patch['mode'] == 'raw' else (pixels * width + 7) // 8
        if _integer(patch.get('offset')) != offset or _integer(patch.get('length')) != length:
            raise ValueError('Invalid native patch offsets or lengths')
        offset += length
    info = manifest.get('payload')
    if not isinstance(info, dict) or info.get('dtype') != '|u1' or info.get('shape') != [offset]:
        raise ValueError('Invalid native payload metadata')
    name = info.get('name')
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Invalid native payload filename')
    payload = target.parent / name
    if payload.is_symlink() or payload.stat().st_size != _integer(info.get('size_bytes')):
        raise ValueError('Native payload file mismatch')
    if _hash_file(payload, budget) != info.get('sha256'):
        raise ValueError('Native payload integrity mismatch')
    with open(payload, 'rb') as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError('Unsupported native NPY version')
        npy_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        data_offset = stream.tell()
        if npy_shape != (offset,) or fortran or dtype != np.dtype('uint8') or data_offset + offset != payload.stat().st_size:
            raise ValueError('Invalid native NPY layout')
        for patch in patches:
            if patch['mode'] == 'raw' or not patch['length']:
                continue
            stream.seek(data_offset + patch['offset'])
            remaining = patch['length']
            last = 0
            while remaining:
                data = stream.read(min(remaining, budget // 8))
                if not data:
                    raise ValueError('Truncated native payload')
                remaining -= len(data)
                last = data[-1]
                if patch['mode'] == 'ternary':
                    for value in data:
                        if any(((value >> shift) & 3) == 3 for shift in (0, 2, 4, 6)):
                            raise ValueError('Reserved native visibility code')
            used = pixels * (1 if patch['mode'] == 'binary' else 2) % 8
            if used and last >> used:
                raise ValueError('Nonzero native visibility padding')
    mapping = np.lib.format.open_memmap(payload, mode='r')
    return MappedVisibility(shape, patches, mapping)
