"""Lossless patch visibility codec v1 and bounded legacy NPZ interchange.

Frozen v1 layout: shape=(rows, cols, patches), public dtype=float32; patches
retain sky order. Each immutable patch has one mode and an immutable bytes
payload. Canonical IEEE-754 +0/1 use 1 bit/pixel; +0/1/2 use 2 bits/pixel.
Codes are 0/1/2, pixels are row-major, and the first pixel uses the low bit(s).
Unused final bits are zero. Any other bit pattern uses row-major little-endian
uint32 payloads, preserving every float32 bit, including -0 and NaN payloads.
No native persisted cache format is introduced here.

Implicit array conversion and patch slices are prohibited. Explicit to_dense
is the compatibility escape hatch; decode_patch retains only one raster plane.
Import/export workspace limits bound explicit numeric buffers and read chunks;
encoded output storage, mapped disk pages, Python objects and zip/zlib's internal
IO buffers are separate. Import maps a unique temporary NPY read-only and encodes
one patch incrementally in bounded pixel chunks. Export streams C-order legacy
float32 members without materializing any visibility cube. Each NPZ replacement
is atomic; multi-artifact transaction/manifests remain the caller's responsibility.
"""
from dataclasses import dataclass
import operator
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

CODEC_VERSION = 1
FLOAT32 = np.dtype(np.float32)
CODEBOOK_BITS = (0x00000000, 0x3F800000, 0x40000000)
LEGACY_MEMBERS = ('shadowmat', 'vegshadowmat', 'vbshmat')
DEFAULT_WORKSPACE_BYTES = 1024 * 1024


def _shape(shape):
    values = tuple(operator.index(value) for value in shape)
    if len(values) != 3 or any(value < 0 for value in values):
        raise ValueError('Visibility shape must be three nonnegative integers')
    return values


def _workspace(value):
    value = operator.index(value)
    if value < 128:
        raise ValueError('Visibility workspace must be at least 128 bytes')
    return value


def _patch(index, count):
    if isinstance(index, (bool, np.bool_)):
        raise TypeError('Visibility patch must be an integer')
    index = operator.index(index)
    if index < 0:
        index += count
    if not 0 <= index < count:
        raise IndexError('Visibility patch index out of range')
    return index


def _index(index, count):
    if not isinstance(index, tuple) or len(index) != 3:
        raise IndexError('Visibility indexing requires [rows, columns, integer_patch]')
    return index[0], index[1], _patch(index[2], count)


@dataclass(frozen=True)
class _EncodedPatch:
    mode: str
    payload: bytes


@dataclass(frozen=True)
class PackedVisibility:
    shape: tuple
    _patches: tuple
    codec_version = CODEC_VERSION
    dtype = FLOAT32
    ndim = 3

    def __post_init__(self):
        object.__setattr__(self, 'shape', _shape(self.shape))
        patches = tuple(_EncodedPatch(value.mode, bytes(value.payload)) for value in self._patches)
        if len(patches) != self.shape[2]:
            raise ValueError('Encoded patch count differs from visibility shape')
        pixels = self.shape[0] * self.shape[1]
        for value in patches:
            if value.mode not in ('binary', 'ternary', 'raw'):
                raise ValueError('Unknown visibility encoding')
            expected = pixels * 4 if value.mode == 'raw' else (pixels * (1 if value.mode == 'binary' else 2) + 7) // 8
            if len(value.payload) != expected:
                raise ValueError('Encoded visibility payload length differs from shape')
        object.__setattr__(self, '_patches', patches)

    @property
    def nbytes(self):
        """Encoded payload bytes; excludes Python object overhead."""
        return sum(len(value.payload) for value in self._patches)

    encoded_nbytes = nbytes
    storage_nbytes = nbytes

    @property
    def modes(self):
        return tuple(value.mode for value in self._patches)

    def __array__(self, dtype=None, copy=None):
        raise TypeError('Implicit dense visibility conversion is prohibited; use to_dense() explicitly')

    def decode_pixels(self, patch, start, stop):
        """Decode a contiguous row-major pixel interval, returning owned float32."""
        patch = _patch(patch, self.shape[2])
        start, stop = operator.index(start), operator.index(stop)
        total = self.shape[0] * self.shape[1]
        if not 0 <= start <= stop <= total:
            raise IndexError('Visibility pixel interval out of range')
        value = self._patches[patch]
        if value.mode == 'raw':
            bits = np.frombuffer(value.payload, dtype='<u4', count=stop-start, offset=start*4)
            return bits.astype(np.uint32, copy=False).view(np.float32).copy()
        width = 1 if value.mode == 'binary' else 2
        per_byte = 8 // width
        indices = np.arange(start, stop, dtype=np.int64)
        packed = np.frombuffer(value.payload, dtype=np.uint8)
        codes = (packed[indices // per_byte] >> ((indices % per_byte) * width)) & ((1 << width) - 1)
        return np.asarray(CODEBOOK_BITS, dtype=np.uint32)[codes].view(np.float32)

    def decode_patch(self, patch):
        """Decode exactly one complete plane, independent of encoded storage."""
        patch = _patch(patch, self.shape[2])
        value = self._patches[patch]
        total = self.shape[0] * self.shape[1]
        if value.mode == 'raw':
            return self.decode_pixels(patch, 0, total).reshape(self.shape[:2])
        packed = np.frombuffer(value.payload, dtype=np.uint8)
        if value.mode == 'binary':
            codes = np.unpackbits(packed, bitorder='little', count=total)
        else:
            codes = np.empty(total, dtype=np.uint8)
            for offset in range(4):
                count = codes[offset::4].size
                codes[offset::4] = (packed[:count] >> (2 * offset)) & 3
        return np.asarray(CODEBOOK_BITS, dtype=np.uint32)[codes].view(np.float32).reshape(self.shape[:2])

    def __getitem__(self, index):
        row, column, patch = _index(index, self.shape[2])
        return self.decode_patch(patch)[row, column]

    def to_dense(self):
        """Explicit compatibility materialization; caller accepts cube allocation."""
        result = np.empty(self.shape, dtype=np.float32)
        for patch in range(self.shape[2]):
            result[:, :, patch] = self.decode_patch(patch)
        return result

    @classmethod
    def from_dense(cls, array, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
        if not isinstance(array, np.ndarray) or array.ndim != 3:
            raise TypeError('from_dense requires a three-dimensional float32 ndarray')
        builder = VisibilityBuilder(array.shape, max_workspace_bytes=max_workspace_bytes)
        for patch in range(array.shape[2]):
            builder.append(array[:, :, patch])
        return builder.finish()


class VisibilityBuilder:
    """Encode each completed 2D patch immediately; never retain its source array."""
    def __init__(self, shape, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
        self.shape = _shape(shape)
        self.max_workspace_bytes = _workspace(max_workspace_bytes)
        self._patches = []
        self._finished = None
        # Multiple-of-eight chunks keep every packed chunk byte-aligned.
        self.chunk_pixels = max(8, (self.max_workspace_bytes // 16 // 8) * 8)

    @property
    def patches_added(self):
        return len(self._patches)

    @property
    def nbytes(self):
        return sum(len(value.payload) for value in self._patches)

    def append(self, plane):
        if self._finished is not None:
            raise RuntimeError('Visibility builder is already finished')
        if len(self._patches) >= self.shape[2]:
            raise ValueError('Too many visibility patches')
        if not isinstance(plane, np.ndarray) or plane.shape != self.shape[:2]:
            raise ValueError('Visibility plane shape differs from builder shape')
        if plane.dtype.kind != 'f' or plane.dtype.itemsize != 4:
            raise TypeError('Visibility planes must contain float32 values')
        total = plane.size
        bit_dtype = np.dtype('u4').newbyteorder(plane.dtype.byteorder)
        ternary = False
        raw = False
        for start in range(0, total, self.chunk_pixels):
            bits = plane.flat[start:start+self.chunk_pixels].view(bit_dtype)
            invalid = (bits != CODEBOOK_BITS[0]) & (bits != CODEBOOK_BITS[1]) & (bits != CODEBOOK_BITS[2])
            if invalid.any():
                raw = True
                break
            ternary = ternary or bool(np.any(bits == CODEBOOK_BITS[2]))
        width = 2 if ternary else 1
        mode = 'raw' if raw else 'ternary' if ternary else 'binary'
        payload = bytearray(total*4 if raw else (total*width+7)//8)
        for start in range(0, total, self.chunk_pixels):
            bits = plane.flat[start:start+self.chunk_pixels].view(bit_dtype)
            if raw:
                payload[start*4:(start+len(bits))*4] = bits.astype('<u4', copy=False).tobytes()
            else:
                codes = np.zeros(bits.size, dtype=np.uint8)
                codes[bits == CODEBOOK_BITS[1]] = 1
                if ternary:
                    codes[bits == CODEBOOK_BITS[2]] = 2
                    padded = np.zeros((codes.size+3)//4*4, dtype=np.uint8)
                    padded[:codes.size] = codes
                    packed = padded[0::4] | (padded[1::4] << 2) | (padded[2::4] << 4) | (padded[3::4] << 6)
                else:
                    packed = np.packbits(codes, bitorder='little')
                offset = start*width//8
                payload[offset:offset+packed.size] = packed.tobytes()
        self._patches.append(_EncodedPatch(mode, bytes(payload)))

    def finish(self):
        if len(self._patches) != self.shape[2]:
            raise ValueError('Visibility builder does not contain every patch')
        if self._finished is None:
            self._finished = PackedVisibility(self.shape, tuple(self._patches))
        return self._finished


@dataclass(frozen=True)
class LazyDiffVisibility:
    """Original float32 diffuse-visibility arithmetic, evaluated one patch at a time."""
    shadow: object
    vegetation: object
    dtype = FLOAT32
    ndim = 3

    def __post_init__(self):
        if self.shadow.shape != self.vegetation.shape or len(self.shadow.shape) != 3:
            raise ValueError('Diffuse visibility channels must have matching 3D shapes')
        if self.shadow.dtype != FLOAT32 or self.vegetation.dtype != FLOAT32:
            raise TypeError('Diffuse visibility channels must contain float32 values')

    @property
    def shape(self):
        return self.shadow.shape

    def __array__(self, dtype=None, copy=None):
        raise TypeError('Implicit dense diffuse visibility conversion is prohibited')

    def __getitem__(self, index):
        row, column, patch = _index(index, self.shape[2])
        shadow = self.shadow[:, :, patch]
        vegetation = self.vegetation[:, :, patch]
        return (shadow - (np.float32(1) - vegetation) * np.float32(1-.03))[row, column]

    def decode_pixels(self, patch, start, stop):
        # Useful for consumers that also support bounded contiguous pixel segments.
        shadow = self.shadow.decode_pixels(patch, start, stop)
        vegetation = self.vegetation.decode_pixels(patch, start, stop)
        return shadow - (np.float32(1) - vegetation) * np.float32(1-.03)


def _channels(shadow, vegetation, vegetation_building):
    result = dict(zip(LEGACY_MEMBERS, (shadow, vegetation, vegetation_building)))
    shapes = {channel.shape for channel in result.values()}
    if len(shapes) != 1:
        raise ValueError('Legacy visibility channels must have matching shapes')
    for channel in result.values():
        if channel.dtype != FLOAT32 or len(channel.shape) != 3:
            raise TypeError('Legacy visibility channels must have float32 3D shapes')
    return result


def export_visibility_npz(path, shadow, vegetation, vegetation_building, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    """Atomically stream exact legacy member names/dtype/shape/C order.

    Numeric chunk buffers plus conservative 64-byte/pixel decoder reserve fit
    max_workspace_bytes. A smaller-than-one-full-pixel budget streams patch
    subsequences instead. No complete cube or complete plane is decoded.
    """
    budget = _workspace(max_workspace_bytes)
    channels = _channels(shadow, vegetation, vegetation_building)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shape = shadow.shape
    pixels, patches = shape[0]*shape[1], shape[2]
    descriptor, temporary = tempfile.mkstemp(prefix='.'+target.name+'.', suffix='.tmp', dir=target.parent)
    os.close(descriptor)
    maximum = 0
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name, channel in channels.items():
                with archive.open(name+'.npy', 'w', force_zip64=True) as stream:
                    np.lib.format.write_array_header_1_0(stream, {'descr':'<f4','fortran_order':False,'shape':shape})
                    if not pixels or not patches:
                        continue
                    pixel_chunk = budget // (4*patches+64)
                    if pixel_chunk:
                        for start in range(0, pixels, pixel_chunk):
                            stop = min(pixels, start+pixel_chunk)
                            chunk = np.empty((stop-start, patches), dtype='<f4')
                            maximum = max(maximum, (stop-start)*(4*patches+64))
                            for patch in range(patches):
                                chunk[:, patch] = channel.decode_pixels(patch, start, stop)
                            stream.write(memoryview(chunk).cast('B'))
                    else:
                        element_chunk = budget // 64
                        total = pixels*patches
                        for start in range(0, total, element_chunk):
                            count = min(element_chunk, total-start)
                            chunk = np.empty(count, dtype='<f4')
                            maximum = max(maximum, count*64)
                            for patch in range(patches):
                                offset = (patch-start) % patches
                                if offset >= count:
                                    continue
                                length = (count-1-offset)//patches+1
                                pixel = (start+offset)//patches
                                chunk[offset::patches] = channel.decode_pixels(patch, pixel, pixel+length)
                            stream.write(memoryview(chunk).cast('B'))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'numeric_workspace_limit_bytes':budget,'conservative_peak_numeric_workspace_bytes':maximum}


def import_visibility_npz(path, *, max_workspace_bytes=DEFAULT_WORKSPACE_BYTES, scratch_dir=None):
    """Return compact channels keyed by shadowmat/vegshadowmat/vbshmat.

    ZIP members are copied in bounded chunks to collision-safe temporary NPYs,
    mapped read-only, then encoded patch by patch. All mappings and files are
    closed/removed before returning; compact channels own only immutable bytes.
    """
    budget = _workspace(max_workspace_bytes)
    result = {}
    with tempfile.TemporaryDirectory(prefix='.visibility-', dir=scratch_dir) as temporary:
        with zipfile.ZipFile(path, 'r') as archive:
            names = archive.namelist()
            for name in LEGACY_MEMBERS:
                member = name+'.npy'
                if names.count(member) != 1:
                    raise ValueError(f'Legacy archive must contain exactly one {member}')
                destination = Path(temporary)/member
                with archive.open(member, 'r') as stream, destination.open('wb') as output:
                    while True:
                        chunk = stream.read(budget)
                        if not chunk:
                            break
                        output.write(chunk)
                mapped = np.lib.format.open_memmap(destination, mode='r')
                try:
                    if mapped.dtype.kind != 'f' or mapped.dtype.itemsize != 4 or mapped.ndim != 3:
                        raise TypeError('Legacy visibility member must have float32 3D shape')
                    # open_memmap accepts trailing bytes, while a legacy member has
                    # exactly its NPY header and declared float32 payload.
                    if destination.stat().st_size != mapped.offset + mapped.size*4:
                        raise ValueError('Legacy visibility member payload length differs from NPY shape')
                    builder = VisibilityBuilder(mapped.shape, max_workspace_bytes=budget)
                    for patch in range(mapped.shape[2]):
                        plane = mapped[:, :, patch]
                        builder.append(plane)
                        del plane
                    result[name] = builder.finish()
                finally:
                    mapped._mmap.close()
        if len({channel.shape for channel in result.values()}) != 1:
            raise ValueError('Legacy visibility channels must have matching shapes')
    return result
