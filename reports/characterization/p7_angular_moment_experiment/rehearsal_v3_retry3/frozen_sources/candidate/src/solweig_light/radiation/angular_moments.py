"""Isolated immutable ownership model for the reviewed longwave moments."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading

import numpy as np

ALGORITHM = "p7-six-reflection-moments-f64-v1"


def _digest_array(digest, value):
    value = np.asarray(value)
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    iterator = np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=["readonly"], order="C", buffersize=131072)
    for chunk in iterator:
        digest.update(chunk.tobytes())


def _visibility_shape(value):
    if hasattr(value, "decode_pixels"):
        rows, cols, patches = value.shape
        return rows*cols, patches
    value = np.asarray(value)
    if value.dtype != np.float32:
        raise TypeError("dense visibility must be exact float32")
    if value.ndim == 3:
        return value.shape[0]*value.shape[1],value.shape[2]
    if value.ndim != 2:
        raise ValueError("dense visibility must be pixels by patches or rows by cols by patches")
    return value.shape


def _visibility_block(value, start, stop, patches):
    if hasattr(value, "decode_pixels"):
        block = np.empty((stop-start, patches), np.float32)
        for patch in range(patches):
            block[:, patch] = value.decode_pixels(patch, start, stop)
        return block
    value=np.asarray(value)
    return value.reshape(-1,patches)[start:stop]


def _digest_visibility(digest, value, *, block_pixels=128):
    pixels, patches = _visibility_shape(value)
    digest.update(np.dtype(np.float32).str.encode())
    digest.update(np.asarray((pixels, patches), np.int64).tobytes())
    for start in range(0, pixels, block_pixels):
        digest.update(_visibility_block(value, start, min(start+block_pixels, pixels), patches).tobytes(order="C"))


def complete_identity(sh, vs, vb, solid, sine, cosine, directions, gate, *, block_pixels=128,
                      logical_shape=None, profile="default", source_fingerprint="unfrozen"):
    digest = hashlib.sha256(ALGORITHM.encode())
    digest.update(str(profile).encode()); digest.update(str(source_fingerprint).encode())
    digest.update(np.asarray(logical_shape or _visibility_shape(sh)[:1], np.int64).tobytes())
    for value in (sh, vs, vb):
        _digest_visibility(digest, value, block_pixels=block_pixels)
    for value in (solid, sine, cosine, directions, gate):
        _digest_array(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class MomentBundle:
    identity: str
    values: np.ndarray  # (pixels, side/down/E/S/W/N), float64
    pixels: int
    patches: int
    logical_shape: tuple
    angular_identity: str
    profile: str
    source_fingerprint: str
    visibility_sources: tuple
    algorithm: str = ALGORITHM

    def __post_init__(self):
        if self.values.dtype != np.float64 or self.values.shape != (self.pixels, 6):
            raise TypeError("invalid moment payload")
        base = self.values
        while isinstance(getattr(base, "base", None), np.ndarray): base = base.base
        if self.values.flags.owndata or not isinstance(base.base, bytes):
            raise TypeError("moment payload must be immutable bytes-backed storage")
        self.values.setflags(write=False)


def build_bundle(sh, vs, vb, solid, sine, cosine, directions, gate, *, block_pixels=128,
                 fail_after_blocks=None, logical_shape=None, profile="default",
                 source_fingerprint="unfrozen"):
    """Construct with bounded pixel blocks; no full visibility cube is copied."""
    if block_pixels < 1:
        raise ValueError("block_pixels must be positive")
    pixels, patches = _visibility_shape(sh)
    if _visibility_shape(vs) != (pixels, patches) or _visibility_shape(vb) != (pixels, patches):
        raise ValueError("visibility shapes differ")
    for value in (sh, vs, vb):
        if not hasattr(value, "decode_pixels"):
            array=np.asarray(value);base=array
            while isinstance(getattr(base,"base",None),np.ndarray):base=base.base
            if array.flags.writeable or not isinstance(getattr(base,"base",None),bytes):
                raise ValueError("dense visibility reuse requires immutable bytes-backed storage")
    angular_values=(solid,sine,cosine,directions)
    if (not 1 <= patches <= 609 or any(not np.isfinite(np.asarray(x)).all() for x in angular_values)
            or any(np.max(np.abs(np.asarray(x)),initial=0)>1 for x in angular_values)):
        raise ValueError("angular moment fast-path domain is unsafe")
    if logical_shape is None: logical_shape = (pixels, 1)
    if len(tuple(logical_shape)) != 2 or any(int(x) < 0 for x in logical_shape) or np.prod(logical_shape) != pixels:
        raise ValueError("logical shape does not match pixel count")
    identity = complete_identity(sh, vs, vb, solid, sine, cosine, directions, gate,
        block_pixels=block_pixels, logical_shape=logical_shape, profile=profile,
        source_fingerprint=source_fingerprint)
    values = np.empty((pixels, 6), np.float64)
    blocks = 0
    try:
        for start in range(0, pixels, block_pixels):
            stop = min(start + block_pixels, pixels)
            if fail_after_blocks is not None and blocks >= fail_after_blocks:
                raise RuntimeError("injected moment construction failure")
            chunks = []
            for channel in (sh, vs, vb):
                chunks.append(_visibility_block(channel, start, stop, patches))
            cs, cv, cb = chunks
            for pixel in range(stop-start):
                row = values[start+pixel]
                row.fill(0)
                for patch in range(patches):
                    if cs[pixel, patch] == 0 or cv[pixel, patch] == 0 or cb[pixel, patch] == 0:
                        fs, fc = np.float64(solid[patch]), np.float64(cosine[patch])
                        row[0] += fs*fc
                        row[1] += fs*np.float64(sine[patch])
                        for direction in range(4):
                            if gate[patch, direction]:
                                row[2+direction] += fs*fc*np.float64(directions[patch, direction])
            blocks += 1
        frozen = np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(pixels, 6)
        values = None
        angular = hashlib.sha256()
        for item in (solid, sine, cosine, directions, gate): _digest_array(angular, item)
        return MomentBundle(identity, frozen, pixels, patches, tuple(logical_shape),
                            angular.hexdigest(), str(profile), str(source_fingerprint),(sh,vs,vb))
    except BaseException:
        # Drop the only payload reference before propagating; owners commit only returns.
        values = None
        raise


class TileMomentOwner:
    """Explicit per-tile owner; never process-global and safe for concurrent tiles."""
    def __init__(self):
        self._lock = threading.Lock()
        self._bundle = None
        self.closed = False

    def build(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError("moment owner is closed")
        bundle = build_bundle(*args, **kwargs)
        with self._lock:
            if self.closed:
                raise RuntimeError("moment owner closed during construction")
            if self._bundle is not None:
                raise RuntimeError("tile moment bundle already constructed")
            self._bundle = bundle
            return bundle

    def close(self):
        with self._lock:
            self._bundle = None
            self.closed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    @property
    def allocation_bytes(self):
        with self._lock:
            return 0 if self._bundle is None else self._bundle.values.nbytes

    @property
    def count(self):
        with self._lock: return int(self._bundle is not None)


def validate_bundle(bundle, logical_shape, sh, vs, vb, solid, sine, cosine, directions, gate,
                    profile="default", source_fingerprint="unfrozen"):
    if not isinstance(bundle, MomentBundle) or bundle.logical_shape != tuple(logical_shape):
        return False
    if bundle.profile != str(profile) or bundle.source_fingerprint != str(source_fingerprint):
        return False
    if any(current is not expected for current,expected in zip((sh,vs,vb),bundle.visibility_sources)):
        return False
    digest = hashlib.sha256()
    for item in (solid, sine, cosine, directions, gate): _digest_array(digest, item)
    return digest.hexdigest() == bundle.angular_identity
