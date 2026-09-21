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
    if value.ndim != 2:
        raise ValueError("dense visibility must be pixels by patches")
    return value.shape


def _visibility_block(value, start, stop, patches):
    if hasattr(value, "decode_pixels"):
        block = np.empty((stop-start, patches), np.float32)
        for patch in range(patches):
            block[:, patch] = value.decode_pixels(patch, start, stop)
        return block
    return np.asarray(value[start:stop])


def _digest_visibility(digest, value, *, block_pixels=128):
    pixels, patches = _visibility_shape(value)
    digest.update(np.dtype(np.float32).str.encode())
    digest.update(np.asarray((pixels, patches), np.int64).tobytes())
    for start in range(0, pixels, block_pixels):
        digest.update(_visibility_block(value, start, min(start+block_pixels, pixels), patches).tobytes(order="C"))


def complete_identity(sh, vs, vb, solid, sine, cosine, directions, gate, *, block_pixels=128):
    digest = hashlib.sha256(ALGORITHM.encode())
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
    algorithm: str = ALGORITHM

    def __post_init__(self):
        if self.values.dtype != np.float64 or self.values.shape != (self.pixels, 6):
            raise TypeError("invalid moment payload")
        self.values.setflags(write=False)


def build_bundle(sh, vs, vb, solid, sine, cosine, directions, gate, *, block_pixels=128,
                 fail_after_blocks=None):
    """Construct with bounded pixel blocks; no full visibility cube is copied."""
    if block_pixels < 1:
        raise ValueError("block_pixels must be positive")
    pixels, patches = _visibility_shape(sh)
    if _visibility_shape(vs) != (pixels, patches) or _visibility_shape(vb) != (pixels, patches):
        raise ValueError("visibility shapes differ")
    identity = complete_identity(sh, vs, vb, solid, sine, cosine, directions, gate, block_pixels=block_pixels)
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
        return MomentBundle(identity, values, pixels, patches)
    except BaseException:
        # Drop the only payload reference before propagating; owners commit only returns.
        del values
        raise


class TileMomentOwner:
    """Explicit per-tile owner; never process-global and safe for concurrent tiles."""
    def __init__(self):
        self._lock = threading.Lock()
        self._bundles = {}
        self.closed = False

    def build(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError("moment owner is closed")
        bundle = build_bundle(*args, **kwargs)
        with self._lock:
            if self.closed:
                raise RuntimeError("moment owner closed during construction")
            existing = self._bundles.get(bundle.identity)
            if existing is None:
                self._bundles[bundle.identity] = bundle
                return bundle
            return existing

    def close(self):
        with self._lock:
            self._bundles.clear()
            self.closed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    @property
    def allocation_bytes(self):
        with self._lock:
            return sum(bundle.values.nbytes for bundle in self._bundles.values())

    @property
    def count(self):
        with self._lock: return len(self._bundles)
