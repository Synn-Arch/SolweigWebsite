"""Source and runtime namespace for persistent Numba entrypoint caches."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import struct
import sys

import llvmlite
import numba
import numpy as np

# cache-namespace-v1

def _framed(hasher, label, value):
    label = label.encode("utf-8")
    value = bytes(value)
    hasher.update(len(label).to_bytes(8, "big"))
    hasher.update(label)
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def bind_cache_identity(function):
    """Namespace Numba's cache without wrapping or changing function code."""
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": list(sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
    }
    hasher = hashlib.sha256()
    _framed(hasher, "defining_module", Path(function.__code__.co_filename).read_bytes())
    _framed(hasher, "namespace_helper", Path(__file__).read_bytes())
    _framed(
        hasher,
        "runtime",
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    function.__qualname__ = function.__qualname__ + "__jitcache_" + hasher.hexdigest()
    return function
