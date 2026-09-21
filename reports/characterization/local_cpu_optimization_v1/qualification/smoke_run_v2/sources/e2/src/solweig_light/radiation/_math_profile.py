"""Versioned portable math profile used only by the isolated admission snapshot."""
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np

from ._sleef_acos import asvf_fma
from ._sleef_classifier import atan_array, tan_array

PROFILE_ID = "solweig-portable-sleef-5a1d179d-v1"
PROFILE = PROFILE_ID  # Compatibility with the earlier isolated identity hook.
SLEEF_COMMIT = "5a1d179df9cf652951b59010a2d2075372d67f68"
COEFFICIENT_POLICY = "numpy-float64-cos-tan-scalar-v1"
TAN_DOMAIN_POLICY = "sleef-u10-fast-abs-lt-125;explicit-legacy-numpy-otherwise-v1"
SOURCE_FILES = ("_sleef_acos.py", "_sleef_classifier.py", "_math_profile.py", "SLEEF_LICENSE.txt")


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _source_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def profile_identity():
    """Return the semantic ID and source/runtime fingerprint used by caches."""
    source = _source_hashes()
    implementation = {
        "profile_id": PROFILE_ID,
        "sleef_commit": SLEEF_COMMIT,
        "coefficient_policy": COEFFICIENT_POLICY,
        "tan_domain_policy": TAN_DOMAIN_POLICY,
        "source_sha256": source,
    }
    runtime = {
        "python": platform.python_implementation() + "-" + platform.python_version(),
        "numpy": _package_version("numpy"),
        "numba": _package_version("numba"),
        "llvmlite": _package_version("llvmlite"),
        "machine": platform.machine(),
        "system": platform.system(),
        "byteorder": sys.byteorder,
        "float_rounds": int(sys.float_info.rounds),
        "fastmath": False,
        "fma": "explicit-llvm.fma.f32",
    }
    payload = json.dumps({"implementation": implementation, "runtime": runtime}, sort_keys=True, separators=(",", ":")).encode()
    return {"id": PROFILE_ID, "fingerprint": hashlib.sha256(payload).hexdigest(), "implementation": implementation, "runtime": runtime}


def asvf(values):
    """Compute acos(sqrt(SVF)) while preserving accepted dtype/domain behavior."""
    values = np.asarray(values)
    if values.dtype == np.float32:
        return asvf_fma(values.ravel()).reshape(values.shape)
    # Explicit preserved legacy path for supported promoted inputs.
    return np.arccos(np.sqrt(values))


def tan32(values):
    """Float32 profile tangent; other dtypes preserve their prior NumPy path.

    NumPy owns the complete legacy-domain operation so its vector/layout/tail
    dispatch is preserved.  The source-derived SLEEF result replaces only the
    validated finite ``abs(x) < 125`` positions.
    """
    values = np.asarray(values)
    if values.dtype == np.float32:
        fast = np.isfinite(values) & (np.abs(values) < np.float32(125.0))
        if np.all(fast):
            return tan_array(values.ravel()).reshape(values.shape)
        result = np.asarray(np.tan(values))
        result[fast] = tan_array(values[fast].ravel())
        return result
    return np.tan(values)


def atan32(values):
    """Float32 profile arctangent; other dtypes preserve their prior NumPy path."""
    values = np.asarray(values)
    if values.dtype == np.float32:
        return atan_array(values.ravel()).reshape(values.shape)
    return np.arctan(values)
