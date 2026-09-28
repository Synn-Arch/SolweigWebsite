# Private demand-specific radiation entry points (optimization v6, task C6-20).
#
# Dossier dossiers/03_demand_specific_radiation.md rule R-A: in
# ``Lside_veg_v2022a`` the anisotropic branch (``anisotropic_longwave == 1``)
# returns exactly ``_operate(np.multiply, LupD, 0.5)`` for the four cardinal
# directions.  The SVF angle transforms, the Lvikt polynomial, the wall terms
# and the all-sky radiance do not feed the returned tuple.  This module owns a
# PRIVATE demand profile for the pipeline route and a guarded fast path that
# computes only the demanded operations with bitwise equality to the original.
#
# Contract (DESIGN_AUTHORITY.md #6, dossiers/09, dossiers/10):
#   * The public function ``Lside_veg_v2022a`` stays untouched and is used
#     verbatim for FULL_DIAGNOSTICS and for every input outside the proved
#     admitted domain (original fallback).
#   * Omitted diagnostics are ``NOT_REQUESTED`` sentinels in the demand
#     profile, never zero arrays.
#   * Warning/failure behavior: inside the admitted domain the only floating
#     point effect of the removed operations is the ``log(1 - svf)``
#     divide-by-zero at pixels where ``svf == 1`` (real SVF rasters contain
#     such pixels).  The fast path re-executes exactly that original
#     ``subtract``+``log`` pair per affected direction so the warning class,
#     message and per-call count match the original; if it raises (warning
#     filters promoting to errors) the original function is called so the
#     raised error comes from the original site.  All other error regimes
#     (np.seterr non-default, strict warning filters) fall back to the
#     original function.
"""Private pipeline radiation demand profiles and the anisotropic Lside path."""
from __future__ import annotations

import enum
import math
import threading

import numpy as np
from numba import njit

from ._math_profile import PROFILE_ID

__all__ = [
    "RadiationDemand",
    "NOT_REQUESTED",
    "LSIDE_ANISOTROPIC_DEMAND_PROFILE",
    "lside_veg_v2022a_demanded",
    "current_demand",
    "radiation_demand",
    "demand_identity",
]


class _NotRequested:
    """Singleton marker for a diagnostic omitted by the demand contract."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):  # pragma: no cover - trivial marker
        return "<not_requested>"

    def __reduce__(self):
        return (_NotRequested, ())


NOT_REQUESTED = _NotRequested()


class RadiationDemand(enum.Enum):
    """Private radiation demand profiles (dossier 09 interface #3).

    ``FULL_DIAGNOSTICS`` is the public behavior: every diagnostic the original
    function produces is produced.  ``PIPELINE_CYLINDER_ANISOTROPIC`` is the
    admitted pipeline profile: the four demanded cardinal results are produced
    exactly; everything outside their backward slice is ``NOT_REQUESTED``.
    """

    FULL_DIAGNOSTICS = "full_diagnostics"
    PIPELINE_CYLINDER_ANISOTROPIC = "pipeline_cylinder_anisotropic"


# Backward-slice record for the anisotropic Lside result (R-A).  The function
# returns (Least, Lsouth, Lwest, Lnorth); every other quantity the original
# evaluates (svfalfa*, vikt*, Lsky_allsky, azi*, Lwallsun, Lwallsh, and Ldown
# which the anisotropic branch never reads) is not requested.
LSIDE_ANISOTROPIC_DEMAND_PROFILE = {
    "Least": "demanded",
    "Lsouth": "demanded",
    "Lwest": "demanded",
    "Lnorth": "demanded",
    "svfalfaE": NOT_REQUESTED,
    "svfalfaS": NOT_REQUESTED,
    "svfalfaW": NOT_REQUESTED,
    "svfalfaN": NOT_REQUESTED,
    "viktveg": NOT_REQUESTED,
    "viktwall": NOT_REQUESTED,
    "viktsky": NOT_REQUESTED,
    "viktrefl": NOT_REQUESTED,
    "Lsky_allsky": NOT_REQUESTED,
    "aziE": NOT_REQUESTED,
    "aziS": NOT_REQUESTED,
    "aziW": NOT_REQUESTED,
    "aziN": NOT_REQUESTED,
    "Lwallsun": NOT_REQUESTED,
    "Lwallsh": NOT_REQUESTED,
    "Lsky": NOT_REQUESTED,
    "Lveg": NOT_REQUESTED,
    "Lrefl": NOT_REQUESTED,
    "Ldown": NOT_REQUESTED,
}


def demand_identity():
    """Return the math-profile provenance of this private path."""
    return {
        "math_profile": PROFILE_ID,
        "fastmath": False,
        "reduction_rule": "R-A",
        "demand_profiles": sorted(member.value for member in RadiationDemand),
    }


# ---------------------------------------------------------------------------
# Admitted-domain envelope (see evidence/lside/README.md for the derivation).
# Values inside this envelope are proved to keep every removed operation free
# of overflow/invalid floating point effects in float32 and float64, with
# >=1000x magnitude margin to the float32 overflow threshold.
# ---------------------------------------------------------------------------
_ERR_DEFAULT = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
_SVF_HI = 1.0          # svfE/S/W/N in [0, 1] (closed: svf==1 handled by replication)
_VEG_BOUND = 100.0     # svf*veg / svf*aveg magnitude bound
_MAG_BOUND = 1000.0    # Ta / Tw / F_sh / CI magnitude bound
_UNIT_BOUND = 100.0    # SBC / ewall / esky magnitude bound
_ANGLE_BOUND = 1.0e6   # azimuth / t magnitude bound
_FAST_DTYPES = (np.float32, np.float64)


@njit(cache=True, nogil=True, fastmath=False)
def _minmax_1d(a):
    """Single fused min/max pass; NaN short-circuits to (nan, nan).

    Empty input returns (nan, nan) explicitly: every later bound comparison
    fails, so empty rasters fall back to the original function deterministically
    (no placeholder out-of-bounds element read).
    """
    if a.size == 0:
        return math.nan, math.nan
    lo = a[0]
    hi = a[0]
    for i in range(a.size):
        v = a[i]
        if v != v:
            return math.nan, math.nan
        if v < lo:
            lo = v
        elif v > hi:
            hi = v
    return lo, hi


def _array_range(x):
    """(min, max) of a bounded check over array-like ``x`` without FP arithmetic.

    Only comparisons run here, so the guard itself can never emit a floating
    point warning.  NaN propagates so every later bound comparison fails.
    """
    a = np.asarray(x)
    if a.dtype in _FAST_DTYPES:
        lo, hi = _minmax_1d(a.ravel())
        return lo, hi
    lo = np.amin(a)
    hi = np.amax(a)
    return lo, hi


def _bounded(x, lo_bound, hi_bound):
    """True when scalar-or-array ``x`` lies elementwise in [lo_bound, hi_bound]."""
    if np.ndim(x) == 0:
        v = float(x)
        return math.isfinite(v) and lo_bound <= v <= hi_bound
    lo, hi = _array_range(x)
    return bool(lo >= lo_bound and hi <= hi_bound)


def _shape(x):
    return np.asarray(x).shape


def _broadcastable(shapes):
    try:
        np.broadcast_shapes(*shapes)
    except ValueError:
        return False
    return True


class _GuardResult:
    __slots__ = ("ok", "log_dirs")

    def __init__(self, ok, log_dirs=()):
        self.ok = ok
        self.log_dirs = log_dirs


def _guard_anisotropic(args):
    """Certify the removed-operation domain for the anisotropic Lside slice.

    Returns ``_GuardResult(ok, log_dirs)``.  ``log_dirs`` lists the cardinal
    indices (0=E, 1=S, 2=W, 3=N) whose plain svf contains exact 1.0 pixels;
    those are the only admitted-domain positions where a removed operation
    (``log(1 - svf)``) can emit a floating point warning, and the fast path
    re-executes that original operation for them.
    """
    try:
        if np.geterr() != _ERR_DEFAULT:
            return _GuardResult(False)
        (
            svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
            svfEaveg, svfSaveg, svfWaveg, svfNaveg,
            azimuth, altitude, Ta, Tw, SBC, ewall, Ldown, esky, t, F_sh, CI,
            LupE, LupS, LupW, LupN, anisotropic_longwave,
        ) = args
        # Ldown and Lup* are deliberately unguarded: the anisotropic branch
        # never reads Ldown, and the demanded multiply is the identical
        # original operation for every possible Lup value.
        if np.ndim(altitude) != 0 or np.ndim(anisotropic_longwave) != 0:
            return _GuardResult(False)
        if not bool(np.asarray(anisotropic_longwave) == 1):
            return _GuardResult(False)
        # The original converts the scalars through _array(..., float32); a
        # magnitude beyond float32 range would warn there (removed op).
        if not _bounded(altitude, -_ANGLE_BOUND, _ANGLE_BOUND):
            return _GuardResult(False)
        alt_positive = bool(np.asarray(altitude) > 0)
        # The original mixes azimuth/t in the prologue azi* sums
        # (engine.py:1375-1378) for BOTH altitude regimes, and the day branch
        # additionally compares them against 0-d scalars.  Array azimuth/t can
        # overflow-warn there (e.g. float32 values beyond 3.4e38 in the add),
        # a removed-operation warning effect, so only 0-d scalars are admitted
        # in either regime; arrays fall back to the original function.
        if np.ndim(azimuth) != 0 or np.ndim(t) != 0:
            return _GuardResult(False)
        if np.ndim(azimuth) == 0 and not _bounded(azimuth, -_ANGLE_BOUND, _ANGLE_BOUND):
            return _GuardResult(False)
        if np.ndim(t) == 0 and not _bounded(t, -_ANGLE_BOUND, _ANGLE_BOUND):
            return _GuardResult(False)
        if not _bounded(Ta, -_MAG_BOUND, _MAG_BOUND):
            return _GuardResult(False)
        if not _bounded(CI, -_MAG_BOUND, _MAG_BOUND):
            return _GuardResult(False)
        if not _bounded(SBC, -_UNIT_BOUND, _UNIT_BOUND):
            return _GuardResult(False)
        if not _bounded(ewall, -_UNIT_BOUND, _UNIT_BOUND):
            return _GuardResult(False)
        if not _bounded(esky, -_UNIT_BOUND, _UNIT_BOUND):
            return _GuardResult(False)
        if not _bounded(Tw, -_MAG_BOUND, _MAG_BOUND):
            return _GuardResult(False)
        if not _bounded(F_sh, -_MAG_BOUND, _MAG_BOUND):
            return _GuardResult(False)
        ta_shape = _shape(Ta)
        sbc_shape = _shape(SBC)
        ewall_shape = _shape(ewall)
        if not _broadcastable((_shape(esky), sbc_shape, ta_shape, _shape(CI))):
            return _GuardResult(False)
        svf_plain = (svfE, svfS, svfW, svfN)
        svf_veg = (svfEveg, svfSveg, svfWveg, svfNveg)
        svf_aveg = (svfEaveg, svfSaveg, svfWaveg, svfNaveg)
        log_dirs = []
        day_shapes = None
        if alt_positive:
            day_shapes = (_shape(Tw), _shape(F_sh))
        for index in range(4):
            plain, veg, aveg = svf_plain[index], svf_veg[index], svf_aveg[index]
            lo, hi = _array_range(plain)
            if not bool(lo >= 0.0 and hi <= _SVF_HI):
                return _GuardResult(False)
            if hi == _SVF_HI:
                log_dirs.append(index)
            lo, hi = _array_range(veg)
            if not bool(lo >= 0.0 and hi <= _VEG_BOUND):
                return _GuardResult(False)
            lo, hi = _array_range(aveg)
            if not bool(lo >= 0.0 and hi <= _VEG_BOUND):
                return _GuardResult(False)
            shapes = (_shape(plain), _shape(veg), _shape(aveg), ta_shape, sbc_shape, ewall_shape)
            if alt_positive:
                shapes = shapes + day_shapes
            if not _broadcastable(shapes):
                return _GuardResult(False)
        return _GuardResult(True, tuple(log_dirs))
    except Exception:
        # Guard evaluation itself hit an unsupported input domain (object
        # arrays, strings, exotic dtypes, ...).  The original function owns
        # whatever warning or exception that domain produces.
        return _GuardResult(False)


def _replicate_log_warnings(svf_plain, log_dirs):
    """Re-execute the original ``log(1 - svf)`` for directions with svf == 1.

    Same ufunc, same input values, same np.seterr regime as the removed
    original chain, so the emitted warning category, message and per-call
    count match.  Any exception (warning filter promoting to error) propagates
    to the caller which falls back to the original function so the raised
    error originates from the original site.
    """
    from .engine import _operate

    for index in log_dirs:
        one_minus = _operate(np.subtract, 1, svf_plain[index])
        np.log(one_minus)


def _demanded_anisotropic_result(LupE, LupS, LupW, LupN):
    """The exact demanded operations of rule R-A, in original return order."""
    from .engine import _operate

    return (
        _operate(np.multiply, LupE, 0.5),
        _operate(np.multiply, LupS, 0.5),
        _operate(np.multiply, LupW, 0.5),
        _operate(np.multiply, LupN, 0.5),
    )


def lside_veg_v2022a_demanded(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                              svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta,
                              Tw, SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW,
                              LupN, anisotropic_longwave, demand=None):
    """Demand-dispatching drop-in for ``Lside_veg_v2022a``.

    ``FULL_DIAGNOSTICS`` (default) calls the untouched original function.
    ``PIPELINE_CYLINDER_ANISOTROPIC`` takes the guarded R-A fast path when the
    admitted domain holds and otherwise falls back to the original function,
    preserving every warning, failure and alias contract bit-for-bit.
    """
    from .engine import Lside_veg_v2022a

    if demand is None:
        demand = current_demand()
    if demand is not RadiationDemand.PIPELINE_CYLINDER_ANISOTROPIC:
        return Lside_veg_v2022a(
            svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
            svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta,
            Tw, SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW,
            LupN, anisotropic_longwave)
    args = (
        svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
        svfEaveg, svfSaveg, svfWaveg, svfNaveg,
        azimuth, altitude, Ta, Tw, SBC, ewall, Ldown, esky, t, F_sh, CI,
        LupE, LupS, LupW, LupN, anisotropic_longwave,
    )
    guard = _guard_anisotropic(args)
    if guard.ok:
        svf_plain = (svfE, svfS, svfW, svfN)
        try:
            _replicate_log_warnings(svf_plain, guard.log_dirs)
        except Exception:
            # A warning filter promoted the replicated warning to an error.
            # Re-run the original so the identical error is raised from the
            # original site.
            return Lside_veg_v2022a(*args)
        return _demanded_anisotropic_result(LupE, LupS, LupW, LupN)
    return Lside_veg_v2022a(*args)


# ---------------------------------------------------------------------------
# Private demand scope for the pipeline driver (integrator-owned plumbing).
# ---------------------------------------------------------------------------
_demand_state = threading.local()


def current_demand():
    """The RadiationDemand active for this thread; FULL_DIAGNOSTICS by default."""
    return getattr(_demand_state, "demand", None) or RadiationDemand.FULL_DIAGNOSTICS


class radiation_demand:
    """Context manager selecting the private radiation demand for this thread."""

    def __init__(self, demand):
        if not isinstance(demand, RadiationDemand):
            raise TypeError("demand must be a RadiationDemand member")
        self._demand = demand
        self._previous = None

    def __enter__(self):
        self._previous = getattr(_demand_state, "demand", None)
        _demand_state.demand = self._demand
        return self._demand

    def __exit__(self, exc_type, exc, tb):
        _demand_state.demand = self._previous
        return False
