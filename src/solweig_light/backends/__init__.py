"""Optional CPU backends for SOLWEIG-light kernels.

The default execution path is unchanged Numba. A backend is active only when
explicitly requested (``SOLWEIG_LIGHT_LW_BACKEND=native``); unsupported
inputs fall back to the Numba reference kernel BEFORE the backend launches,
and a requested-but-unavailable backend fails with an explicit error rather
than silently falling back.
"""
