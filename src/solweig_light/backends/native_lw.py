"""Optional native (ISPC) longwave primary backend — B7-40 integration.

Kernel and adapter are the reviewed B7-20 candidate C_native (review record
optimization_v7_backends/evidence/reviews/b7_30_ispc_review.json; selection
record evidence/trials/b7_32_selection.json), copied byte-identical into
``solweig_light/backends/native/``:

  lw_primary.ispc  sha256 52652a598cd85ead04747a83a52b1184035c880aa0b32cf518efef06b961663c
  lw_native.py     sha256 a1ce7547562a44a9548f9c34ccd3c3f0f0c5aaac4ed130975eb894bbe562ddd6
  build.sh         toolchain-pinned ISPC build + assembly FMA audit gate

The shared library is NOT shipped: on first use this module builds it into a
user cache directory (``SOLWEIG_LIGHT_NATIVE_CACHE`` or
``~/.cache/solweig-light/native``), rebuilding automatically when the kernel
source changes. The build script refuses to emit a library whose assembly
contains any fused multiply-add mnemonic, so a contraction-free kernel is a
build-time invariant, not a hope.

Dispatch policy (enforced by the caller in radiation.cylinder_longwave):

* the default path never touches this module;
* ``UnsupportedInput`` (unsupported dtype/shape/stride/scalar provenance) is
  raised BEFORE any native work; the caller falls back to the Numba kernel;
* a requested backend that cannot build fails loudly (RuntimeError with the
  build command) — never a silent fallback;
* the native kernel schedules identically for serial and parallel labels and
  runs on one thread; measured behavior and caveats (multi-threaded runtimes
  on large blocks) are in the selection record.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import solweig_light.backends.native.lw_native as lw_native

_HERE = Path(__file__).resolve().parent
_SRC_DIR = _HERE / 'native'
_KERNEL = _SRC_DIR / 'lw_primary.ispc'

GANG = 8  # neon-i32x8; lane width matched to the reviewed layout control W=8

_ISPC_FALLBACK = '/opt/homebrew/bin/ispc'


def _native_cache_env():
    """The legacy native-cache override, read HERE only.

    N8-41 vendoring made this module the package's single env-read site
    for the variable (the DX env-read surface is frozen
    module-by-module). N9 F4 archived the N8 dispatch machinery that
    used to resolve the same override through this reader (research
    copy: experiments/optimization_v8/native_dispatch/native_handle.py).
    """
    return os.environ.get('SOLWEIG_LIGHT_NATIVE_CACHE')


def _cache_dir():
    env = _native_cache_env()
    base = Path(env) if env else Path.home() / '.cache' / 'solweig-light' / 'native'
    base.mkdir(parents=True, exist_ok=True)
    return base


def _kernel_digest():
    return hashlib.sha256(_KERNEL.read_bytes()).hexdigest()


def _build_needed(cache):
    dylib = cache / f'liblw_native_g{GANG}.dylib'
    if not dylib.exists():
        return True
    stamp = cache / 'build_stamp.json'
    if not stamp.exists():
        return True
    try:
        return json.loads(stamp.read_text()).get('kernel_sha256') != _kernel_digest()
    except (OSError, ValueError):
        return True


def _build(cache):
    """Build the dylibs into the cache dir; the build's own FMA audit gates
    the result. Raises RuntimeError with the exact command on any failure."""
    import time

    ispc = shutil.which('ispc') or _ISPC_FALLBACK
    if not Path(ispc).exists():
        raise RuntimeError(
            'SOLWEIG_LIGHT_LW_BACKEND=native requested but ispc was not found. '
            'Install ISPC (>= 1.31) or build the library manually: '
            f'cd {_SRC_DIR} && ISPC=<ispc-path> zsh build.sh, then copy '
            f'liblw_native_g*dylib into {cache}')
    work = cache / 'build'
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_KERNEL, work / 'lw_primary.ispc')
    shutil.copy2(_SRC_DIR / 'build.sh', work / 'build.sh')
    proc = subprocess.run(['/bin/zsh', 'build.sh'],
                          cwd=work, capture_output=True, text=True,
                          env={**os.environ, 'ISPC': ispc})
    if proc.returncode != 0:
        raise RuntimeError(
            f'native backend build failed (rc={proc.returncode}):\n'
            f'{proc.stdout[-800:]}\n{proc.stderr[-800:]}')
    for gang in (4, GANG):
        built = work / f'liblw_native_g{gang}.dylib'
        if not built.exists():
            raise RuntimeError(f'build reported success but {built} is missing')
        shutil.copy2(built, cache / built.name)
    (cache / 'build_stamp.json').write_text(json.dumps({
        'kernel_sha256': _kernel_digest(),
        'ispc': ispc,
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }))


def _ensure_loaded():
    """Point the reviewed adapter at the cache-built dylibs and pin them."""
    cache = _cache_dir()
    if _build_needed(cache):
        _build(cache)
    if lw_native._LIB_DIR != cache:
        lw_native._LIB_DIR = cache
    lw_native._load(GANG)  # pins/validates the dylib before any timed work


UnsupportedInput = lw_native.UnsupportedInput


def native_longwave_primary(sh, vs, vb, sun, shade, solid, sine, cosine,
                            directions, gate, solar_gate, sky_down, sky_side,
                            surface_sun, surface_sh, lup, reflection_factor):
    """17-argument drop-in for _longwave_primary; returns float32 [B,7].

    Raises UnsupportedInput for inputs outside the reviewed admission domain
    (the caller falls back to the Numba kernel); anything else is a real
    backend failure and propagates.
    """
    _ensure_loaded()
    return lw_native.primary(sh, vs, vb, sun, shade, solid, sine, cosine,
                             directions, gate, solar_gate, sky_down, sky_side,
                             surface_sun, surface_sh, lup, reflection_factor,
                             gang=GANG)


def identity():
    """Runtime fingerprint for tests and diagnostics."""
    cache = _cache_dir()
    stamp = {}
    try:
        stamp = json.loads((cache / 'build_stamp.json').read_text())
    except (OSError, ValueError):
        pass
    return {
        'backend': 'native_ispc_lw_primary',
        'gang': GANG,
        'kernel_sha256': _kernel_digest(),
        'adapter_sha256': hashlib.sha256(
            (_SRC_DIR / 'lw_native.py').read_bytes()).hexdigest(),
        'cache_dir': str(cache),
        'build_stamp': stamp,
        'effective_threads': 1,
    }


__all__ = ['native_longwave_primary', 'UnsupportedInput', 'identity', 'GANG']
