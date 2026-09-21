# E4v3 persistent-JIT source packet

This packet supersedes the preserved E4v2 packet. It is bound to durable baseline commit `8ca23d444a3b05bdeb76655329c0a09c3dc4d6e8`. E4v3 authoring was source-only: no candidate import, JIT compilation, numerical test, or timing ran.

## Design

Both candidates add `radiation/_jit_cache.py`. `@bind_cache_identity` is placed below each `@njit(cache=True, ...)`, so it runs on the raw Python function first. It hashes framed bytes for the defining module, namespace helper, and Python/NumPy/Numba/llvmlite/platform architecture identity. It changes only `function.__qualname__`, returns the same function, creates no wrapper, and does not alter `__code__`.

Installed Numba 0.67.0 source shows that `CacheImpl` reads `py_func.__qualname__` into `filename_base`, while source freshness otherwise uses only `(mtime, size)`. Old and new namespace filenames therefore coexist in the configured cache. The design does not mutate or delete `NUMBA_CACHE_DIR`. See `numba_source_proof.json` for source hashes, anchors, and the remaining execution gate.

- `wall_candidate`: independent `_wall13_serial` and `_wall23_serial` definitions retain AST-identical arguments and bodies, including `prange`; only serial definitions are cached. Parallel definitions remain uncached.
- `math_candidate`: caches `asvf_fma`, `tan_array`, and `atan_array`. The classifier owns AST-identical copies of canonical `F`, `fma`, and `dfmul`, including identical decorators. All scalar arithmetic and array bodies remain unchanged.
- Both candidates add `_jit_cache.py` to `_math_profile.SOURCE_FILES`. `PROFILE_ID` stays `solweig-portable-sleef-5a1d179d-v1`; the implementation fingerprint changes naturally.

## Qualification commands — not run

Run from the repository root on the reserved numerical host. Every repository-suite invocation binds one candidate through `PYTHONPATH`, imports the package and relevant numerical modules in the same Python process, verifies their origins, and only then calls `pytest.main`.

```sh
e4_packet="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/e4v3"
e4_baseline="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/baseline/src"

E4_BASELINE_SRC="$e4_baseline" \
  .venv-light/bin/python -m pytest -q \
  "$e4_packet/tests/test_e4_source_proof.py"

E4_BASELINE_SRC="$e4_baseline" \
  .venv-light/bin/python -m pytest -q -s \
  "$e4_packet/tests/test_e4_subprocess.py"

for e4_source in \
  "$e4_packet/wall_candidate/src" \
  "$e4_packet/math_candidate/src"
do
  run_label=$(basename "$(dirname "$e4_source")")
  E4_SOURCE="$e4_source" PYTHONPATH="$e4_source" \
    NUMBA_CACHE_DIR="/tmp/e4v3-repo-cache-${run_label}-targeted" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=10 \
    .venv-light/bin/python -c '
import os
from pathlib import Path
import sys
import pytest
import solweig_light
from solweig_light.radiation import _math_profile, _sleef_acos, _sleef_classifier, wall_shadows
expected = Path(os.environ["E4_SOURCE"]).resolve()
for module in (solweig_light, _math_profile, _sleef_acos, _sleef_classifier, wall_shadows):
    actual = Path(module.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)
raise SystemExit(pytest.main(sys.argv[1:]))
' -q tests/differential/test_wall_shadows_compiled.py tests/unit/test_portable_profile.py tests/unit/test_persistent_jit_cache.py

done

for e4_source in \
  "$e4_packet/wall_candidate/src" \
  "$e4_packet/math_candidate/src"
do
  run_label=$(basename "$(dirname "$e4_source")")
  E4_SOURCE="$e4_source" PYTHONPATH="$e4_source" \
    NUMBA_CACHE_DIR="/tmp/e4v3-repo-cache-${run_label}-full" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=10 \
    .venv-light/bin/python -c '
import os
from pathlib import Path
import sys
import pytest
import solweig_light
from solweig_light.radiation import _math_profile, _sleef_acos, _sleef_classifier, wall_shadows
expected = Path(os.environ["E4_SOURCE"]).resolve()
for module in (solweig_light, _math_profile, _sleef_acos, _sleef_classifier, wall_shadows):
    actual = Path(module.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)
raise SystemExit(pytest.main(sys.argv[1:]))
' -q

done

for e4_source in \
  "$e4_packet/wall_candidate/src" \
  "$e4_packet/math_candidate/src"
do
  run_label=$(basename "$(dirname "$e4_source")")
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=1 \
    PYTHONPATH="$e4_source" .venv-light/bin/python tools/capture_exact_pipeline.py \
    --fixture tests/reference/small_original_cpu/scene \
    --source "$e4_source" \
    --profile-tool tools/profile_p7_pipeline.py \
    --run "/tmp/e4v3-exact-trace-${run_label}" \
    --threads 1 --block-pixels 128 --geometry-state cold \
    --capture-writer-outputs --execute
done
```

The subprocess suite checks module origins; raw bytes, dtypes, shapes, NaN/Inf masks, and sign bits; opposite cold/warm wall call orders in one cache; cold-only LLVM inspection; warm cache hits without `inspect_llvm`; and public NumPy fallback paths. Same-size/restored-mtime mutations cover the wall module, `_sleef_acos.py`, `_sleef_classifier.py`, and the shared namespace helper. The math cases require the precise affected entrypoint set to receive new namespaces and cache misses, retain old and new artifacts, match an independent mutated-cold process exactly, and change the source/runtime fingerprint. Real-pipeline trace manifests must then compare exactly with the qualified baseline trace. None of these correctness runs are performance evidence.
