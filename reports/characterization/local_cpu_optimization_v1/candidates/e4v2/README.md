# E4v2 persistent-JIT source packet

This packet supersedes E4v1 without modifying it. It is bound to durable baseline commit `8ca23d444a3b05bdeb76655329c0a09c3dc4d6e8`. Authoring was source-only: no candidate import, JIT compilation, numerical test, or timing ran.

## Design

Both candidates add `radiation/_jit_cache.py`. `@bind_cache_identity` is placed below each `@njit(cache=True, ...)`, so it runs on the raw Python function first. It hashes framed bytes for the defining module, the namespace helper, and the Python/NumPy/Numba/llvmlite/platform architecture identity. It changes only `function.__qualname__` and returns the same function; it creates no wrapper and does not alter `__code__`.

Installed Numba 0.67.0 source shows that `CacheImpl` reads `py_func.__qualname__` into `filename_base`, while source freshness otherwise uses only `(mtime, size)`. Old and new namespace filenames therefore coexist in the configured cache. The design does not mutate or delete `NUMBA_CACHE_DIR`. See `numba_source_proof.json` for hashes, anchors, and the remaining execution gate.

- `wall_candidate`: real independent `_wall13_serial` and `_wall23_serial` definitions retain AST-identical arguments and bodies, including `prange`; only serial definitions are cached. Parallel definitions remain uncached.
- `math_candidate`: caches `asvf_fma`, `tan_array`, and `atan_array`. The classifier owns AST-identical copies of canonical `F`, `fma`, and `dfmul`, making its custom dependency graph source-local. All scalar arithmetic and array bodies remain unchanged.
- Both candidates add `_jit_cache.py` to `_math_profile.SOURCE_FILES`. `PROFILE_ID` stays `solweig-portable-sleef-5a1d179d-v1`; the implementation fingerprint changes naturally.

## Qualification commands — not run

From the repository root on the reserved numerical host:

```sh
E4_BASELINE_SRC="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/baseline/src" \
  .venv-light/bin/python -m pytest -q \
  /tmp/solweig-exact-e4v2-YNJ1EA/tests/test_e4_source_proof.py

E4_BASELINE_SRC="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/baseline/src" \
  .venv-light/bin/python -m pytest -q -s \
  /tmp/solweig-exact-e4v2-YNJ1EA/tests/test_e4_subprocess.py

.venv-light/bin/python -m pytest -q \
  tests/unit/test_radiation.py \
  tests/unit/test_portable_profile.py

.venv-light/bin/python -m pytest -q

for e4_source in \
  /tmp/solweig-exact-e4v2-YNJ1EA/wall_candidate/src \
  /tmp/solweig-exact-e4v2-YNJ1EA/math_candidate/src
do
  run_label=$(basename "$(dirname "$e4_source")")
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 NUMBA_NUM_THREADS=1 \
    PYTHONPATH="$e4_source" .venv-light/bin/python tools/capture_exact_pipeline.py \
    --fixture tests/reference/small_original_cpu/scene \
    --source "$e4_source" \
    --profile-tool tools/profile_p7_pipeline.py \
    --run "/tmp/e4v2-exact-trace-${run_label}" \
    --threads 1 --block-pixels 128 --geometry-state cold \
    --capture-writer-outputs --execute
done
```

The subprocess suite checks module origins; raw bytes, dtypes, shapes, NaN/Inf masks, and sign bits; opposite cold/warm wall call orders in the same cache; distinct artifacts; cold-only LLVM FMA inspection; warm cache hits without `inspect_llvm`; public float32 wide/nonfinite and float64 NumPy fallbacks; and same-size/same-mtime defining-source and helper mutations. Each mutation must create a new qualified namespace and cache miss, then match an independent mutated-cold process exactly. The two real-pipeline trace manifests must subsequently be compared exactly with the qualified baseline trace. These are correctness checks and must not be counted as performance evidence.
