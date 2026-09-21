# E4 persistent-JIT source packet

This packet is bound to durable baseline commit `8ca23d444a3b05bdeb76655329c0a09c3dc4d6e8` by `baseline_source_manifest.json`. It was authored source-only; no candidate import, JIT compilation, numerical test, or timing was run.

## Candidates

- `wall_candidate`: admits the independent `_wall13_serial` and `_wall23_serial` source definitions for qualification. Their bodies and argument lists are AST-identical to the existing parallel definitions, including `prange`; only the serial definitions use `cache=True`, `fastmath=False`. Parallel dispatchers remain uncached.
- `math_candidate`: separately admits `asvf_fma`, `tan_array`, and `atan_array` with `cache=True`. `asvf_fma` already has its complete custom helper graph in `_sleef_acos.py`. The classifier now owns AST-identical copies of `F`, `fma`, and `dfmul`; all other scalar helpers and both array bodies remain unchanged. This makes each cached entrypoint's custom dependency graph source-local.

The semantic `PROFILE_ID` remains `solweig-portable-sleef-5a1d179d-v1`. The math candidate naturally changes the implementation fingerprint because `_math_profile.SOURCE_FILES` already includes `_sleef_acos.py`.

## Qualification commands (not run)

Run on the reserved numerical host from the repository root:

```sh
E4_BASELINE_SRC="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/baseline/src" \
  .venv-light/bin/python -m pytest -q \
  /tmp/solweig-exact-e4-8s8t7i/tests/test_e4_source_proof.py

E4_BASELINE_SRC="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/baseline/src" \
  .venv-light/bin/python -m pytest -q -s \
  /tmp/solweig-exact-e4-8s8t7i/tests/test_e4_subprocess.py
```

The subprocess tests bind `PYTHONPATH` and verify module origins, compare baseline/candidate raw bytes and masks, exercise both wall call orders, use a fresh process for cold and warm cache checks, inspect distinct cache artifacts, and exercise `asvf`, float32 mixed wide tangent fallback, float64 NumPy fallback, and `atan32` through the public math profile. They do not time anything.

If both candidates qualify, apply them as separate commits or retain the split in the durable experiment record. The duplicated classifier helpers carry a maintenance invariant: their AST must remain identical to the canonical `_sleef_acos.py` definitions. The source-proof test enforces this and must remain with the change.
