# Combined-v1 permanent regression staging

These files are draft permanent regressions for the reviewed `combined_v1` source inventory `dc6e151cfcd6c8bd476921e850666d22f6dfcd2237eb8b0afa649736002af0c6`. They remain under characterization until executable qualification and final review. No staged file was imported or executed while authored.

`manifest.json` maps every staged file to its intended repository destination. The tests resolve the active `solweig_light` package from the pytest process. They contain no report-candidate source paths and no old baseline-packet fingerprint comparisons.

## Exact future qualification command

Run from the repository root when the numerical host is released. This binds the reviewed combined source, asserts all affected module origins in the same process, and runs the staged tests plus the established numerical suites. The cache directory is disposable and outside the candidate source.

```sh
integration_stage="$PWD/reports/characterization/local_cpu_optimization_v1/integration_tests"
combined_source="$PWD/reports/characterization/local_cpu_optimization_v1/candidates/combined_v1/src"
integration_cache=$(mktemp -d /tmp/solweig-combined-integration-cache-XXXXXX)
trap 'rm -rf "$integration_cache"' EXIT

CANDIDATE_SOURCE="$combined_source" \
PYTHONPATH="$combined_source" \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR="$integration_cache" \
.venv-light/bin/python -c '
import os
from pathlib import Path
import sys
import pytest
import solweig_light
from solweig_light.geometry import visibility_compiled
from solweig_light.radiation import (
    _jit_cache, _math_profile, _sleef_acos, _sleef_classifier,
    engine, ground_view, patch_radiation, wall_shadows,
)
expected = Path(os.environ["CANDIDATE_SOURCE"]).resolve()
for module in (
    solweig_light, visibility_compiled, _jit_cache, _math_profile,
    _sleef_acos, _sleef_classifier, engine, ground_view,
    patch_radiation, wall_shadows,
):
    actual = Path(module.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)
raise SystemExit(pytest.main(sys.argv[1:]))
' -q \
  "$integration_stage/tests/differential/test_ground_view_dispatch_exact.py" \
  "$integration_stage/tests/unit/test_duplicate_visibility_decode.py" \
  "$integration_stage/tests/unit/test_exact_persistent_jit_cache.py" \
  tests/differential/test_ground_view_compiled.py \
  tests/differential/test_p7_ground_view_angular.py \
  tests/unit/test_visibility_blocks.py \
  tests/differential/test_patch_radiation.py \
  tests/differential/test_wall_shadows_compiled.py \
  tests/unit/test_portable_profile.py
```

The E4 subprocess test finds its probe beside the staged test tree, so this command is executable before promotion. No result from it is performance evidence.

## Intended promotion and permanent command

After the combined implementation is admitted and review authorizes test promotion:

```sh
integration_stage="$PWD/reports/characterization/local_cpu_optimization_v1/integration_tests"
cp "$integration_stage/tests/differential/test_ground_view_dispatch_exact.py" \
  tests/differential/test_ground_view_dispatch_exact.py
cp "$integration_stage/tests/unit/test_duplicate_visibility_decode.py" \
  tests/unit/test_duplicate_visibility_decode.py
cp "$integration_stage/tests/unit/test_exact_persistent_jit_cache.py" \
  tests/unit/test_exact_persistent_jit_cache.py
cp "$integration_stage/tests/helpers/exact_persistent_jit_cache_probe.py" \
  tests/helpers/exact_persistent_jit_cache_probe.py

permanent_cache=$(mktemp -d /tmp/solweig-permanent-regression-cache-XXXXXX)
trap 'rm -rf "$permanent_cache"' EXIT
PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR="$permanent_cache" .venv-light/bin/python -m pytest -q \
  tests/differential/test_ground_view_dispatch_exact.py \
  tests/unit/test_duplicate_visibility_decode.py \
  tests/unit/test_exact_persistent_jit_cache.py
```

## Coverage and limits

- E1 uses all ten original GVF case fixtures at requested budgets 1, 4, and 10. Each request is capped to the smaller host CPU limit and Numba thread limit, so low-CPU CI runs without a skip or invalid `RuntimeOptions`. It compares all 17 returned fields, every mutated input, and both recorded error classes exactly. A host exposing only one thread cannot establish the multi-thread branch; higher-budget qualification remains required on the reserved host.
- E2 retains the accepted v2 repeated-decode, fallback, mapped-owner, closed-owner, subclass, range, and error-precedence logic. Only an installed-origin assertion was added.
- E4 uses the active distribution and fresh subprocesses. It checks independent serial/parallel identities, opposite call orders, cold saves and warm loads, same-size/restored-mtime invalidation from owned source copies, all three math entrypoints, FMA lowering, NumPy fallbacks, raw special bits, and the current profile identity. Warm cached processes never call `inspect_llvm`.
- The retained upstream/reference and subsystem suites remain the numerical oracle. These permanent tests do not rerun historical isolated baseline-packet comparisons and do not claim a universal proof from finite cases.
