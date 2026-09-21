# Original upstream CPU ground-view diagnostic

This diagnostic is separate from benchmarking. It uses the same 9×11 synthetic
fixture and both land-cover modes from the queued independent UMEP test. No
production source, test tolerance or fixture value was changed.

The task-owned checkout is pinned at
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec` and is clean. Its installed `.py`
and data files match the checkout hashes. The isolated environment contains no
candidate namespace and uses Python 3.12.14, GDAL 3.13.3 and
`torch 2.14.0+cpu`. The actual Conda and pip locks are preserved in `results/`.

Original Torch and candidate execution were performed in separate Python
environments and serialized independently. For both `landcover=0` and
`landcover=1`:

- all 17 outputs pass the unchanged per-field policy;
- `gvfSum` (field 15) and `gvfNorm` (field 16) are bitwise equal float32 arrays;
- every non-longwave output is bitwise equal;
- longwave fields 0, 3, 6, 9 and 12 differ by at most one float32-scale step,
  `3.0517578125e-05`, within their predeclared `rtol=1e-5, atol=0.05` gate; and
- observable `Tg` mutation is exact.

This distinguishes the independent UMEP failure from a candidate compatibility
regression. Unchanged UMEP NumPy returns float64 `gvfSum` and differs from both
the original Torch float32 implementation and candidate by at most
`1.7106533043431682e-06`, crossing the authored `1e-6` independent-component
allowance. Candidate `gvfSum` exactly preserves the pinned upstream result. The
independent UMEP gate remains failed as authored; this diagnostic does not relax
it or establish which arithmetic should govern a future scientific-correction
mode.

Evidence:

- `results/upstream_cpu_install_verification.json`
- `results/upstream_cpu_{environment,conda_explicit,pip_freeze,git_commit,git_status,source_sha256}.txt`
- `results/ground_view_original_torch_cpu.npz`
- `results/ground_view_candidate_cpu.npz`
- `results/ground_view_original_torch_comparison.json`
- `results/ground_view_diagnostic_scripts_sha256.txt`
- `results/result_sha256.txt`

All downloaded result artifacts verify against the refreshed result manifest.
The remote checkout and environment remain inside the cleanup-ledger root for
review and later authorized follow-up.
