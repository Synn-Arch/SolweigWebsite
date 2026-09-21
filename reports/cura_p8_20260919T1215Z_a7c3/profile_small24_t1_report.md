# Cura candidate small-24h cProfile attribution

Evidence class: one-thread candidate source-run diagnostic on the Cura Xeon and
isolated Conda environment. This is not the M1 profile, an installed-wheel
validation, a paired timing, or a benchmark.

The existing `tools/profile_p7_pipeline.py` ran unchanged with the genuine
32×35 original-CPU scene, all 24 meteorological rows, default 153 patches and all
ten output flags. It completed with status `profile_completed_not_a_benchmark`.
All ten expected GeoTIFFs contain 24 timestamped 32×35 bands; all three cold SVF
artifacts were published. Fixture and candidate-source inventories remained
unchanged. The cProfile-instrumented process took 17.733 seconds, a perturbed
diagnostic wall time.

## Attribution

Startup/JIT dominates this cold source run. Numba compile-lock paths account for
15.524 seconds cumulative, with `compile_extra` at 15.477 seconds. These paths
overlap and must not be added. The largest global self costs are Numba IR variable
discovery (`1.624 s`), llvmlite FFI calls (`1.259 s`), Python `isinstance`
(`0.781 s`), filesystem `fsync` (`0.296 s`), and Numba use/def analysis
(`0.260 s`).

The largest candidate call-site cumulative costs are:

| Call site | Calls | Cumulative seconds | Self seconds |
|---|---:|---:|---:|
| `pipeline._run_tile` | 1 | 15.318 | 0.0028 |
| `comfort.utci_calculator_uniform` | 24 | 7.059 | 0.0027 |
| `radiation.engine.Solweig_2022a_calc` | 24 | 6.801 | 0.0046 |
| `patch_radiation.Lcyl_v2022a` | 24 | 3.800 | 0.0018 |
| `patch_radiation.define_patch_characteristics` | 24 | 3.786 | 0.0544 |
| `visibility_compiled.decode_block` | 1,152 | 1.202 | 0.0528 |
| `patch_radiation.Kside_veg_v2022a` | 14 | 1.059 | 0.0212 |
| `ground_view._gvf` | 14 | 0.965 | 0.0028 |
| `cache.geometry.get_or_create` | 1 | 0.947 | 0.00004 |
| `pipeline.produce_geometry` | 1 | 0.919 | 0.00001 |

These cumulative values include first-call compilation charged to Python call
sites. Native/Numba kernel instructions are opaque to cProfile, so the table
cannot separate steady-state physics from compilation. It supports the bounded
claim that startup/JIT is the primary attributed cost in this specific small cold
run; it cannot rank warm physics kernels reliably or support speedup, regression,
scalability, memory, or larger-scene conclusions.

`stderr.log` contains two divide-by-zero warnings from `log(1-svfE)` and one
invalid-divide warning from the compatibility arithmetic. They did not prevent
complete, validated outputs and are retained rather than suppressed.

The downloaded directory contains raw `worker.pstats`, the complete cumulative
table, function JSON, provenance, exact job/options, stdout/stderr, outcome with
output validation and immutable-input guards, and `evidence_sha256.txt`. Every
listed artifact passed checksum verification.
