# P6 wind resource characterization

All three frozen `benchmarks/protocols/p6_wind_resources_v1.json` cases passed.
Raw measurements, logs, source/protocol/harness hashes, dependency manifests,
original-reference checks and generated completion records are under
[`characterization/p6_wind_resources_v1`](characterization/p6_wind_resources_v1).

| Direction workers | Process elapsed (s) | Peak summed RSS (bytes) | RSS samples |
| --- | ---: | ---: | ---: |
| 1 | 1.950 | 152,797,184 | 61 |
| 2 | 2.008 | 154,992,640 | 62 |
| 4 | 1.558 | 157,073,408 | 48 |

Each process ran the public wind workflow on fresh copies of the original
precedence fixture, with all twelve directions and a 4 GiB memory budget.
Every original field, nonfinite mask, artifact name and TIFF metadata record
passed exact comparison. Completion records confirmed actual admitted worker
counts and maximum pending futures of 1, 2 and 4 respectively. Source hashes
were unchanged throughout. Installed-wheel/source identity was verified by
the final P6 optional gate before characterization.

Elapsed time includes imports, generation and verification. Summed process-tree
RSS was sampled every 20 ms; it can double-count shared pages and miss peaks.
These are single trials on a small fixture, not a speedup comparison or a
universal memory bound. The admission inventory remains an estimate. All native
thread environment settings used by these processes are recorded per case;
the wind implementation also limits GDAL threads per dataset.

Command: `.venv-light/bin/python tools/measure_p6_wind.py --directory reports/characterization/p6_wind_resources_v1`.
