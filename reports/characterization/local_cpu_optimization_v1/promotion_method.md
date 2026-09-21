# Local CPU promotion method

This method is declared before reading the new development timings. It
specifies how a later source-bound promotion protocol will be constructed;
it does not freeze a candidate or authorize promotion by itself.

## Selection and scope

The three-pair development matrix is diagnostic. A candidate with median
paired baseline/candidate elapsed time at least 1.03 may advance if every
correctness gate passes. Do not pool different thread budgets, block sizes,
fixtures or cache regimes. Independent admitted changes may be combined,
but their gains must be measured together rather than multiplied.

For any one fixed candidate source, compare the median absolute candidate
elapsed time of its tested joint (native budget, block size) settings on the
warm development fixture. Let m be the minimum of those medians. Among
settings with median <= 1.03 * m, select the smallest native budget, then the
smallest block size. This is one joint selection, not a sequence of pairwise
ties. Do not compare absolute times from different source variants to infer
an unmeasured combined setting: a combined source needs its own predeclared
development setting comparison if its budget/block choice is not already
fixed by the selected component experiment. Record the actual selection and
all rejected alternatives before freezing the promotion protocol. The existing
public defaults remain unchanged; an explicitly selected runtime setting can
be recommended without silently changing the API contract.

The selected source and settings receive new trials. The two distinct
256-square dense-urban and vegetation-rich fixtures are evaluation workloads;
they are not used to choose the initial budget or block. Dense1024 and
vegetation1024 additionally remain large correctness admissions against both
the committed candidate baseline and their original SLEEF references.

## Required local evaluation cells

Every cell has five complete pairs. Use the same native budget on both sides,
one tile worker, all outputs, 153 patches, all 24 chronological timesteps,
checkpoint interval one and the existing logical-tile extent. The baseline
uses block size 128; the candidate uses the selected block size except in the
two explicitly unchanged-default cells below, which also use 128. Bind actual
source inventories, fixture hashes, admissions and harness hashes in the
executable protocol before the first trial.

| Fixture | Regime | Native budget | Role |
| --- | --- | --- | --- |
| repeated_block_256 | geometry warm | selected | Primary benefit |
| repeated_block_256 | geometry warm | 1 | Single-thread guard |
| small_original_cpu | geometry warm | selected | Small-input guard |
| dense_urban_256 | geometry warm | selected | Independent workload guard |
| vegetation_rich_256 | geometry warm | selected | Independent workload guard |
| repeated_block_256 | geometry cold, compiled caches retained | selected | Geometry-cold guard |
| small_original_cpu | first use | 1 | One-thread selected-block startup guard |
| repeated_block_256 | first use | selected | Selected-budget startup guard |
| repeated_block_256 | geometry warm | 1 | Unchanged-default guard: candidate block128 |
| small_original_cpu | first use | 1 | Unchanged-default startup guard: candidate block128 |

Deduplicate only identical complete configurations, keeping all applicable
decision roles. For example, if the selected budget is one, the first two
entries coincide; if the selected block is 128, the two unchanged-default
cells coincide with previously listed guards. This is a local candidate
promotion matrix, not the full P7/P8 upstream performance or memory matrix.

## Statistics and predeclared decisions

For pair i, let A_i be baseline elapsed time and B_i be candidate elapsed
time, with r_i = A_i / B_i. Pairings and all five observations are retained.
The primary median ratio must be at least 1.05. Its paired geometric-mean
bootstrap 95% lower endpoint must exceed 1.0.

For exactly five pairs, enumerate all 5^5 ordered resamples with replacement
of the five log ratios. The statistic is the exponential of each resample's
mean log ratio. Sort all 3,125 statistics and compute the 0.025 and 0.975
quantiles using linear interpolation at index (N - 1) * p. Enumeration has
no random seed or Monte Carlo uncertainty. Also report the observed geometric
mean, median, every ratio, range and paired elapsed times. Five pairs provide
limited uncertainty information; this bootstrap does not establish immunity
to thermal drift or different machines. A future protocol with a different
sample count must define its method before running.

Every warm and geometry-cold guard must have median B_i/A_i <= 1.03.
Each first-use cell must have median B_i/A_i <= 1.10. Any first-use cell
above 1.03 additionally requires primary median r_i >= 1.15 and explicit
disclosure of the startup tradeoff. All cells must pass; neither averages
across cells nor a favorable subset can qualify a failing candidate.

Each trial must remain below the existing 12 GiB observed summed process-tree
RSS cap. Within every paired cell, candidate median sampled peak RSS must
also be no more than 1.10 times baseline median plus 64 MiB. Report the raw
peaks, sample intervals and the shared-page/missed-peak limitations. A failure
is not cured by changing memory thresholds after reading the result.

Failures, interrupted trials and exactness mismatches stay in the record.
Complete the predeclared matrix; do not add trials selectively until a
candidate wins. If a harness defect invalidates a trial, preserve the attempt,
fix and independently review the defect, then freeze a new attempt before
rerunning. Do not combine trials from different source or harness versions.

## Correctness and final integration

Promotion requires exact candidate-to-baseline finite bits including signed
zero, special-value masks, dtypes, categorical outputs, all chronological
state, metadata, timestamps and cache/export contracts. Every-timestep traces
are correctness-only evidence. Retain the original-reference field gates and
the approved four scientific exceptions unchanged.

The final combined source needs component/domain/fallback checks, genuine
chronological execution, cold and warm cache checks, thread/block variants,
restart/cache-invalidation coverage appropriate to the edits and an installed
wheel origin check. Newly cached math and wall entrypoints additionally need
fresh-process cold/warm identity and invalidation checks, including serial
and parallel call order. Requalify both large original-reference admissions
on the exact final source. Inspect a diff and the evidence independently
before copying the source into the production tree and committing.

Report only measured local candidate-to-candidate gains with the exact
hardware, configuration, regime and limitations. No result here establishes
an original-upstream speedup, Linux performance, default-3600 feasibility,
GPU verification or completion of P7/P8.
