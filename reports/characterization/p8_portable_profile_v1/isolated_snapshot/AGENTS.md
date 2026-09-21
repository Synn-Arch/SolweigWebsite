# AGENTS.md — SOLWEIG-light

## 1. Mission

Implement and verify a complete CPU-native `solweig-light` in this repository.

The behavioral baseline is `nvnsudharsan/SOLWEIG-GPU` commit:

`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`

The handoff documents are a source-inspection specification, not evidence that numerical equivalence or speedups have already been demonstrated. Treat all proposed tolerances and performance targets as provisional until P0 characterizes the real upstream baselines and freezes the comparison protocol.

The implementation is complete only when Section 13 of `SOLWEIG_LIGHT_IMPLEMENTATION_PLAN.md` is satisfied. A prototype, isolated optimized kernel, import-compatible stub, or partial API is not completion.

---

## 2. Instruction loading and context discipline

### First code-changing session

Before changing code in a fresh worktree or fresh root task, read:

1. this `AGENTS.md`;
2. `SOLWEIG_LIGHT_IMPLEMENTATION_PLAN.md`;
3. `TASKS.yaml`.

Do this once to establish the product contract and current work-package state. Do **not** blindly reread the full plan or full task file before every edit, test, or delegated subtask.

### After the initial read

Use progressive disclosure:

- Read only the plan sections relevant to the current work package or failure.
- Read only the source files needed to understand the code path being changed.
- Reopen the full plan only when the scope, compatibility contract, numerical policy, release criteria, or work-package boundary is uncertain.
- Reopen `TASKS.yaml` when choosing the next dependency-ready task, changing milestone status, or validating a gate.
- Do not scan the whole repository for a small local change unless evidence indicates the issue crosses module boundaries.
- Do not reload large documents merely to reassure yourself that they still exist.

A worker should receive a small task packet containing the exact objective, relevant files, required invariants, reference command or fixture, and completion gate. Do not send full chat history or unrelated design material when a concise handoff is sufficient.

---

## 3. Autonomous execution and stopping boundaries

For local repository work, proceed without asking for approval at every intermediate step. You are explicitly authorized to:

- inspect and edit repository files;
- create local test fixtures, generated reference manifests, temporary files, caches, and benchmark outputs;
- create isolated local Python environments or worktrees when needed for upstream/candidate separation;
- install development/test dependencies required by the repository into isolated local environments;
- run relevant tests, linters, profilers, numerical comparisons, and benchmarks;
- fix failures caused by your changes and rerun the smallest relevant validation set;
- continue from implementation through execution, result inspection, debugging, and verification for the assigned milestone.

Do not stop merely because the first implementation compiles, an import test passes, or one unit test passes.

Stop and escalate only when one of these is true:

- a scientific/model behavior decision would intentionally differ from the frozen compatibility policy;
- requirements conflict and the conflict cannot be resolved from the plan, tests, upstream source, or recorded policy;
- a required oracle cannot be created without modifying upstream behavior, and the repair policy is not already defined;
- required external credentials, unavailable hardware, or a non-local service prevents further verification;
- an operation would publish, push, release, modify a remote system, destroy irreplaceable data, or otherwise cross the local-development boundary;
- continuing would require weakening an already frozen numerical, artifact, or benchmark gate.

Missing GPU hardware is `not available`, never `passed`.

---

## 4. Multi-agent operating model

Use multiple agents only when decomposition reduces wall-clock work or isolates well-defined reasoning. More agents are not inherently better.

If the environment exposes Astra, Luna, Sol, or equivalent tiers, use the following role model. If those names are unavailable, map the roles to the strongest planning/review model and cheaper bounded implementation workers that are available.

### 4.1 Astra: lead architect and final reviewer

Use Astra for expensive decisions, not repetitive labor.

Preferred Astra involvement:

- at the start of a substantial milestone: resolve architecture, scope, dependencies, scientific-risk boundaries, benchmark design, or ambiguous equivalence questions;
- when a worker reports a genuine cross-cutting blocker or evidence that the current design is wrong;
- at the end of a substantial milestone: review diffs, verification evidence, unresolved deviations, concurrency safety, and whether the gate is actually satisfied;
- for final P8 release review and claim qualification.

Do not use Astra to repeatedly search for files, perform mechanical ports, write repetitive fixture code, rerun the same test loop, or poll workers for status.

### 4.2 Luna: bounded implementation worker

Prefer Luna for tasks with a clear scope and an executable completion condition, including:

- mechanical or localized ports from Torch to NumPy/Numba;
- fixture construction from an already defined reference procedure;
- API/CLI snapshot generation;
- repetitive compatibility tests;
- local refactors with frozen behavior;
- documentation/status updates based on completed work;
- benchmark harness plumbing;
- implementation/debug cycles whose affected subsystem and test oracle are already defined.

A Luna task should normally own one coherent subsystem or one narrowly defined deliverable, not an open-ended repository-wide mission.

### 4.3 Sol: complex bounded implementation and integration

Use Sol when the task is still implementation-focused but needs more reasoning than a routine worker task, for example:

- porting one complete numerical family with nontrivial state or edge semantics;
- diagnosing a differential-test failure across several functions;
- integrating a worker implementation into the chronological pipeline;
- designing a lossless visibility representation after its value set has been characterized;
- resolving a concurrency or memory ownership problem within an already approved architecture.

### 4.4 Prefer lead-at-start, workers-in-the-middle, lead-at-end

For large milestones, the default pattern is:

1. **Lead review:** define the problem, invariant set, work decomposition, and gates.
2. **Worker execution:** implement, run, inspect, fix, and verify bounded tasks.
3. **Lead review:** inspect only the resulting diffs, evidence, failures, and unresolved decisions; decide whether to accept, revise, or continue.

Do not keep the lead agent continuously in the loop when no lead-level decision is required.

### 4.5 No status polling

Do not repeatedly ask a subagent whether it has finished. Do not reread its entire running transcript.

A delegated worker should report only when:

- the assigned task is complete;
- a real blocker prevents further progress;
- evidence shows that a frozen assumption or lead decision must be reconsidered.

If the orchestration environment provides completion events, rely on them rather than polling.

### 4.6 Worker result format

Workers return a compact handoff, not a diary:

- **Scope:** what was assigned and what was intentionally not changed.
- **Changed:** files and important implementation decisions.
- **Commands:** exact relevant commands executed.
- **Results:** tests/comparisons that passed, including counts when useful.
- **Failures:** unresolved failures or unavailable checks, with evidence.
- **Measurements:** raw timing/memory artifacts when the task involved performance.
- **Next:** dependency-ready follow-up or the decision required from the lead.

Do not forward routine intermediate conversation to the lead.

### 4.7 Parallelism rules for agents

Parallelize only independent work.

- Do not have two agents edit the same file or numerical family concurrently unless one is explicitly review-only.
- Do not split tightly coupled chronological-state code across agents merely to increase concurrency.
- Prefer separate worktrees/branches or explicit file ownership for simultaneous implementation.
- Keep a small number of useful workers rather than launching many overlapping workers. Default to at most four concurrent implementation workers unless independence and integration cost justify more.
- Do not nest delegation recursively unless the parent task is genuinely too large and the child tasks have independent gates.
- Merge and verify one coherent change at a time when numerical parity is being established.

---

## 5. Product and compatibility contract

Preserve all seven public workflows:

- `thermal_comfort`
- `preprocess`
- `build_inputs`
- `build_wind_ext_coeff`
- `run_walls_aspect`
- `calculate_svf`
- `run_utci_tiles`

Preserve Python argument semantics, defaults, accepted TIFF inputs, legacy CLI flags, artifact naming, GeoTIFF metadata semantics, masks, timestamp conventions, and legacy cache/export schemas unless an explicitly versioned compatibility policy says otherwise.

The normal package namespace is `solweig_light`.

Provide zero-change `solweig_gpu` imports and the legacy `thermal_comfort` executable only through the explicitly installed compatibility distribution described in the implementation plan. Detect and reject ambiguous installation with upstream rather than silently shadowing or colliding with it.

The own-met TIFF execution path must be CPU-native and must not require Torch, CUDA, network access, ERA5/WRF acquisition dependencies, or other optional service dependencies merely to import or run.

Default compatibility mode preserves the upstream physical/numerical workload. Never create speedup by silently reducing:

- sky-patch count;
- raster resolution;
- timesteps;
- ray support;
- vegetation representation;
- radiation components;
- chronological state;
- output computation required by the model.

Do not replace the model with an isotropic shortcut, lookup approximation, surrogate, or unrelated thermal-comfort package under the default API.

---

## 6. Required execution order

Respect dependency order from `TASKS.yaml`.

### P0 — baseline and executable contract

Complete P0 before performance-oriented implementation work:

- inspect the complete upstream source at the exact commit;
- record source and environment provenance;
- snapshot the Python API, CLI, files, metadata, caches, and artifact behavior;
- create isolated upstream CPU and, where available, CUDA reference environments;
- build deterministic executable fixtures;
- record actual upstream failures instead of hiding them;
- collect raw baseline timings and peak-memory measurements;
- characterize candidate numerical tolerances and freeze them before optimization;
- freeze benchmark workloads/hardware/configuration before making speedup claims.

The handoff's numerical and performance thresholds are proposed gates until this work is done.

### P1 — correctness-first CPU pipeline

Implement a real chronological CPU path and streaming output layer before calling the project optimized. The pipeline must run at least one genuine own-met TIFF-to-TIFF case without Torch.

### P2–P4 — numerical families

Port and verify one family at a time. Establish an executable before/after comparison before optimizing the family further.

### P5–P8 — completeness, robustness, optimization, release

Complete optional workflows, dependency isolation, cache and scheduler correctness, restart behavior, independent scientific checks, full benchmark matrices, packaging, and release evidence.

A wall-only, shadow-only, UTCI-only, or import-only implementation is not a completed milestone beyond its declared scope.

---

## 7. Numerical implementation policy

Prefer NumPy, Numba, SciPy, GDAL/rasterio as appropriate to the frozen dependency policy. The numerical core must not require Torch or CUDA.

For Numba kernels:

- use nopython-compatible code;
- begin with `fastmath=False`;
- establish serial parity before parallelization;
- verify dtype promotion, integer/float conversions, rounding, NaN behavior, signed zero where relevant, min/max behavior, division semantics, comparison boundaries, and operation ordering;
- inspect parallel diagnostics for important `parallel=True` kernels;
- test multiple thread counts for deterministic behavior within the frozen numerical contract.

Do not assume an algebraic rewrite is equivalent simply because it is mathematically similar. Preserve the actual executed model until tests prove the transformation acceptable.

### Chronological state

Timesteps are not independent. Preserve and test all carried thermal state. Parallelize independent pixels, execution blocks with safe ownership, and independent logical tiles, not dependent timesteps.

### Shadow/ray semantics

Until equivalence is proved, preserve:

- ray sample sequences;
- rounding rules;
- first-step behavior;
- termination conditions;
- global bush predicates and vegetation ordering;
- wall-height and wall-sun quantities;
- source/destination slicing;
- border and padding effects.

Do not introduce early exit merely because one output appears settled if another shadow/vegetation/wall output can still change.

### Visibility representation

Do not cast visibility to Boolean because it looks binary on a normal scene. Characterize exact value sets on targeted edge fixtures first.

Use bit packing only for channels proven exactly binary. Otherwise use a lossless categorical representation or a lossless fallback. Legacy exports must round-trip to the required values and schema.

### Deterministic reductions

Do not let parallel patches write into shared output pixels or shared packed words without explicit safe ownership. Prefer pixel/block ownership with a deterministic local reduction order.

---

## 8. Logical tiles, execution blocks, and buffer ownership

A logical tile defines model context and compatibility behavior. An execution block is only an internal locality/memory mechanism.

Changing execution-block size must not change:

- logical tile extent;
- upstream overlap semantics;
- solar position;
- meteorological aggregation;
- geometry context;
- forcing selection;
- output extent or metadata.

Do not overwrite a source field while a neighboring block can still read the old value. Use explicit input/output buffers, barriers between stages, or another provably safe ownership scheme.

Each writable raster dataset has one owner. Temporary SVF/ZIP/NPZ export paths must be tile-specific and collision-safe.

Bound native threads, worker processes, writer queues, scratch workspaces, and live raster blocks together. Do not optimize CPU utilization by causing uncontrolled process-tree memory growth.

---

## 9. Streaming, memory, and cache rules

Stream requested output bands whenever the model no longer needs them. Do not keep complete raster histories solely for convenient writing.

Not saving an output does **not** mean skipping an intermediate quantity that the physical model needs.

Cache only values with understood dependencies. Cache keys/manifests must include every input and policy element needed to make reuse valid.

Geometry may often be reusable. Dynamic ground-view radiance, changing shadows, material temperatures, forcing-dependent quantities, and chronological state are not automatically static.

Caches must handle stale inputs, corruption, interrupted writes, and concurrency safely. Prefer atomic completion and an explicit manifest over guessing that a file's presence means it is valid.

Checkpoint/restart support must serialize the complete required simulation state, not just visible outputs.

---

## 10. Reference and verification integrity

Never generate an upstream golden from `solweig-light`.

Keep upstream and candidate installations isolated so `solweig_gpu` compatibility modules cannot contaminate the reference environment.

For every golden/reference artifact, record enough provenance to reproduce it, including as applicable:

- upstream repository and commit;
- upstream patch hash if a repair was required;
- candidate commit;
- fixture hash;
- environment/dependency manifest;
- invocation;
- hardware and thread settings;
- cache/JIT state.

If unmodified upstream fails, record the failure. If a minimal repair is required to obtain a reference, keep it as an explicit patch and label outputs as a **patched reference**, not an original upstream result.

Distinguish at least these evidence classes:

1. original upstream CPU reference;
2. original upstream CUDA reference, when available;
3. patched upstream reference, when necessary;
4. independent scientific/analytic expectation;
5. candidate output.

Investigate CPU/CUDA disagreement instead of selecting the result that makes the candidate look correct.

---

## 11. Verification requirements

Verification must cover more than final UTCI.

Compare, where applicable:

- input normalization and masks;
- tile extents and metadata;
- wall heights and aspects;
- generated sky-patch tables/order;
- SVF and directional/vegetation SVF fields;
- shadow and visibility channels;
- shortwave components;
- longwave components;
- ground-view components;
- surface-temperature/delay state after each timestep;
- TMRT;
- UTCI;
- WBGT;
- diagnostic Ta and wind fields;
- cache/export round trips;
- output band timestamps and schemas.

Freeze justified numerical tolerances before performance tuning. Once frozen, do not loosen them, delete a difficult fixture, add a skip, or redefine a benchmark because an optimization fails.

At least one genuine small CPU-only own-met TIFF-to-TIFF test must run in ordinary PR CI with no numerical-pipeline mocks and no unconditional skip.

Mocks for optional remote/services are allowed only when clearly labeled as mocks and must not be presented as live-service verification.

Test concurrency-sensitive paths with different block sizes, thread counts, worker counts, cold/warm caches, stale/corrupt caches, interrupted output, and restart when those features are implemented.

---

## 12. Scientific behavior and deviations

Maintain `docs/model_deviations.md`.

Record upstream quirks and suspected bugs with reproducing tests. Optimization work must not silently change:

- scientific constants;
- canopy transmission;
- wind-speed floors;
- UHI placement;
- WBGT sun/shade selection;
- altitude handling;
- land-cover normalization/aliasing;
- tile boundaries and overlap behavior;
- timestamp/timezone conventions;
- other behavior identified in P0's behavior register.

Scientific corrections require an explicit versioned policy, separate tests, and a reference report. Keep compatibility behavior and corrected behavior distinguishable.

Do not bury a scientific change inside a performance refactor.

---

## 13. Dependency and packaging discipline

Keep the base own-met TIFF runtime minimal. Optional acquisition/preprocessing integrations must be lazily imported or placed behind optional extras so they do not become import-time requirements for the core numerical path.

Do not reintroduce Torch as a hidden dependency through compatibility modules, tests, or packaging.

Test installed wheels/distributions, not only source-tree imports.

The compatibility distribution that owns `solweig_gpu` must detect or document conflict with the real upstream package. It must never silently win an ambiguous import collision.

---

## 14. Performance evidence policy

Performance work begins from measurement, not assumption.

Compare equal work:

- same logical tiles;
- same raster resolution;
- same sky patches;
- same timesteps;
- same physical options;
- same forcing;
- same requested outputs;
- explicitly matched cache state.

Measure at least:

- upstream default CPU behavior;
- strongest documented/tuned upstream CPU configuration that preserves the workload;
- `solweig-light` one-thread behavior;
- `solweig-light` selected CPU-budget behavior;
- upstream CUDA separately when suitable hardware is actually available.

Keep first-use, geometry-cold, geometry-warm, and kernel-only measurements distinct.

First-use timing includes relevant JIT startup and I/O. Do not present a warm kernel microbenchmark as end-to-end performance.

Record raw trials, failures, process-tree peak RSS, hardware/configuration, thread/affinity settings, cache state, stage timings, and uncertainty. Do not publish a speedup inferred from decorators, theoretical complexity, or a single unqualified timing.

Performance thresholds in the handoff become release gates only after P0 freezes the workloads, baselines, hardware/configuration, and justified interpretation.

---

## 15. Skills and repository instructions

Keep skills narrow and trigger-specific.

- Do not install or load many skills speculatively.
- A skill's description should make its use condition obvious and short.
- For a multi-workflow skill, read the root instructions first and open only the referenced material needed for the active workflow.
- Do not force every agent to reread broad documentation that is irrelevant to its task.
- Remove or revise instructions that repeatedly cause redundant repository scans, redundant tests, or unnecessary approval pauses.

Repository instructions should describe durable contracts and boundaries. Put detailed one-off procedures in task-specific docs or scripts rather than expanding `AGENTS.md` indefinitely.

---

## 16. Testing cadence

Use the smallest test set that can falsify the current change while developing. Expand validation at milestone gates.

A normal implementation loop is:

1. reproduce or establish the reference;
2. implement the smallest coherent change;
3. run the targeted differential/unit test;
4. inspect actual output, not only exit status;
5. fix failures;
6. rerun the targeted test;
7. run the relevant subsystem suite;
8. at the milestone gate, run the required integration/contract/performance checks.

Do not rerun the full suite after every tiny edit unless the change has repository-wide risk. Do not skip the full required gate merely because targeted tests passed.

---

## 17. Progress records and milestone status

Keep milestone status reproducible but concise.

At P0, create or adopt a repository progress record if one does not already exist. It should link to durable artifacts rather than embedding long transcripts.

For every completed milestone or significant checkpoint, record:

- candidate commit/worktree state;
- changed subsystems;
- exact relevant commands;
- tests/comparisons executed;
- failures and unavailable checks;
- raw measurement artifact paths when performance was measured;
- deviations or unresolved scientific questions;
- next dependency-ready task.

Update `TASKS.yaml` milestone status only when its stated gate is actually satisfied. `planned`, `implemented`, `tested`, `verified`, `benchmarked`, and `complete` are not synonyms.

Do not mark unavailable GPU verification, unexecuted benchmarks, skipped end-to-end paths, or planned optional workflows as complete.

---

## 18. Definition of done

Do not declare the repository complete until all requirements in Section 13 of `SOLWEIG_LIGHT_IMPLEMENTATION_PLAN.md` are satisfied, including at minimum:

- all seven public workflows are operational, not stubs;
- the core own-met TIFF path runs without Torch/CUDA;
- normal `solweig_light` packaging works;
- opt-in legacy `solweig_gpu`/`thermal_comfort` compatibility works without silent collision;
- required API, CLI, artifact, metadata, and cache contracts pass;
- required intermediate and final numerical comparisons pass frozen gates;
- chronological state is verified across representative day/night sequences;
- scientific checks and `docs/model_deviations.md` are complete for the supported release scope;
- at least one genuine CPU-only TIFF-to-TIFF CI path is non-skipped;
- caches, execution blocks, concurrency, output ownership, and restart paths satisfy their robustness gates;
- output history does not cause unbounded memory growth;
- benchmark evidence is reproducible and uses equal work;
- any performance claims are limited to measured hardware/configuration/regimes;
- missing hardware, failed cases, mocks, patched references, and unsupported workflows are labeled accurately.

When unsure whether a gate is satisfied, treat it as not yet satisfied and report the evidence still needed.

---

## 19. Prohibited shortcuts

Never use any of these to make the project appear complete or faster:

- numerical stubs or placeholder outputs;
- unconditional skips on required end-to-end tests;
- candidate-generated upstream goldens;
- reduced sky patches, resolution, timesteps, rays, vegetation, or radiation physics in default mode;
- hidden isotropic/surrogate/lookup substitutions;
- silent scientific corrections;
- relaxed tolerances after observing an optimization failure;
- deleting or weakening difficult fixtures;
- redefining the benchmark after seeing results;
- counting unavailable GPU checks as passes;
- reporting microbenchmark speedup as end-to-end speedup;
- using cached work on one side of a benchmark without matching and documenting cache conditions;
- silently replacing an upstream installation through the compatibility package;
- keeping full time-history rasters in memory when streaming is sufficient;
- parallelizing dependent timesteps;
- unsafe shared writes merely because they appear to work in one run;
- repeated lead-agent polling or repeated full-context ingestion as a substitute for good task decomposition.

The goal is not to produce a convincing demo. The goal is a source-provenanced, CPU-native SOLWEIG implementation whose compatibility, numerical behavior, robustness, and measured performance are independently reproducible.
