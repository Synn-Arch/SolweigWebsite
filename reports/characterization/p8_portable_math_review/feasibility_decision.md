# Portable compatibility: bounded decision review

Decision: no tested portable profile is ready for Linux integration. User instruction excludes a pinned MKL production dependency. This is an unresolved compatibility problem, not proof that a portable solution is impossible. No source, numerical gate, or release status changed in this review.

## Evidence and its limits

The retained source-derived set includes CORE-MATH, MIT Intel IGC HA, SLEEF u10 and native u35 scalar/AVX512, BSD Intel SVML LA, AMD AOCL default scalar, and OpenLibm scalar. Restrictively headed SVML HA assembly is excluded. Source-defined explicit FMA is not a free compiler tuning parameter; changing it creates another algorithm requiring an independent rationale and validation.

Latest native results on the 189,220-input corpus and actual 774×153 scalar-patch oracle:

| Profile | Finite ASVF bit differences | Complete sun/shade differences | Original-ASVF diagnostic sun/shade |
|---|---:|---:|---:|
| AOCL | 24,670 | 60 / 40 | 0 / 4 |
| OpenLibm | 9,693 | 9 / 20 | 5 / 13 |
| Deployed glibc 2.39 | 6,558 | 10 / 13 | 4 / 10 |

Both native symbol origins were checked. `cura_remaining_native_profiles/manifest.json` retains build, source, library and input identities; owner verified all 30 downloaded entries and cleanup. Earlier results are retained in `review.json`, `uniform_source_combinations.json`, and `cura_native_tangent/manifest.json`. Native SLEEF u35 scalar and AVX512 give identical tested outputs and both fail 19/9 with original ASVF, actual coefficients and CORE atan; BSD SVML LA fails 9/20 in that diagnostic. These are not complete-profile measurements. No boundary-failing new profile received a full chronological pipeline run.

Not all differing float bits are disqualifying. With original hsvf substituted for diagnosis, CORE atan has 25 differing float32 results but zero classifier differences. Conversely, the existing main implementation already fails an actual original-CPU dense1024 TMRT gate: 0.04949951171875 versus 0.01 degrees. A single hotspot match, a smaller ASVF mismatch count, or correct rounding does not repair that admission.

## What the existing policy actually requires

The 774-input diagnostic corpus was added during diagnosis; it is not an enumerated P0 fixture. `comparison_v1.json` itself retains the status `frozen_P0_comparison_proposal_not_validated_candidate_contract`. Do not describe this newly captured corpus as an original P0 bitwise-transcendental gate.

Nevertheless, plan §10.3 requires categorical equality, inspection of intermediates and boundaries, and explicitly rejects excusing a wrong shadow boundary with low aggregate error. §4's numerical policy requires preserving comparison thresholds and actual operation ordering. P4 freezes unchanged field gates and exact masks. The accepted research packets explicitly require zero differences against the corrected actual-call classifier oracle before promotion. Dropping that test after failure would therefore abandon a known compatibility regression and the current experiment gate. Intermediate float bit equality was never a universal requirement.

The original vectorized classifier capture had wrong scalar promotion; replacing it with the 153 actual upstream calls corrected an invalid oracle. That correction is evidence-based and distinct from choosing a more convenient numerical backend. Choosing M1/SLEEF, correctly rounded arithmetic, a newer Torch build, or patched upstream as the Linux reference merely because it passes would change reference policy. Plan §9 requires explicit compatibility-profile choices and investigation of CPU/CUDA disagreements. Even setting the synthetic corpus aside would leave current-main dense1024 failed, and the new profiles' dense1024 outcomes unmeasured.

## Finite next research boundary

There is no currently evidenced, ready-to-build untested algorithm with a demonstrated reason to reproduce the remaining original rounding. Do not start another broad combination, compiler, FMA, ISA, or coefficient sweep.

The final bounded lead was executed after its source-audit entry condition passed: deployed Ubuntu glibc 2.39-0ubuntu8.9 has distinct tanf range reduction/kernel and atanf source relative to pinned OpenLibm. Its one coherent native profile also fails: 6,558 finite ASVF differences and 10 sun / 13 shade differences; original-ASVF diagnostic 4/10. `cura_glibc_audit/manifest.json` retains source/package/build ID, symbol provenance, raw results and cleanup. Its evidence inventory was independently rehashed in this review. No full-pipeline run followed the failed boundary gate.

The bounded search has exhausted its currently justified known leads. Stop trial-and-error algorithm/configuration sweeps. A legitimate restart needs new source evidence, an independently justified algorithm, or an explicit reference-policy decision. Failure of this finite set cannot establish impossibility over all algorithms or platforms. The portable Linux compatibility problem remains unresolved; MKL production integration is not reconsidered.

## Release branches within the user's constraints

1. Continue portable research only when a source-based hypothesis meets the bounded entry condition above. Existing original-Cura field and categorical gates remain. This needs no scientific-policy change; no currently justified untested lead or passing implementation remains.
2. Prepare a platform-qualified M1 candidate branch using the retained SLEEF prototype and finish its actual-call, chronological, installed-wheel and supported-platform gates. This does not solve Linux compatibility. Advertising a narrower supported release requires an explicit product/release-scope decision; it cannot complete the current cross-platform contract silently.
3. Propose a separate, versioned deterministic portable math profile with scientifically justified rounding and explicit disagreement reports. This requires approval because it intentionally changes the numerical/reference contract, even if continuous errors are small. It must not replace or erase original-Cura failures. No such policy is adopted here.

MKL production integration is excluded by the latest user decision. A portable research instruction does not authorize changing tolerances, discarding fixtures, selecting easier reference outputs, fitted lookup behavior, or altering strict classifier boundaries.

## Independent science and actionable release work

Plan §10.5 requires independently derived checks and matched, separately pinned UMEP comparisons. Section 13 says selected checks must actually execute and numerical budgets must be met; it does not demand silently correcting every inherited scientific defect. Compatibility and physical validity are separate evidence classes.

The two UMEP ground-view failures are inherited precision differences: original Torch and candidate match while UMEP's float64 sum differs by up to 1.7106533043431682e-6 against the predeclared 1e-6 allowance. The two directional SVF unity failures have an inherited initializer defect: 153 entries declared, 149 initialized. These results remain failed independent checks, not successful physical validation. Source agreement cannot turn them green.

A corrected SVF initializer or different ground-view arithmetic requires a separately versioned scientific policy and downstream validation. A compatibility-only release need not invent such a correction merely to satisfy an analytic expectation, but the current records explicitly leave scientific-policy disposition unresolved. A release decision must expressly retain these defects, limit scientific claims and explain the failed expectations; execution alone is not permission to claim all scientific gates passed. No tolerance change or skip is an acceptable disposition.

Dependency-ready work is a concrete four-failure disposition package: retain executable original/candidate/independent comparisons on the same inputs; cross-link each failing assertion to its preconditions, exact originating operation, original upstream reproduction and release claim affected; retain the newly captured exact 19-output original/dense/compact SVF evidence, which establishes inherited behavior but still fails six directional-unity expectations. Draft compatibility-retained and separately-corrected policy alternatives without implementing either. Reuse retained same-source executions; run only a missing reproducer identified by this audit. Then have the release owner decide supported compatibility scope explicitly. This is distinct from another status report or a broad rerun, and is independent of ASVF replacement research.

Additional unblocked integration work should reuse the existing installed-wheel/CI checkpoint rather than duplicate it: close only identified unsupported-runner, optional-workflow or artifact-provenance gaps with exact executable gates. No release or P7/P8 completion follows from this review.
