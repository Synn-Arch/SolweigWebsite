# Four independent scientific failures: proposed release disposition

Status: Alternative A explicitly approved by the user: “Retain upstream behavior; document exception.” The approval is implemented in `reports/scientific_release_exceptions.json` and `docs/model_deviations.md`. All four test outcomes remain failed; numerical source, tests and tolerances are unchanged. The proposal and alternatives below are retained as the decision record; statements requesting approval describe the earlier proposal and are superseded only for Decision 1. Decision 2 (portable reference policy) remains pending. No numerical rerun was needed for this documentation-only disposition.

The decision is whether a future compatibility release may explicitly retain these inherited behaviors while reporting the independent failures, or whether a separately versioned scientific correction must precede that release. This decision cannot waive the separate current-main dense1024 original-CPU TMRT failure or any other unresolved release gate.

## Exact assertions and observed values

| Assertion / executed case | Scientific property and fixed allowance | Retained result |
|---|---|---|
| `test_default_full_patch_svf_fields_are_physically_bounded[open-svf_calculator]` | On the unobstructed 9×13 float32 scene, all ten vegetation/combined-vegetation SVF fields equal 1 within absolute 0.00025; all 16 SVF quantities must first be finite and bounded by the same allowance. Full option-2, 153 patches. | Bounds checks complete before the failure. First failed unity field is returned index 6 (`svfNaveg`): 0.9752265214920044 at all 117 cells versus 1; deficit 0.024773478507995605. |
| `test_default_full_patch_svf_fields_are_physically_bounded[open-svf_calculator_compact]` | Identical scientific property using lossless compact visibility. | Same first failed field, value and 117/117 violations. This is a second implementation path, not a second independent physical phenomenon. |
| `test_umep_full_ground_view[0]` | All 17 fields agree with unchanged pinned UMEP for the 9×11 landcover-off scene; non-longwave fields absolute 1e-6, zero relative tolerance; longwave absolute 0.05 and relative 1e-5. Masks and relevant mutations also checked. | First failure is field 15, `gvfSum`: 12/99 cells exceed 1e-6. At [6,7], candidate = original Torch = 9.240001678466797; UMEP = 9.239999967813493; difference 1.7106533043431682e-6. |
| `test_umep_full_ground_view[1]` | Same comparison with landcover on, including water-class mutation semantics. | Same `gvfSum` maximum and 12/99 violations. Original Torch and candidate field 15 are bitwise identical. |

The SVF test stops at its first failed unity assertion. A retained separate diagnostic reports additional open-scene deficits: indices 9/10 are 0.9484643340110779 and 13/14 are 0.8594677448272705. These are reported diagnostics, not extra executed test assertions. The earlier output values 1,3,4 equal one. The test name must not be shortened to “SVF bounds fail”: the observed failure is directional open-scene unity, after the finite/range checks passed.

## Evidence and source cause

Evidence root: `reports/cura_p8_20260919T1215Z_a7c3/`.

- `results/p8_queued_science_tests.xml` retains the four exact failures and assertion traces.
- `results/svf_ground_view_diagnostics.txt` retains all SVF field ranges and UMEP fieldwise errors, including serial/parallel agreement.
- `admission/admission_manifest.json`, key `svf_initializer_execution`, records actual pinned Torch and candidate initializer execution. Both declare band counts [31,30,28,24,19,13,7,1] but write [31,29,27,24,19,12,6,1]: 149 entries for 153 consumed slots. Float32 reciprocal steps followed by integer truncation shorten four bands. The complete azimuth arrays agree. For example, the nominal 12-degree step is 12.000000953674316. The consumer uses declared band lengths, crossing initialized band boundaries and consuming trailing zeros.
- **Full SVF evidence now captured:** `reports/characterization/p8_science_failure_disposition/svf_full_open_original_cpu/manifest.json` and its raw original/dense/compact NPZ files establish bitwise agreement across all 19 outputs on the exact 9×13 fixture. Original Torch reproduces all six directional-unity failures (indices 6,7,9,10,13,14), including the exact candidate values above. Every independent bound check passes. This closes the original-full-field evidence gap. A corrected-initializer counterfactual has still not established that this change alone repairs every deficit; that is a correction-design gate, not a missing compatibility observation.
- `results/ground_view_original_torch_cpu.npz`, `ground_view_candidate_cpu.npz`, and `ground_view_original_torch_comparison.json` establish all 17 original/candidate outputs pass unchanged compatibility limits in both modes; `gvfSum` and `gvfNorm` are exactly equal float32 arrays. Largest longwave difference is 3.0517578125e-05, within its separate flux gate.
- Unchanged UMEP `gvf_2018a.py` initializes `gvfSum` with NumPy default float64 zeros (line 49), accumulates 18 azimuth contributions (line 81), and normalizes by their count (line 134). Pinned Torch/candidate retain float32 accumulation. This is an inherited arithmetic-precision difference, not demonstrated UMEP physical error. `gvfNorm` differs by at most 1.0828177132715666e-7 and passes its authored limit. Passing normalized values does not erase failure of the independently asserted sum.

UMEP source is pinned to `3fcc0c3dca67d1d5644a6d34b9148d7a365743ba`; SOLWEIG-GPU is pinned to `0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. Test source and source manifests identify precise fixtures, hashes and environments. These results do not establish measured environmental accuracy or whole-model UMEP parity.

## Alternative A: explicit compatibility-retained release exception

Proposed approval wording:

> Permit these four named independent-check failures as documented limitations of the pinned compatibility profile, with the now-complete matched original/candidate SVF reproduction retained. Preserve their failed results, thresholds and reproducible tests. Retain legacy arithmetic and azimuth behavior. Do not claim physically correct unobstructed directional SVF or full UMEP agreement at the authored tolerance. This exception does not waive original-reference compatibility, other scientific checks, final numerical budgets, packaging, runtime or performance gates.

This exception would allow release consideration despite specifically retained physical/reference limitations; it would not make the assertions pass. No user approval is inferred. Regression checks must still distinguish a changed/worsened behavior from the approved known failure; no unconditional skip, fixture deletion, wider allowance, or silent expected-pass conversion is proposed. The existing exact-zenith visibility defect and other deviations require their own recorded scope and are not automatically covered by this four-failure exception.

## Alternative B: separately versioned corrections

For SVF, first establish and review a declared-count-driven initializer with independent expected patch positions; do not assume filling four missing slots suffices. Verify every patch identity/order, all directional and vegetation fields, open/obstructed geometry, supported patch options, legacy exports and caches. Keep original compatibility behavior available and distinguish corrected cache identities. Run downstream state and all-field chronological original-versus-corrected impact comparisons over full 153-patch/24-hour workloads, including dense and vegetation scenes. Corrected-mode scientific acceptance must be independently justified before implementation, never calibrated after observing errors.

For UMEP precision, specify exactly which accumulator/intermediates use float64 and why. Re-run the unchanged independent component assertion, original compatibility comparisons, mutation/nonfinite cases, serial/parallel/block variants and downstream chronological propagation. Keep compatibility mode's float32 arithmetic separate. Increased precision alone is not proof of improved physical accuracy or compatibility, and no original-reference tolerance is silently transferred to the corrected profile.

Either correction changes scientific/numerical policy and needs explicit approval before production changes. A source repair prototype may be studied separately, but cannot silently become the default release policy.

## Meaning of scientific-check completion

AGENTS §18 requires scientific checks and deviations complete for the supported release scope. Plan §10.5 requires actual independent checks; §13 requires executed checks and met numerical budgets. Neither authorizes describing a recorded failure as passed. Conversely, they do not mandate secretly correcting inherited model behavior to force every physical expectation green while claiming legacy equivalence.

At present the checks executed, but the scientific release disposition is incomplete. An approved, narrowly recorded release exception can define the supported compatibility claim; it is a policy decision, not a new measurement. If the selected release requires every named independent assertion to pass, Alternative A is insufficient and the failures remain blockers until a versioned correction is validated. P8 cannot be marked complete from this memo.

The narrowly requested original-SVF capture is complete: all 19 outputs are bitwise equal for original/dense/compact, and the scientific unity failures remain. The evidence manifest was independently rehashed during this review. No further execution is needed to present the following decisions.

## Concrete decisions for user review

**Decision 1 — inherited scientific limitations.** Approve the exact Alternative A exception above for these four named failures, or require versioned correction and validation before release consideration. Approval of Alternative A changes the supported scientific-release claim; it does not approve skipped tests or label their outcomes passed. Approval of correction authorizes a separate policy/design effort, not an unreviewed replacement of legacy default arithmetic. No choice is inferred from the user's portable-math preference.

**Decision 2 — portable Linux reference contract, separate from these failures.** Retain the original-Cura contract and leave Linux numerical admission unresolved pending a new justified algorithm, or authorize design of a separately named deterministic portable profile whose deviations from original Cura remain measured and explicit. The latter changes the compatibility contract and needs its own predeclared scientific gates before implementation or release. The finite licensed-source search has no passing profile; this is not proof of impossibility. This decision is not an MKL dependency proposal and cannot be implied by approving Decision 1.

The first option in each decision preserves the existing default model/reference behavior. Keeping those choices leaves outstanding release blockers accurately open; approval is not needed merely to continue reporting them. A platform-limited release would require a further explicit support-scope decision and final platform-specific qualification, not automatic fallback to the easiest oracle.
