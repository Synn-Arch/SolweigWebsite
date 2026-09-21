# Numerical contract under P0 characterization

Status: **frozen P0 comparison proposal; not yet validated for optimization**.
The machine-readable authority is `benchmarks/protocols/comparison_v1.json`,
covering 96 named component return fields with units and context-sensitive rules.
No candidate result has been
used to select or relax these budgets. The original plan's final-output budgets
are retained. P0/P1 must finish field-specific propagation and broader-fixture
checks before validating the affected family for optimization. Existing budgets
must not be relaxed in response to a candidate failure.

Evidence anchors:

- `reports/geometry_characterization.json`: float32/float64 patch and shadow
  cases, actual value sets, and separately recorded analytic failures.
- `reports/boundary_capture_verification.json`: component dtypes and carried
  state across one full day/night sequence.
- `reports/initial_repeatability.json`, `reports/capture_noninterference.json`:
  same-host exact TIFF checks. These do not measure cross-platform differences.
- `reports/upstream_cpu_tuning.json`: once complete, exact TIFF checks at three
  native-thread configurations. This cannot establish intermediate-state
  thread invariance without separately captured runs.

## Comparison procedure

Shapes and finite/NaN/+Inf/−Inf locations must match before numerical errors are
computed. Sentinel masks are checked separately from legitimate numeric values.
Categorical geometry, patch identity/order, tile indices, output timestamp order
and GeoTIFF schemas require exact equality. ZIP bytes and compression layout are
not semantic equality requirements. CRS is compared semantically, with the full
affine transform retained. NoData metadata is checked separately from NaNs.

Report bias, MAE, RMSE, p95/p99 and maximum absolute error, relative error where
meaningful, and worst coordinates/timestep. The max-error limit applies even if
whole-domain RMSE is small. Geometry classification mistakes cannot be hidden
inside a continuous radiative error budget.

| Family | Proposed gate | Evidence/rationale and remaining check |
|---|---|---|
| Wall heights | Exact float32 on finite stencil fixtures | Discrete stencil and threshold should be reproducible; boundary/tie fixtures still needed. |
| Visibility | Exact values and masks | A combined channel can equal 2; Boolean compression is disproved. Require lossless fallback for uncharacterized values. |
| SVF | Maximum absolute 1e-6 | Plan budget, not yet candidate-validated. All 15 fields, not just total SVF. |
| Radiative fluxes and delayed `Tgmap1*` | `atol=0.05 W/m²`, `rtol=1e-5` | Delayed fields carry radiance in the same units; check state errors at every step and propagation into TMRT. |
| TMRT and surface-temperature `TgOut1` | Maximum absolute 0.01 °C | Temperature state must not accumulate drift; longer sequences remain needed. |
| UTCI/WBGT | Maximum absolute 0.02 °C | Plan budget; preserve missing-value behavior and report category flips near downstream thresholds separately. |
| `firstdaytime`, timestamps, iteration/patch identities | Exact | These are control state, not approximate physical fields. |
| `timeadd`, `timestepdec` | Exact on the generated chronology | Inputs and updates must retain ordering; calendar and irregular-step cases remain needed. |
| `CI`, `CI_Tg`, `CI_TgG`, emissivity coefficients | Maximum absolute 1e-6 proposed | Dimensionless scalar changes can propagate to many pixels; inspect flux/temperature sensitivity before freezing. |
| Solar angles and aspect fallback | Pending field-specific calibration | Discrete orientation choices remain exact; continuous angular budgets must not conceal ray-classification changes. |

The reference raster outputs observed so far are float32. Main-return scalar
`CI`, `I0`, `radI` and `radD` change dtype across day/night; a uniform float64
rewrite is not automatically compatible. Preserve actual operation order,
promotion and scalar-versus-array behavior before experimenting with regrouping.

The scientific expectations and compatibility oracle are separate. Exact-zenith
flat-scene failures remain visible; default compatibility must not silently fix
them. CUDA is unavailable on the present host, so no CPU/CUDA agreement or
universal-device claim is established.

## Verified family evidence through P3 and active P4 calibration

The original proposal remains unchanged; verification is family- and fixture-
specific. P2 wall/UTCI and P3 shadow/SVF comparisons have passed their recorded
local gates (`reports/p2_installed_verification.json`,
`reports/p3_installed_verification.json`). The pending entries above must not be
read as permission to relax a failed optimization or as completed broad
scientific calibration.

P4 GVF sensitivity: `reports/p4_gvf_sensitivity.json` records a zero-error
unperturbed original replay and coherent signed/checkerboard perturbations of
all dimensionless GVF fields by 1e-6. Maximum TMRT change across the 24-step
scene was 0.00006103515625 C. This supports the proposed GVF budget for that
scene; other scenes and arbitrary error patterns are not bounded by the
experiment. Solar/emissivity parameterization remains unchanged.

Forty original thermal-delay fixtures cover both first-day flags, the adjacent
float64 values around 59/1440, and 1/30/60/120-minute steps. Their exact control
state and 0.01-temperature-budget array comparisons pass. See
`tests/reference/delay_original_cpu/manifest.json` and
`tests/differential/test_delay_reference.py`. Longer chronological evidence now
includes an original 48-hour driver capture with daily water-temperature and CI
resets; production restart robustness is still a separate P6 requirement.
