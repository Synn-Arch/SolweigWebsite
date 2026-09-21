# P0 static audit: radiation and chronological state

Evidence class: source inspection only. Inspected all 2,225 lines of
`solweig_gpu/solweig.py` in upstream commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec` (local checkout HEAD verified).
Anchors below link to that immutable source; line intervals describe the inspected
block. This does not establish executability, numerical parity, tolerances, or
performance. Plan sections 5.4–5.5 guide the hazards to characterize.

Source: [pinned solweig.py](https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/solweig.py).
All `L<number>` anchors below refer to this file.

## Complete local function graph

There are 24 definitions. The following graph includes every definition and every
call between them; imported numerical operations are omitted.

```text
Solweig_2022a_calc (1907)
  daylen (47)
  clearnessindex_2013b (831) -> sun_distance (815)
  diffusefraction (908) -> ensure_tensor (30)
  Perez_v3 (1207) -> imported shadow.create_patches
  shadowingfunction_wallheight_13 (956)
  shadowingfunction_wallheight_23 (1059)
  gvf_2018a (291) -> sunonsurface_2018a (75)
  TsWaveDelay_2015a (451): six calls on daytime branch
  cylindric_wedge (406)
  Kup_veg_2015a (490)
  Kside_veg_v2022a (564)
    Kvikt_veg (510)
    shaded_or_sunlit (524)
  imported shadow.create_patches (night anisotropic branch)
  Lcyl_v2022a (1610)
    model2 (1374): selected by hardcoded emis_m=2
    model1 (1343): source branch exists, inactive with emis_m=2
    model3 (1396): source branch exists, inactive with emis_m=2
    define_patch_characteristics (1419) -> shaded_or_sunlit (524)
  Lside_veg_v2022a (1737) -> Lvikt_veg (1719)
```

Top-level imports also require GDAL, SciPy, NumPy, Torch, and `.shadow`, and invoke
`gdal.UseExceptions()` (16–28). This module contains no UTCI/WBGT evaluator, raster
writer, cache loader, or public workflow dispatcher.

## Dependency and output inventory

| Function/block | Inputs actually used and outputs/dependencies |
|---|---|
| `ensure_tensor`, 30–45 | Input value and optional device; tensors return unchanged, including their existing device/dtype. Non-tensors use `torch.tensor`. |
| `daylen`, 47–73 | DOY and latitude; outputs day length, declination, sunset-like `SNDN`, sunrise `SNUP`. `SOC` clamped to [-1,1]. Main uses only SNUP. |
| `sun_distance`, 815–829 | Julian day; returns harmonic distance correction D. |
| `clearnessindex_2013b`, 831–906 | Zenith (radians), Julian day, Ta, RH fraction, radG, latitude, P; returns I0, CI, Kt, I0et, CIuncorr. Latitude and day choose G table; pressure sentinel changes p. |
| `diffusefraction`, 908–954 | radG, altitude (degrees), Kt, Ta, RH percent; returns **radI, radD**, opposite the docstring tuple. Missing-temperature/humidity branch differs from measured branch. |
| Shadow 13, 956–1056 | DSM, solar angles (degrees), scale, walls, aspect (radians); returns sh, wallsh, wallsun, facesh, facesun. Ray recurrence and wall geometry feed main ground view and shadow. |
| Shadow 23, 1059–1204 | Above plus vegetation crown/trunk elevations, amaxvalue, bush; returns vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun. Main uses vegsh/sh/wallsun and combines transmission psi. |
| `sunonsurface_2018a`, 75–288 | Search azimuth, scale, buildings, shadow, sunwall, first/second distances, aspect/walls; ground/wall/air/water temperature, ground emissivity/albedo, wall emissivity/building albedo, SBC, landcover flag/grid. Returns gvf, gvfLup (excess emission plus local correction), gvfalb, gvfalbnosh, gvf2. Mutates Tg on water and sunwall. All dynamic radiation inputs remain dependencies even when geometry is fixed. |
| `gvf_2018a`, 291–404 | Above with wall-sun heights/directions and rows/cols; calls 18 directions, returns 17 fields: Lup/albedo/unshadowed albedo globally and E/S/W/N, gvfSum, gvfNorm. Adds baseline air-temperature emission to five Lup fields. |
| `TsWaveDelay_2015a`, 451–488 | Current field, firstdaytime, timeadd, timestepdec, prior field; returns delayed field, updated clock, prior-state replacement. Applies independently to five Lup fields and Tg output. |
| `cylindric_wedge`, 406–449 | Zenith (radians), svfalfa, shape; returns F_sh only (docstring incorrectly says tuple). Main replaces NaNs by 0.5. |
| `Kup_veg_2015a`, 490–508 | radI/D/G, altitude, svfbuveg, building albedo, F_sh, global/cardinal gvfalb and gvfalbnosh; returns Kup/E/S/W/N. |
| `Kvikt_veg`, 510–521 | svf, svfveg, vikttot; outputs vegetation and wall weights. |
| `shaded_or_sunlit`, 524–562 | Sun and patch angles, asvf; returns two strict Boolean comparisons, with equality in neither set. |
| `Kside_veg_v2022a`, 564–812 | radI/D/G, shadow, directional SVF/vegetation SVF, sun angles, psi, t, albedo, F_sh, Kup cardinal, cyl; anisotropic branch additionally uses lv, diffsh, asvf and three visibility matrices. Shape controls allocation. Returns Keast/S/W/N, KsideI/D, Kside. |
| `Perez_v3`, 1207–1341 | Zenith **degrees**, azimuth **degrees** despite docstring, radD/I, jday, patchchoice/option; returns lv (altitude, azimuth, normalized luminance when patchchoice=1), PerezClearness, PerezBrightness. Patch order comes from imported create_patches. |
| `model1/2/3`, 1343–1416 | Patch altitude and esky; Ta parameter unused. Return patch-normalized emissivity and altitude-band emissivity; normalized result is not used by Lcyl. |
| `define_patch_characteristics`, 1419–1607 | Sun angles, patch angles/solid angles/asvf, shmat/vegshmat/vbshvegshmat, Lsky_down/side, Lup, Ta/Tgwall/ewall, shape; returns Ldown, Lside, five side components and E/W/N/S. Parameter Lsky unused. First patch sweep computes sky/vegetation/sunlit/shaded contributions; second computes reflection from **Ldown_sky + Lup**, not total Ldown. |
| `Lcyl_v2022a`, 1610–1717 | esky, patch geometry, Ta/Tgwall/ewall/Lup, visibility matrices, sun angles/asvf, shape; selects model2, builds solid angles and sky directional emission, returns Ldown/Lside/E/W/N/S. Original luminance column is replaced in cloned working patch tables. |
| `Lvikt_veg`, 1719–1734 | svf, svfveg, svfaveg, vikttot; returns vegetation/wall/sky/reflection weights. |
| `Lside_veg_v2022a`, 1737–1904 | Directional SVF, vegetation and above-vegetation SVF; sun angles, Ta/Tw, SBC/ewall, Ldown/esky, t/F_sh/CI and Lup cardinal, anisotropic flag. Returns E/S/W/N; anisotropic branch returns only 0.5*directional Lup, other components are added by caller. |

## Main state and chronological ordering

[Main signature, 1907–1913](https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/solweig.py#L1907)
contains configuration, immutable geometry/material fields, forcing/time, and
carried state. `i` is unused. TgK/Tstart/TmaxLST and wall variants are parameters
of the diurnal surface-temperature curve, not prior-step state (2046–2066).

Carried state is `firstdaytime`, `timeadd`, five `Tgmap1*` fields, `TgOut1`, and
`CI`. `timestepdec` passes through unchanged. Daytime recomputes CI from forcing,
possibly repartitions radI/D when onlyglobal=1, and builds diffuse/shadow/surface
fields (2004–2072). Six delay calls all receive the same incoming clock. The
first five updated clocks are discarded; only TgOut's call updates returned
`timeadd` (2075–2083). After shortwave computation firstdaytime becomes zero.

Delay semantics (472–486): morning initializes prior field to current. At
`timeadd >= 59/1440`, the weighted result becomes both output and stored state;
clock resets to timestepdec only if timestepdec > threshold, otherwise zero.
Below the threshold, clock increases first, output is blended, and stored state
remains its previous value. These distinct branches must be captured, not merged.

Night is altitude <= 0 (2102–2141): shortwave, Tg, shadow and F_sh become zero;
Tgwall is scalar zero. Lup uses ambient air emission with special water emission.
LupE/S/W/N alias Lup. CI passes through; CI_Tg/CI_TgG copy it. timeadd resets to
zero and firstdaytime becomes one. Five prior maps and TgOut1 remain unchanged;
TgOut is current Ta. This is a reset flag/clock, not a reset of every state map.

Longwave then runs in either sky mode (2145–2189). Anisotropic CI<0.95 modifies
esky itself; isotropic CI<0.95 blends Ldown while returned esky remains its
original value. Main adds anisotropic cardinal contributions **before** TMRT for
box, **after** TMRT for cylinder (2191–2220). Returned cardinal diagnostics thus
need not be the exact intermediate cardinal fields used for cylinder TMRT.

[Actual return, 2222–2225](https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/0d7fe742abeeddd890dd58fc76ed7f78bd47faec/solweig_gpu/solweig.py#L2222)
has 39 values, in order:
`Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime,
timestepdec, timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast,
Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut,
radI, radD, Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside`.
The abbreviated docstring is not the executable output contract.

## In-place, dtype, and edge hazards

- Device selection independently chooses CUDA whenever available throughout this
  file; passing CPU tensors does not universally enforce CPU execution.
  `ensure_tensor` does not migrate existing tensors. Main reconstructs geometry
  scalars but assumes `.item()` exists on jday/Ta/RH/radG/P/amaxvalue; many helper
  calls create CPU scalar tensors even when helper workspaces use CUDA.
- `torch.zeros` generally has no explicit dtype; workspace and reduction precision
  follow Torch default dtype. `.float()` forces float32 masks/water values; GVF
  directions and cylindric beta explicitly float32 (323, 420). Incoming NumPy
  values and copied tensors can retain other dtypes. In-place `+=` accumulators
  round to their own dtype on every iteration; replacing by a wide reduction
  changes execution. No blanket float64 policy can be inferred from this source.
- Ground-view writes sunwall in place (122) and Tg water in place (130). Lup is
  computed **before** that Tg mutation (128), while local correction at 277 uses
  mutated Tg. Subsequent search directions see mutated Tg. Parallelizing the 18
  directions or snapshotting water too early changes this ordering.
- Ground-view shifted workspaces are allocated once, then only overlapping slices
  are overwritten (136–155, 188–207); outside-slice history persists between ray
  steps. They are not cleared like shadow workspaces. Wall excess emission is
  multiplied at destination by cumulative wall-hit masks (212), not shifted
  from source. Preserve both details until executable evidence classifies them.
- Ray offsets use Torch round and truncated integer conversion (167–186).
  First/second are rounded distances, first is floored to one, but second is not
  floored. `ind=1` is **inside** each ray iteration (217), so first-range snapshots
  are cumulative sums divided by one, not an incrementing denominator. Zero or
  negative second can leave first snapshots/local deletion variables undefined.
  Large distances have no raster-bound termination in this loop; characterize
  slice shape errors rather than adding silent clipping.
- GVF wall mask uses `(wallsun / walls * buildings) == 1` (342): 0/0 becomes NaN
  before exact comparison; partly sunlit walls do not satisfy equality. gvfNorm
  is overridden to one on buildings==0 (395). Only gvf2>1 is clipped (256).
- Shadow 13 starts index one; Shadow 23 starts zero (999,1106). Both check old
  dx/dy/dz at loop entry and compute the next offsets inside the loop. Shadow 13
  clears its workspace and Shadow 23 clears five workspaces each step (1012,
  1125–1129). Do not share GVF's retained-border behavior with these kernels.
  Shadow 23 carries dzprev, cumulative vegetation/building volumes and Boolean
  vegetation ordering; bush threshold is >1. Empty-loop cases require testing.
- `cylindric_wedge` changes global NumPy error settings (415), has unguarded
  tangent/division/square-root domains, and returns potential NaNs/infinities.
  Main replaces only NaNs (2087); do not automatically clamp infinities or F_sh.
- Clearness uses signed latitude, with no G branch for latitude>=90 (861–878),
  pressure -999 sentinel, RH log without floor, and division by I0/I0et without
  floor. I0 NaN is changed to zero but later CI may still be inf/NaN (898–904).
  Python `min(...,1.0)` caps only the upper side and may return a Python float.
- Diffusefraction evaluates beam division before clipping (947–952); zero solar
  altitude, Kt at 0.3/0.78, Ta/RH sentinel/NaN and altitude<1 are distinct edges.
  radD is clipped only above radG; do not invent a lower bound.
- Perez clearness has the exact nested division at 1268; do not substitute a
  textbook equation. Coefficients (1232–1251) and clearness cutpoints
  1.065/1.230/1.500/1.950/2.800/4.500/6.200 (1285–1300) are immutable baseline
  constants. Air mass branches at 10 degrees and zero; Idh<=10 forces brightness
  zero. The altitude<0 branch calls `torch.complex(altitude, 0)` and needs runtime
  characterization. Sky-sun cosine is not clamped before arccos (1328–1334);
  luminance normalization is unguarded.
- Shortwave solid angles use the first patch altitude as band half-width and
  predecessor patch for singleton band (683–693); longwave duplicates this
  schedule (1668–1676). The schedule assumes patch order, not only set membership.
  radTot normalization has no zero guard. `shaded_or_sunlit` excludes equality
  from both sets (558–560). Visibility tests use exact 0/1 and products ==1;
  nonbinary channels cannot be replaced with Boolean without characterization.
- Anisotropic Kside cylinder classifies every building patch with sunlit helper;
  box adds an azimuth-difference gate >90 and <270 (709,757). Cardinal boundaries
  differ between diffuse incidence (`<=`) and reflection (`<`) (722–784).
  The sunlit reflected formula places albedo on beam but not the radD*0.5 term
  (700,738); preserve the actual parentheses.
- Isotropic Kside reaches `del temp_vegsh,temp_vbsh,temp_sh` although those locals
  are assigned only in anisotropic loops (804). This is a static predicted
  UnboundLocalError, not a recorded execution failure. Box direct `torch.where`
  conditions derive from Python azimuth scalars (644–655); runtime acceptance
  must also be tested. Empty patch arrays can trigger deletion failures.
- Longwave model selection is hardcoded model2 (1654), with hardcoded anisotropic
  sky true (1679); normalized emissivity and Ldown_prata are computed but unused.
  model1/3 are definitions, not selectable public behavior here. Model Ta unused.
  Lcyl copies patches to CPU NumPy for unique every call (1646–1649).
- Patch vegetation/sky/building classifications and reflection mask differ
  (1488,1497,1522–1523,1578); they are not interchangeable complements. Second
  longwave sweep depends on completed Ldown_sky and delayed Lup (1576); unsafe
  simultaneous patch writes or combining sweeps loses this dependency.
- Lside's F_sh is locally replaced by `2*F_sh-1` (1786), not caller mutation;
  SVF->angle log/arcsin domains are unguarded (1775–1778). Even anisotropic mode
  evaluates intermediate wall/sky weights before selecting ground-only output.
- Main diurnal temperature denominator can be zero (2048–2049); Tgwall lower
  clamp occurs before clearness scaling, Tg lower clamp only with landcover=1.
  No lower clearness clamp. Night water emissivity is fixed 0.98, not emis_grid.
  At altitude exactly zero nighttime branch runs but altitude<0 copy does not.
- TMRT uses nested sqrt with no negative/zero absorptivity guard (2213).
  Final `+=` cardinal updates are in place, with ordering differing by cyl.

## Constants to preserve and diagnostic capture requirements

SBC is `5.67051e-8` (1992,1461,1637). Thermal emission conversions use `273.15`;
TMRT subtraction uses **273.2** (2213). Ground-view direction sequence is
`5,25,...,345` (18 directions), global averaging denominator 18 and cardinal
9; mixed first/second weight is `(0.5*a+0.4*b)/0.9` (275–282). Delay uses
`33.27` and `59/1440` (475–485). Short/longwave SVF polynomial uses
`63.227,-161.51,156.91,-70.424,16.773,-0.4863`, normalization `4.4897`
(514–516,598,1723–1729,1780). Model1 uses 0.67/0.094, model2 0.308/1.7,
model3 1.8 (1354–1406). Clear-sky Itoa is **1370**, Perez solar constant
**1367** (854,1271); their distinct harmonic expressions must remain distinct.
Latitude/season G table is at 861–887. Low-sun correction is
`0.1473*log(90-zenDegrees)+0.3454` (900,2054). Vapor pressure constants
6.107/7.5/237.3 and Prata constants 46.5/1.2/3.0 are at 1998–2002.
All remaining dewpoint/transmission and diffusefraction coefficient tables are
specified directly at 856–857,890–897 and 932–945; Perez's complete 20-vector
coefficient table is at 1232–1251. This inventory does not authorize rounding or
replacement with constants from another SOLWEIG version.

Executable follow-up must capture five un-delayed GVF Lup fields, albedo and
unshadowed albedo fields, gvfSum/Norm; each delay output and stored field before
and after every step; directional short/longwave and sky/vegetation/wall/reflection
components; diffuse patch tables/order; all 39 main outputs plus input mutations.
Use day/night/morning sequences, water, vegetation, partial sunlit walls, exact
angular/clearness boundaries, retained-border rays and zero/invalid inputs.
Record original failures separately from explicitly patched references. Static
hazards above are hypotheses for characterization, not permission to repair the
upstream oracle silently. No runtime checks or numerical comparisons were run
for this document.
