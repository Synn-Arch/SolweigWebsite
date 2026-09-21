# Geometry audit of the pinned upstream

This is a static source audit of `nvnsudharsan/SOLWEIG-GPU` at commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. The files were read from the
pinned checkout; neither module was imported and no raster or shadow case was
executed. Claims below labelled “observed” come directly from source. Items in
the final section are hypotheses that require an executable oracle.

## Function graph

The wall/aspect path is:

```text
run_parallel_processing
  -> process_file_parallel (one process-pool job per .tif)
       -> findwalls
       -> filter1Goodwin_as_aspect_v3
            -> get_ders
                 -> cart2pol
       -> GDAL GeoTIFF writes
```

`run_parallel_processing` creates `walls/` and `aspect/`, enumerates visible
`.tif` files, and submits every file to a `ProcessPoolExecutor`; completion is
consumed with `as_completed` ([`walls_aspect.py:271-309`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L271-L309)). A worker reads band 1 as
`float32`, obtains geotransform/projection, computes the two arrays, and writes
`walls_{filename[13:-4]}.tif` and `aspect_{filename[13:-4]}.tif`
([`walls_aspect.py:168-213`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L168-L213)). It retries each output creation up to three times with delays, but a
worker returns the filename after reporting most errors; the caller does not
validate the returned status ([`walls_aspect.py:215-269`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L215-L269)).

The SVF path is:

```text
svf_calculator
  -> load_raster_to_tensor                 [standalone save mode only]
  -> create_patches
  -> for each patch: shadow
       -> ensure_tensor
  -> annulus_weight                         [per patch and annulus]
  -> save_svf_zip_npz_outputs               [save mode only]
       -> save_raster_like_gdal
            -> tensor_to_numpy
```

The complete graph and the call order are visible at
[`shadow.py:35-164`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L35-L164) and
[`shadow.py:409-624`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L409-L624).
`svf_calculator` stores three per-pixel/per-patch matrices while iterating
patches, then returns 19 values (the 15 SVF fields, three visibility matrices,
and `SVFtotal`) ([`shadow.py:510-624`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L510-L624)).

## Walls and aspect

`findwalls` allocates a zero-valued array with the input shape. For each
interior location it examines the four-neighbour cross selected by
`[[0,1,0],[1,0,1],[0,1,0]]`, stores the maximum neighbour elevation, subtracts
the center DSM, and sets values below `walllimit` to zero. The outermost rows
and columns are then forced to zero ([`walls_aspect.py:27-58`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L27-L58)). The module default threshold is
`walllimit = 3.0` ([`walls_aspect.py:24-25`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L24-L25)). Since the comparison is `<`, an exact difference of 3.0 remains
in the result; this is observed source behavior, not a tested numerical claim.

`filter1Goodwin_as_aspect_v3` derives a filter size as
`floor((scale + 1e-10) * 9)`, clamps it to 3 when at most 2, and increments
other even sizes to odd except size 9 ([`walls_aspect.py:114-123`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L114-L123)). It builds a vertical line filter and a horizontal two-sided build filter, then scans rotations `h = 0..179` in order. The wall filter uses `scipy.ndimage.rotate(..., order=1, reshape=False, mode='nearest')` followed by `np.round`; the build filter uses order 0 and the same reshape/border mode ([`walls_aspect.py:125-146`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L125-L146)). At `h` 30 and 150 the last build-filter column is cleared. At indices 225 and 135, the corresponding diagonal corners are forced into the wall filter.

For each wall pixel, `z` is updated only when the current wall-filter sum is
strictly greater than the stored sum. Equal sums retain the earlier rotation.
The side label `x` is 1 only when the side-1 DSM sum is strictly greater than
side 2; equality selects side 2. The direction `y` is set to `index = 270-h`.
After the scan, side-1 directions subtract 180 degrees and negative values are
wrapped by adding 360 ([`walls_aspect.py:148-164`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L148-L164)). Thus tie behavior is order-dependent and must be preserved by a port. Pixels whose resulting `y` is zero fall back to the DSM derivative aspect in degrees. `get_ders` uses `np.gradient`, `cart2pol`, `arctan`, sign inversion, and one-time addition of `2π` for negative aspect ([`walls_aspect.py:78-97`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L78-L97)).

The nested filter loops do not visit every edge: their ranges are bounded by
`filthalveceil - 1` and `row/col - filthalveceil - 1`
([`walls_aspect.py:148-154`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L148-L154)).
The source worker writes float32 GeoTIFFs and copies only geotransform and
projection, with no explicit nodata or metadata copying
([`walls_aspect.py:190-213`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L190-L213),
[`walls_aspect.py:223-240`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L223-L240)).

## Sky patches and order

`create_patches` has four implemented options. The exact altitude bands,
azimuth starts, and patches per band are:

| option | altitude centers | patches per band | total |
|---|---|---|---:|
| 1 | 6, 18, 30, 42, 54, 66, 78, 90 | 30, 30, 24, 24, 18, 12, 6, 1 | 145 |
| 2 | 6, 18, 30, 42, 54, 66, 78, 90 | 31, 30, 28, 24, 19, 13, 7, 1 | 153 |
| 3 | 6, 18, 30, 42, 54, 66, 78, 90 | 62, 60, 56, 48, 38, 26, 14, 1 | 305 |
| 4 | 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90 | 62, 62, 60, 60, 56, 56, 48, 48, 38, 38, 26, 26, 14, 14, 1 | 609 |

These values and the `annulino`, `azistart`, and `skyvaultaziint = 360 /
patches_in_band` construction are literal source data
([`shadow.py:352-406`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L352-L406)). The docstring's “144 or 2304” description is stale relative to the four branches; no branch handles those numbers directly ([`shadow.py:359-371`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L359-L371)).

Within each band, patch order is increasing `k`: altitude is repeated and
azimuth is `k * (360 / count) + azistart`. `svf_calculator` reconstructs the
azimuth list in the same band-major order, subtracting 360 only when a value is
strictly greater than 360 ([`shadow.py:534-540`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L534-L540)). It then iterates bands and integer azimuth positions, calls `shadow`, and writes the three visibility channels at the current linear index before accumulating weighted SVF fields ([`shadow.py:542-585`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L542-L585)).

The three matrices are allocated as `(rows, cols, sum(aziinterval))`; here
`aziinterval` is the per-band patch-count tensor returned by
`create_patches`, so its final dimension is 145, 153, 305, or 609
([`shadow.py:526-532`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L526-L532)). The source allocates them as Torch default floating tensors and does not cast them before computation. The legacy NPZ writer later casts each to float32.

## Shadow ray recurrence

`shadow` converts azimuth and altitude from degrees to radians. Exact azimuth
zero is replaced with `1e-12` before conversion ([`shadow.py:194-201`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L194-L201)). It initializes scalar Torch tensors `dx=dy=dz=0`, `index=1`, and full-domain buffers. `f` starts as a clone of the DSM `a`; `bushplant` is `(bush > 1.).float()`, and `vegsh` starts as that bush mask ([`shadow.py:203-225`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L203-L225)).

The loop condition is evaluated before each step:

```text
amaxvalue >= dz and abs(dx) < sizex and abs(dy) < sizey
```

Thus the initial state always attempts step 1 when `amaxvalue` is nonnegative;
the first step is not skipped because `dz`, `dx`, and `dy` start at zero
([`shadow.py:240-255`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L240-L255)). For azimuth sectors `[45°,135°)` or `[225°,315°)`, the recurrence sets
`dy = sign(sin(azimuth))*index`,
`dx = -sign(cos(azimuth))*abs(round(index/tan(azimuth)))`, and `ds = abs(1/sin(azimuth))`. In the other sectors it sets
`dy = sign(sin(azimuth))*abs(round(index*tan(azimuth)))`,
`dx = -sign(cos(azimuth))*index`, and `ds = abs(1/cos(azimuth))`. In both cases
`dz = ds * index * tan(altitude) / scale` ([`shadow.py:227-255`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L227-L255)). Rounding is exactly the framework call `torch.round`; conversion of offsets to slice bounds uses Python `int(...)` ([`shadow.py:246-269`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L246-L269)).

Each iteration clears temporary arrays, copies shifted source slices minus `dz`
into destination slices, and updates `f = max(f, temp)`. Building shadow is
`sh = 1` where `f > a` during the loop and zero otherwise
([`shadow.py:257-277`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L257-L277)). Canopy and trunk comparisons are `tempvegdem > a` and `tempvegdem2 > a`; their float difference is merged into `vegsh`, and vegetation shadow is cleared where it overlaps building shadow ([`shadow.py:279-286`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L279-L286)).

The first-step branch computes `firstvegdem = tempvegdem - temp`, replaces
nonpositive values with 1000, sets `vegsh` to 1 where `firstvegdem < dz`,
multiplies it by `(vegdem2 > a)`, and clears accumulated combined vegetation
shadow ([`shadow.py:288-293`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L288-L293)). Bush handling is conditional on both `bush.max() > 0` and a positive
`max(fabovea * bush)`. It tracks a shifted maximum `g` only on `bushplant`,
then after the ray loop compares `g - bush`, clips positive/negative values to
binary states, subtracts the initial bush mask from `vegsh`, and restores the
result ([`shadow.py:295-316`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L295-L316)). Finalization complements `sh`, thresholds combined vegetation shadow
to binary before subtracting, then complements `vegsh` and `vbshvegsh`
([`shadow.py:303-316`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L303-L316)).

The source explicitly calls `torch.cuda.empty_cache()` after deleting temporary
arrays even when the selected device is CPU ([`shadow.py:318-322`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L318-L322)).

## SVF weighting and output schemas

For every patch, the code accumulates annulus weights from `annulus_weight`.
The weight uses a `90`-degree normalization, the azimuth step `360/aziinterval`,
and sine terms evaluated in Torch ([`shadow.py:324-350`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L324-L350)). Directional fields classify azimuth into East `[0,180)`, South `[90,270)`, West `[180,360)`, and North `[270,360) ∪ [0,90)` ([`shadow.py:555-583`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L555-L583)). After accumulation, `svfS` and `svfW` receive `3.0459e-4`; vegetation directional fields receive the same value through `last` where `vegdem2 == 0`. All SVF fields are then upper-clipped at 1, and `SVFtotal = svf - (1 - svfveg) * (1 - 0.03)` ([`shadow.py:586-612`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L586-L612)).

When saving, `save_svf_zip_npz_outputs` writes these artifacts, with an optional
`_{number}` suffix ([`shadow.py:100-120`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L100-L120)):

```text
svfs[_{number}].zip
shadowmats[_{number}].npz
SkyViewFactor[_{number}].tif
```

The ZIP contains 15 root-level, float32 GeoTIFFs with exactly these member
names: `svf.tif`, `svfE.tif`, `svfS.tif`, `svfW.tif`, `svfN.tif`,
`svfveg.tif`, `svfEveg.tif`, `svfSveg.tif`, `svfWveg.tif`, `svfNveg.tif`,
`svfaveg.tif`, `svfEaveg.tif`, `svfSaveg.tif`, `svfWaveg.tif`, and
`svfNaveg.tif` ([`shadow.py:125-152`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L125-L152)). Temporary files use those same fixed names directly in `output_dir`, are deleted after ZIP creation, and the standalone total GeoTIFF is written afterward ([`shadow.py:143-164`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L143-L164)).

The NPZ has exactly three named members: `shadowmat`, `vegshadowmat`, and
`vbshmat`. Each is converted to CPU NumPy and cast to float32 before compressed
writing; the source matrices have `(rows, cols, patches)` order
([`shadow.py:159-164`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L159-L164),
[`shadow.py:530-532`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L530-L532)).
`save_raster_like_gdal` copies only the template geotransform and projection and
writes one float32 band ([`shadow.py:71-97`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L71-L97)).

## Concurrency and cache hazards observed in source

The wall path uses one process per submitted tile, with a bounded executor
(`min(32, cpu_count)` on non-Windows and `min(8, cpu_count//2)` on Windows),
but does not coordinate output ownership beyond one worker per input filename
([`walls_aspect.py:271-309`](../../.upstream/SOLWEIG-GPU/solweig_gpu/walls_aspect.py#L271-L309)).
The SVF writer deletes an existing ZIP, overwrites fixed temporary TIFF names,
and removes those names after zipping ([`shadow.py:118-155`](../../.upstream/SOLWEIG-GPU/solweig_gpu/shadow.py#L118-L155)). Concurrent calls sharing an `output_dir` can therefore race on the
ZIP, temporary TIFFs, and cleanup. There is no lock, unique attempt directory,
atomic completion marker, or manifest in this code. This is a source-observed
hazard; it has not been reproduced by execution in this audit.

## Untested hypotheses and required fixtures

The following points must be established with a pinned CPU oracle and, where
available, a separate CUDA oracle before numerical behavior is frozen:

- The exact Torch dtype of scalar tensors created by `ensure_tensor`, the dtype
  of `torch.round`, and CPU/GPU behavior at half-integer ray offsets.
- Whether all three visibility channels are exactly binary on normal, border,
  first-step, bush, and fractional-height fixtures. The source performs
  thresholding, but no source inspection alone proves the complete value set
  before finalization.
- Slice behavior for negative and zero ray offsets at cardinal and near-cardinal
  azimuths, including whether empty slices silently preserve zero temporary
  arrays as expected.
- Numerical effects of `amaxvalue`, altitude 90 degrees, `scale`, NaNs, and
  negative or zero vegetation values on loop termination and bush predicates.
- Whether the order-dependent wall ties and rotated-filter interpolation produce
  the same arrays under a NumPy/Numba port at all boundary and exact-tie cases.
- ZIP member byte order/compression metadata and GeoTIFF metadata beyond the
  explicitly copied geotransform, projection, dimensions, and float32 band.

No item in this section is evidence of equivalence or a performance result.
