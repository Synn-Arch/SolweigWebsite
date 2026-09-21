# Upstream wrapper and preprocessing source audit

Evidence: static reading of the complete `solweig_gpu/solweig_gpu.py` and
`solweig_gpu/preprocessor.py` at upstream commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. All line anchors below refer to
that commit. This report establishes source behavior and potential failure
paths; it does not claim executable integration parity. Downstream
`create_inputs`, wind, walls, shadow and radiation implementations require
their separate audits.

## Seven public wrappers

| Workflow and source anchor | Routing, defaults and observable behavior |
|---|---|
| `preprocess`, `solweig_gpu.py:22` | Defaults: Building_DSM/DEM/Trees TIFFs, no land cover or wind coefficients, tile size 3600, overlap 20, own met true, UHI true. Optional start/end/source/folder/met/output-directory values are None. Creates the supplied preprocessing directory as given, or `{base_path}/processed_inputs`, calls `ppr` with the arguments, then returns the directory string. |
| `build_inputs`, `solweig_gpu.py:136` | Lazily imports `run_create_inputs`; forwards lat/lon, city None, buffer 8 km, reductions 3/1 km, base folder None, resolution 2 m. Returns the downstream result directly. Download/credential behavior is outside this wrapper audit. |
| `build_wind_ext_coeff`, `solweig_gpu.py:184` | Input and ERA5 directories required; physical parameters keyword-only. Directions 0 through 330 by 30; z0_ref .03, hmin_b/hmin_t 1, z_eval/zref 10, LAI_t 2, a0_t .5, a1_t .4, alpha_min/max .2/2.5, coefficient min/max .1/1, lp_min_open .02, workers None. Lazily imports and forwards everything to `calculate_wind_ext_coeff`; returns `str(Path(input_dir))` without resolving to an absolute path. |
| `run_walls_aspect`, `solweig_gpu.py:255` | No execution options exposed; calls `run_parallel_processing` on preprocessing `Building_DSM`, `walls`, and `aspect` paths. Implicit None return. |
| `calculate_svf`, `solweig_gpu.py:274` | Its base path is the preprocessing directory. Patch option 2 and overwrite false. Requires Building_DSM/DEM/Trees directories; intersects their filename keys, sorts lexically, errors on no common key and warns about missing DEM/Trees. Skips a tile only if all three expected TIFF/ZIP/NPZ files exist. Calls `svf_calculator(save_rasters=True)` serially by tile; deletes returned tensors and empties available CUDA cache. Implicit None return. |
| `run_utci_tiles`, `solweig_gpu.py:387` | Creates output paths under `{base_path}/output_folder/{key}`. Required maps intersect DSM/Trees/DEM/met/walls/aspect; a nonempty optional landcover map further restricts the intersection, whereas wind does not. A requested tile-key list errors only if its intersection is empty, permitting partial matches. No requested list and no matching tiles produces an empty loop. Keys must split into two integer coordinates for numeric sorting. Loads met with `np.loadtxt(skiprows=1, delimiter=' ')`, calls `compute_utci`, forwards ten save flags, then unconditionally calls CUDA empty-cache. TMRT defaults true; all other save flags false; UTCI is downstream unconditional. Implicit None return. |
| `thermal_comfort`, `solweig_gpu.py:500` | Defaults match preprocess except roughness discovery true; output flags match run_utci_tiles. If roughness discovery is enabled it calls wind generation first and catches every Exception, reporting a generic missing-ERA5 warning regardless of actual cause. On success passes base_path as wind input. Calls preprocess, walls, standalone SVF with patch option 2/overwrite false, then all UTCI tiles. Standalone SVF is generated even when save_svf is false. Implicit None return. |

The wrapper module itself imports only typing/pathlib at module scope. Calling
preprocess imports the entire preprocessor, whose module-level imports include
netCDF4, xarray, pandas, Shapely, matplotlib, timezonefinder, SciPy and GDAL
(`preprocessor.py:14`). Thus lazy wrapper imports do not isolate optional
forcing dependencies from an upstream own-met execution. SVF and runtime
wrappers explicitly import Torch. These are source dependency findings, not
proof of any particular installed environment's import failure.

## Raster paths, validation and logical tiles

`ppr` joins base_path with DSM/DEM/Trees names (`preprocessor.py:1434`);
absolute names follow normal `os.path.join` behavior. Landcover and wind use
the explicit absolute-or-base-relative resolver (`:282`). A user-supplied
preprocess_dir is used relative to the process working directory, not joined
to base_path. Own-met paths go directly to `shutil.copy` (`:1395`), so relative
own_met_file paths are relative to the working directory. The copier's
base_path argument is unused.

`check_rasters` (`:237`) compares dimensions, pixel width/height and literal
projection strings. It does not compare raster origins, rotation coefficients,
square pixels, projected meter units or semantic CRS equivalence, nor inspect
NoData/values. Consequently acceptance is weaker than geospatial alignment.
`ppr` catches ValueError from validation, prints and exits(1); other exception
classes propagate (`:1450`).

Tiling (`:341`) enforces `0 <= overlap < tilesize`; source windows advance by
tilesize and extend by overlap toward increasing x/y offsets, clipped at the
edge. Names encode source pixel offsets `prefix_i_j`, not tile ordinal numbers.
A tile covering both dimensions is `prefix_0_0` with original dimensions.
GDAL Translate preserves its own source-window metadata semantics; read-back
verification remains necessary. Normal layer folders are deleted before
retiling (`:418`); wind folders are deleted if coefficients are supplied
(`:387`). Optional Landcover/WindCoeff folders are not deleted when their
inputs become absent, so stale optional tiles can survive a repeated call.
SVF folders likewise survive retiling, and existence-based wrapper reuse does
not verify changed input geometry or patch option (`solweig_gpu.py:364`).

Wind discovery (`preprocessor.py:290`) accepts a directory containing
WindCoeff_dir*.tif, a glob, or a legacy single file. If any filename matches
the directional three-digit suffix regex, nondirectional files are discarded
and all twelve standard directions must be present; extra directions are not
rejected. Missing input paths or standard directions raise FileNotFoundError.

## Own meteorology

`create_met_files` (`:1369`) clears metfiles and copies the source file verbatim
for every Building_DSM tile with the expected prefix. It neither parses nor
filters dates, start/end, missing values or units. `selected_date_str`,
start/end and use_uhi do not transform own-met content in `ppr` (`:1488`).
The main runtime therefore receives every source row; downstream date/solar
semantics need separate characterization. A one-row file is loaded as a 1-D
array by the wrapper's unqualified loadtxt, a potential downstream shape
failure requiring an executable check.

## ERA5 normalization and conversion

`_normalize_time_coord` (`:456`) first renames valid_time to time, otherwise
adds time and step, stacking a 2-D forecast grid, or attempts decode_cf for
plain time. Numeric step conversion first uses `pd.to_timedelta` without an
explicit unit and only retries hours on an exception; it must not be assumed
to interpret numeric hours correctly. The step-only swap-dims branch and
time+step stack/drop-vars branch require actual xarray fixtures. There is no
explicit monotonicity/duplicate-time policy.

Standard ERA5 (`:505`) reads fixed instant/accum filenames, parses inclusive
UTC start/end bounds, intersects stream timestamps, and rejects empty streams
or intersections. It uses t2m/d2m/sp/u10/v10 and ssrd; RH uses the Magnus
formula and clips to 0–100. Wind direction is `(270 - degrees(atan2(v,u))) %
360`, with calm speed <= .01 assigned zero. SSRD is divided by 3600 without
inspection of actual accumulation interval or deaccumulation. Longwave is
not written. Output T2/PSFC values remain K/Pa, but standard-mode metadata
labels them degC/kPa (`:606`); later met conversion subtracts 273.15/divides
1000 regardless of labels. Time is encoded as Gregorian hours since 1970.
Standard writing explicitly transposes to time/latitude/longitude; UHI
writing takes raw variable arrays, assuming their dimension order (`:786`).
Opened xarray datasets are not explicitly closed by either ERA5 function.

UHI ERA5 (`:661`) selects an additional 24 hours on each side, intersects
available timestamps, computes UHI on available padded rows, then trims to
the requested interval. Padding does not guarantee complete nights if files
are short. No explicit post-trim empty check precedes the final first/last
timestamp print. This variant writes correct K/Pa unit labels.

## UHI amplitude and calendar behavior

Timezone selection uses the middle ERA5 coordinate indices (`:45`) and
converts UTC into local naive timestamps (`:61`) before daily resampling.
Daily amplitude (`:126`) is
`[mean(SWDOWN/(1.225*1005)) * (Tmax-Tmin)^3 / max(mean(wind),1)]^0.25`,
with nonfinite/negative numerator normalized to zero and missing amplitude
filled zero. Daily amplitudes are forward-filled onto original timestamps.

Night detection actually uses SWDOWN <= 5 W/m² (`:91`), despite the helper
docstring's <= 0 statement. Finite contiguous index segments receive a sine
profile from 0 to pi multiplied by the mean segment amplitude; segments
shorter than two samples or nonpositive amplitude remain zero. The helper
default extends one sample after night, but its UHI caller explicitly passes
zero extensions (`:183`). Segment shape uses sample indices, not elapsed
time, so gaps can affect physical interpretation. Removing timezone offsets
can create duplicated DST local timestamps; their xarray resampling outcome
is not verified here.

## WRF behavior and source-level defect

Strict filename parsing (`:208`) accepts d01–d09 and HH_MM_SS, HH:MM:SS or
HH timestamps, returning `(datetime, domain_integer)`. Discovery ignores
nonmatching names and sorts by this tuple. The time-window list comprehension
at `:881` compares a datetime directly against that tuple. Python does not
order these types; this is a source-level TypeError path for discovered valid
WRF files, not an executed failure result in this audit. A reference repair
must be explicit and separately provenanced.

Beyond that defect, the function creates a synthetic hourly timeline from
requested bounds rather than reading actual file timestamps; concatenated
file arrays may not match its length. Domains are not selected independently.
It reads T2/Q2/PSFC/TSK/SWDOWN and winds; GLW is commented out. If both
COSALPHA/SINALPHA exist it rotates grid winds into Earth coordinates (`:912`),
expanding 2-D rotation arrays; otherwise uses raw winds. RH clips to 0–100,
wind direction/calm semantics match ERA5, and lat/lon come from the first
file's first time slice. Output units are K/Pa. WRF does not compute UHI_CYCLE.

## NetCDF-to-tile met sampling

`process_metfiles` (`:1079`) builds an axis-aligned geographic bounding box
from four raster corners computed without rotation coefficients. Failed
transforms fall back to assuming coordinates are EPSG:4326. It infers each
tile's timezone, selects UTC data belonging to the selected local day from
00:00 to 23:59, then writes local year/day/hour/minute. Thus timezone and
forcing aggregation depend on logical tile context.

Sampling uses a KD-tree in raw lon/lat coordinate space; estimated cell sizes
use haversine neighbor distances (`:1031`). If both cell dimensions exceed
tile dimensions, selects the nearest center. Otherwise averages grid centers
inside the geographic box, falling back to nearest center when no points or
no finite mean exist. Wind direction receives an arithmetic mean, not a
circular mean (`:1290`), then wraps modulo 360. Sampling exceptions/missing
variables become -999; UHI errors/missing/disabled values become zero.

Generated files have 25 columns, ending Wd/uhii, with rain before Kdn
(`:1129`); temperature converts K to Celsius and pressure Pa to kPa. The
intermediate row order puts Kdn before rain and DataFrame reordering corrects
it. Wind prints five decimals; other physical fields print two. Filenames
are `metfile_i_j_YYYY-MM-DD.txt` (`:1188`), distinct from copied own-met names.
Unlike the own-met copier this function does not clear old metfiles, allowing
multiple dates/stale files to coexist. Downstream key mapping behavior must
be checked separately. Output Gregorian calendar decoding does not explicitly
forward the NetCDF time variable's calendar attribute.

## Unresolved executable checks

Required follow-up: reference wrapper artifact/return/error snapshots;
single-row own-met and wrong-date rows; absolute/relative met paths; stale
optional layers and changed-geometry caches; all ERA5 time layouts and
accumulation intervals; empty UHI trim and short padded nights; DST duplicates
and gaps; WRF defect reproduction and authorized repair policy; multitime
files/domain selection/timeline length; rotated rasters, offsets and CRS
representations; circular-direction edge cases; generated-met key selection
with multiple dates. No tolerances, speedups, or scientific corrections are
established by this static audit.
