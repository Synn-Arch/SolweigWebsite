# Upstream comfort, solar and material source audit

Evidence class: static source inspection at upstream commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec`. Completely read:
`calculate_utci.py` (332 lines), `calculate_wbgt.py` (761 lines),
`sun_position.py` (1010 lines), and `Tgmaps_v1.py` (53 lines), all under
`solweig_gpu/`. Truncated combined reads were followed by smaller reads of
the omitted wet-bulb sections. Anchors below refer to these pinned files.
No executable numerical comparison, coefficient reconstruction, independent
scientific validation or golden generation belongs to this audit.

## UTCI function graph and numerical contract

`utci_calculator` (`calculate_utci.py:259`) constructs an invalid mask when
any input is <= -999, selects valid elements, computes saturation vapour
pressure, derives `D_Tmrt = Tmrt - Ta` and vapour pressure in kPa, then calls
`utci_polynomial` (`:24`). The latter evaluates the explicit sixth-degree
four-variable polynomial in the written addition/multiplication order;
powers and many intermediate tensor expressions are recomputed. A port must
preserve the actual coefficient literals and establish parity before Horner
reordering or coefficient specialization. Ta/Tmrt use Celsius and va is the
supplied 10 m wind in m/s. No wind floor, upper wind limit, RH clipping or
polynomial applicability-domain check occurs in these functions.

The humidity conversion creates an eight-element float32 coefficient tensor
without an explicit device (`:311`). It starts with the logarithmic term,
adds seven powers of Kelvin temperature with exponents -2 through 4, then
exponentiates and scales to hPa (`:314`); relative humidity is percent and
division by ten yields kPa. Device placement follows Torch's default-device
state rather than necessarily matching inputs. This is a dependency/device
pitfall to characterize, not an executed CUDA failure claim; individual
coefficient accesses are zero-dimensional tensors.

The returned raster is always float32 (`:327`), including when arithmetic
inputs are float64, with -999 in masked positions. Inputs are expected to be
Torch tensors with compatible masks/shapes; the implementation indexes each
input using the combined mask, so ordinary arithmetic broadcasting does not
guarantee supported calculator broadcasting. NaNs do not satisfy <= -999 and
therefore enter calculation; infinities and out-of-domain Kelvin/logarithm
values receive no separate protection. Coefficient tensors are recreated
for every call. The direct polynomial itself does not impose a dtype or
missing-value policy.

Attribution in the source: Bröde, Fiala, Błażejczyk et al. (2012),
“Deriving the operational procedure for the Universal Thermal Climate
Index (UTCI),” Int J Biometeorol 56:481–494. This records the upstream
attribution; the audit did not independently verify the reference or establish
that every coefficient matches its authoritative published implementation.

## Wet-bulb graph, constants and convergence

The wet-bulb module imports NumPy, Numba vectorize and Torch at module scope
(`calculate_wbgt.py:10`). Its RH entry point (`:691`) calls
`specific_humidity_from_relative_humidity` (`:476`) then
`isobaric_wet_bulb_temperature` (`:512`). The latter calls dewpoint (`:412`),
initializes Tw with the one-third rule, and runs a Newton iteration under
either Warren or Romps equations. Its default is liquid phase/Romps/limit
true. Despite its entry-point docstring describing Kelvin output, the RH
entry point returns `Tw - 273.15` in Celsius (`:715`). The lower-level
isobaric routine returns Kelvin and limits Tw to <= T when requested.
It calls dewpoint with that routine's default limit true independently of
its own limit argument.

Helper graph: effective moist specific heat (`:67`); linear temperature
dependence of latent vaporization/sublimation heats (`:93`, `:110`);
mixed-phase heat (`:127`); vapour pressure (`:156`); saturation vapour
pressure (`:179`) calling latent-heat helpers; saturation specific humidity
(`:241`) calling saturation pressure; relative humidity (`:270`) calling
vapour/saturation pressure. Ice fraction and derivative (`:364`, `:389`)
use a cosine transition between 253.15 and 273.15 K. Mixed dewpoint and
mixed Newton branches use those fraction helpers and saturation derivatives.
RH inversion divides input percent by 100 and does not clip RH.

Constants (`:18` onward): Rd 287, Rv 461.5, eps Rd/Rv; cpd 1005,
cpv 2040, cpl 4220, cpi 2097 J/(kg K); T0 273.16 K; es0 611.657 Pa;
Lv0 2.501e6, Lf0 .333e6, Ls0 their sum in J/kg; transition temperatures
273.15/253.15 K; convergence precision .001 K; iteration cap 20.
These differ in some cases from constants in other upstream meteorological
helpers and must not be silently unified in a port.

`_lambertw` (`:297`) is an explicitly typed Numba vectorized ufunc with
float32 and float64 signatures, nopython true. It approximates the lower
Lambert-W branch using three rational/logarithmic intervals at -1/e, -.34,
-.1 and zero, followed by one refinement. Outside the covered negative
domain its initial NaN persists; refinement also has singular endpoint
expressions that require testing. Replacing it with SciPy Lambert-W changes
the executed algorithm and requires differential evidence.

Mixed dewpoint and both wet-bulb methods stop using a global
`np.nanmax(abs(new-old)) < .001` (`:459`, `:595`, `:672`). Every element keeps
iterating until the array-wide criterion succeeds; per-pixel stopping can
change results. NaNs are ignored in that maximum, and all-NaN inputs may
warn and iterate until the cap. On nonconvergence the routines print a
message and return the last iterate instead of raising or returning a
failure status. Empty arrays and mixtures of fast/slow convergence need
characterization. NumPy helper arithmetic generally preserves or promotes
supplied dtype according to actual operations; no uniform output dtype is
imposed. Ice helpers convert inputs with atleast_1d and return `.item()`
for a single element, changing scalar/array behavior. Pressure is Pa and
temperature Kelvin; no systematic sentinel/domain validation is present.

## Black globe and WBGT integration boundary

`black_globe_temperature` (`calculate_wbgt.py:717`) is a separate Torch
function, not called by the wet-bulb helpers. hcg supplies dtype/device;
emissivity .95 and sigma 5.670374419e-8 are cast to them. It converts Ta/Tmrt
to Kelvin and evaluates a closed-form quartic solution using rounded
constants 1.73205, 3.4943 and .381571 (`:747`). Its k denominator is floored
at `torch.finfo(dtype).tiny`; square-root arguments are clamped to zero;
the final result is Celsius. Neither nonfloating hcg nor device mismatch is
explicitly validated. Overflow/cancellation and zero convection require
tests; substituting a different quartic solver is not automatically parity.

There is no combined WBGT function in this module. Convective coefficient,
sun/shade weighting and driver pressure/temperature conversions belong to
the chronological-driver audit. This audit alone does not establish their
scientific validity or operational WBGT parity.

Upstream comments attribute wet bulb to the AusClimateService atmos package
and Warren (2025), QJRMS 151(766), e4866; latent-heat/specific-heat constants
also name Ambaum, Wagner and Pruß, Feistel and Wagner, and Guildner et al.
The Newton method comments cite Romps (2026), initial guess Knox et al.
(2017), and Lambert approximation Vazquez-Leal et al. (2019). Globe comments
attribute the calculation to Shonk et al. (2026), “UCanWBGT,” e70082. These
are source-reported references, not independently checked bibliographic or
equation verification. Attribution/license review of borrowed atmos code
remains necessary; this file does not carry the GPL header found in the
other three audited files.

## Solar function graph and precision assumptions

`sun_position` (`sun_position.py:26`) executes these helpers in order:
julian_calculation (`:119`), earth_heliocentric_position_calculation (`:189`),
sun_geocentric_position_calculation (`:514`), nutation_calculation (`:528`),
true_obliquity_calculation (`:722`), abberation_correction_calculation (`:741`),
apparent_sun_longitude_calculation (`:748`),
apparent_stime_at_greenwich_calculation (`:755`),
sun_rigth_ascension_calculation (`:773`),
sun_geocentric_declination_calculation (`:786`),
observer_local_hour_calculation (`:797`), topocentric_sun_position_calculate
(`:806`), topocentric_local_hour_calculate (`:843`) and
sun_topocentric_zenith_angle_calculate (`:850`). Several call set_to_range
(`:887`). Misspelled helper names are the actual upstream identifiers.

Inputs are dictionaries containing local calendar fields and a supplied
fixed UTC offset, plus latitude/longitude degrees and altitude meters.
Julian calculation subtracts UTC from the local hour. Its datetime-object
fallback assumes UTC zero without processing tzinfo. Calendar handling
uses a Julian/Gregorian switch in 1582; the nonexistent-date branch mutates
the supplied dictionary after D was already computed, so its warning should
not be taken as verified corrected-date arithmetic. Delta_t actually equals
zero (`:180`) despite a higher-level comment mentioning 33.184 seconds.

Ephemeris coefficient arrays use NumPy's inferred floating dtype, ordinarily
float64; L5 and R4 one-row arrays leave size-one array fields in returned
dictionaries. The model is effectively scalar, with Python `if` in
set_to_range; larger vector inputs are not a supported vector API. Series
reductions use NumPy sum/dot ordering. Angle normalization subtracts
max_interval times floor(var/max_interval) and conditionally adds it; this
is tailored to zero-based 360-degree ranges, not a generic interval-width
normalizer.

Topocentric parallax uses altitude/6378140 and .99664719 (`:806`). Refraction
(`:860`) uses `1.02/[60*tan(elevation + 10.3/(elevation+5.11))]` without a
horizon cutoff or pressure/temperature correction. It applies even to
negative elevations; singular/shallow-angle behavior needs fixtures.
Azimuth is navigation convention, north zero/east 90, wrapped to [0,360);
zenith includes refraction. The source attributes SPA to Reda and Andreas
(2004), Solar Energy 76(5):577–589. Its accuracy statement is a docstring
claim, not measured evidence for this particular port or its constants.
SciPy interpolation/rotate imports occur even though no audited function
calls them, adding import-time dependencies.

## Met solar timeline and vegetation season

`Solweig_2015a_metdata_noload` (`sun_position.py:905`) requires a 2-D met
array with year/day-of-year/hour/minute in columns 0–3. It returns eight
values `(YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax)` (`:1009`),
not the larger tuple advertised in its docstring. Seven outputs have shape
(1, number_of_rows) and NumPy default float64; dectime is a 1-D expression
from met dtype. Altitude/azimuth are degrees, zen radians, decimal time day
of year plus local hour/minute fractions.

For two or more rows, every solar evaluation subtracts half the interval
between the first two rows (`:935`, `:980`); later gaps or changed cadence
do not alter that half interval. One-row input subtracts zero. Decimal
time itself stays unshifted. Solar seconds are fixed zero even when a
fractional offset would have seconds. Year/day-of-year reconstructs dates
with integer conversion, while hour/minute enter timedelta directly.
Timezone/DST choice is external and supplied as one UTC offset for all rows.

Maximum altitude is searched in 15-minute steps beginning at 10:15 local
time until altitude decreases (`:959`). Search executes on the first row
or the decimal-time modulo-integer predicate, normally midnight; otherwise
previous maximum carries forward. This is not a globally bracketed daily
maximum search and needs polar/high-longitude/timezone edge tests. Solar
zenith in (89,90] degrees is forced to 89 (`:990`), setting that apparent
near-horizon altitude to 1 degree. Day-of-year output comes from the original
row date rather than the half-step-shifted solar date (`:1002`).

Leaf thresholds are 97 and 300, but the predicate is OR:
`(doy > 97) | (doy < 300)` (`:1004`). It is true for every ordinary finite
day-of-year, so leafon is always one on ordinary calendar inputs. This
source quirk must be preserved in compatibility behavior until a separately
versioned scientific policy authorizes correction.
The chronological driver subsequently replaces this returned leafon array with
its own seasonal predicate (`utci_process.py:597–605`); this helper defect alone
does not establish an always-leaf-on full simulation.

## Material lookup and aliasing

`Tgmaps_v1` (`Tgmaps_v1.py:15`) takes a land-cover array and six-column
lookup table `[id, albedo, emissivity, TgK, Tstart, TmaxLST]`. It starts five
output arrays as copies of the class array, preserving its dtype, and
iterates sorted unique original ids. Each assignment matches the **current
output array's values**, not the original land-cover mask (`:37`). If an
earlier replacement equals a later class id, those cells can be remapped
again; all five property fields have this aliasing risk. Integer class grids
also truncate fractional property assignments. This actual behavior must
be characterized before replacing it with direct indexed lookup.

Missing/duplicate lookup matches are not explicitly checked and can cause
assignment shape errors. Class 99 supplies wall thermal parameters using
`np.where` indexing (`:46`), yielding arrays whose dimensional shape must
be characterized; the routine does not scalarize them or validate uniqueness.
It returns five raster properties and three wall properties. Upstream driver
land-cover normalization is separate; this helper itself performs none.
GPL 3-or-later notices and Kamath/Sudharsan attribution appear in UTCI,
solar and material files and must remain in derivative implementations.

## Verification still required

Freeze coefficient checksum and independent UTCI cases; test float32/64,
device/default-device states, sentinel/nonfinite masks and all-invalid input.
Characterize Lambert endpoints/domain, RH extremes, mixed-phase thresholds,
array-wide convergence and empty/all-NaN/nonconverged arrays; check globe
energy-balance residuals and driver sun/shade WBGT selection. Compare solar
results against an independent documented SPA case while retaining a
separate compatibility reference; cover leap/year boundaries, irregular
cadence, one row, fixed-offset DST effects, maximum-altitude search and
refraction singularities. Reproduce all-season leafon and material remapping
aliasing with targeted fixtures. Check third-party scientific attribution
and licensing. No change to constants or source quirks is authorized by
this audit; no numerical tolerance is frozen here.
