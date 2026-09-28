// lw_primary.ispc -- B7-20 candidate C_native, first island (longwave primary reducer)
//
// Exact port of solweig_light.radiation.cylinder_longwave._longwave_primary
// (source blob 27ba6ce48454399c7b97285c8408511d050a32da, frozen at HEAD
// dca2035c4046ac4387f694a6c7b66967e590500c) for the v7 CPU backend track.
//
// Execution model: one ISPC gang per pixel block; lanes own independent
// pixels (programIndex over the pixel index). Patch sweeps are strictly
// sequential per lane; the ten per-lane float32 accumulators a0..a9 map 1:1
// to the numba baseline's accum[] array. No cross-lane reduction exists.
//
// float32 rounding node map (numba baseline -> this kernel):
//   np.float32(f32 + f32)      -> f32 add (the np.float32 wrap is a typed
//                                 no-op in the baseline IR)
//   np.float32(f32 + f64chain) -> (float)((double)acc + chain)  f64 add,
//                                 f64->f32 round-to-nearest-even cast
//   bool * float               -> (bool ? 1.0 : 0.0) * float  the multiply
//                                 is kept so NaN/Inf and signed-zero
//                                 propagate exactly as in the baseline
//   np.float32(np.pi)          -> bit reinterpretation of 0x40490FDB
//   == / != float comparisons -> ordered compares; the baseline's int64
//                                 literal compares promote to float64,
//                                 which is result-identical for these
//                                 exactly-representable f32 operands
//
// The solar_gate branch is uniform per patch and compiled as a real branch:
// the inactive sun/shade expressions are never evaluated.
//
// Build contract: -O2 --opt=disable-fma --math-lib=default, no --fast-math.
// The emitted assembly is audited for contraction (fmadd/fmla/fmsub/fnm*)
// and the audit log ships with the evidence. Mixed mul+add nodes here are
// separate IR operations in the baseline (fastmath=False), so contraction
// would change the arithmetic graph even where rounding happens to agree.

static inline float f32_pi() {
    uniform uint32 bits = 0x40490FDBu; // np.float32(np.pi), exact bits
    return *((uniform float *uniform)&bits);
}

// SUFFIX: exported symbol suffix identifying the surface-scalar
// specialization. ST: arithmetic type of the vegetation/sun/shade surface
// chains and of the accumulator add feeding their np.float32 rounding node.
//   f32: surface_sun/surface_sh are float32 scalars; chains stay float32.
//   f64: surface_sun/surface_sh are float64 scalars; chains stay float64 and
//        round to float32 only at the accumulator add, as numba emits it.
// Everything else (sky chain, building/sky/veg/mask predicates, reflection,
// second sweep, output folds) is float32 in both specializations.
#define LW_PRIMARY(SUFFIX, ST)                                                          \
export void lw_primary_##SUFFIX(                                                        \
        uniform const float *uniform sh,        /* [B*P] float32 visibility */          \
        uniform const float *uniform vs,        /* [B*P] */                             \
        uniform const float *uniform vb,        /* [B*P] */                             \
        uniform const uint8 *uniform sun,       /* [B*P] uint8 0/1 */                   \
        uniform const uint8 *uniform shade,     /* [B*P] uint8 0/1 */                   \
        uniform const float *uniform solid,     /* [P] */                               \
        uniform const float *uniform sine,      /* [P] */                               \
        uniform const float *uniform cosine,    /* [P] */                               \
        uniform const uint8 *uniform solar_gate,/* [P] uint8 0/1 */                     \
        uniform const float *uniform sky_down,  /* [P], element stride sd_str */        \
        uniform int64 sd_str,                                                           \
        uniform const float *uniform sky_side,  /* [P], element stride ss_str */        \
        uniform int64 ss_str,                                                           \
        uniform ST surface_sun, uniform ST surface_sh,                                  \
        uniform const float *uniform lup,       /* [B] */                               \
        uniform float reflection_factor,                                                \
        uniform int64 B, uniform int64 P,                                               \
        uniform float *uniform out)             /* [B*7], owned by caller */            \
{                                                                                       \
    foreach (x = 0 ... B) {                                                             \
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;                    \
        float a5 = 0.0f, a6 = 0.0f, a7 = 0.0f, a8 = 0.0f, a9 = 0.0f;                    \
        const int64 base = (int64)x * P;                                                \
        for (uniform int64 p = 0; p < P; ++p) {                                         \
            const float shv = sh[base + p];                                             \
            const float vsv = vs[base + p];                                             \
            const float vbv = vb[base + p];                                             \
            const bool sky = (shv == 1.0f) && (vsv == 1.0f);                            \
            const bool veg = (vsv == 0.0f) || (vbv == 0.0f);                            \
            const bool building = ((1.0f - shv) * vbv) == 1.0f;                         \
            const float skyf = sky ? 1.0f : 0.0f;                                       \
            const float vegf = veg ? 1.0f : 0.0f;                                       \
            const float buif = building ? 1.0f : 0.0f;                                  \
            a0 = a0 + skyf * sky_down[p * sd_str];                                      \
            a5 = a5 + skyf * sky_side[p * ss_str];                                      \
            const ST vside = (((ST)surface_sh * (ST)solid[p]) * (ST)cosine[p]) * (ST)vegf;  \
            const ST vdown = (((ST)surface_sh * (ST)solid[p]) * (ST)sine[p]) * (ST)vegf;    \
            a6 = (float)((ST)a6 + vside);                                               \
            a1 = (float)((ST)a1 + vdown);                                               \
            if (solar_gate[p] != 0) {                                                   \
                const float sunf = (sun[base + p] != 0) ? 1.0f : 0.0f;                  \
                const float shaf = (shade[base + p] != 0) ? 1.0f : 0.0f;                \
                const ST sun_side = ((((ST)surface_sun * (ST)sunf) * (ST)solid[p]) * (ST)cosine[p]) * (ST)buif;      \
                const ST shade_side = ((((ST)surface_sh * (ST)shaf) * (ST)solid[p]) * (ST)cosine[p]) * (ST)buif;     \
                const ST sun_down = ((((ST)surface_sun * (ST)sunf) * (ST)solid[p]) * (ST)sine[p]) * (ST)buif;        \
                const ST shade_down = ((((ST)surface_sh * (ST)shaf) * (ST)solid[p]) * (ST)sine[p]) * (ST)buif;       \
                a8 = (float)((ST)a8 + sun_side);                                        \
                a7 = (float)((ST)a7 + shade_side);                                      \
                a3 = (float)((ST)a3 + sun_down);                                        \
                a2 = (float)((ST)a2 + shade_down);                                      \
            } else {                                                                    \
                const ST shade_side = (((ST)surface_sh * (ST)solid[p]) * (ST)cosine[p]) * (ST)buif; \
                const ST shade_down = (((ST)surface_sh * (ST)solid[p]) * (ST)sine[p]) * (ST)buif;   \
                a7 = (float)((ST)a7 + shade_side);                                      \
                a2 = (float)((ST)a2 + shade_down);                                      \
            }                                                                           \
        }                                                                               \
        const float r0 = a0 + lup[x];                                                   \
        const float r1 = r0 * reflection_factor;                                        \
        const float r2 = r1 * 0.5f;                                                     \
        const float reflected = r2 / f32_pi();                                          \
        for (uniform int64 p = 0; p < P; ++p) {                                         \
            const float shv = sh[base + p];                                             \
            const float vsv = vs[base + p];                                             \
            const float vbv = vb[base + p];                                             \
            const bool occl = (shv == 0.0f) || (vsv == 0.0f) || (vbv == 0.0f);          \
            const float occlf = occl ? 1.0f : 0.0f;                                     \
            a9 = a9 + ((reflected * solid[p]) * cosine[p]) * occlf;                     \
            a4 = a4 + ((reflected * solid[p]) * sine[p]) * occlf;                       \
        }                                                                               \
        out[x * 7 + 0] = (((a0 + a1) + a2) + a3) + a4;                                  \
        out[x * 7 + 1] = (((a5 + a6) + a7) + a8) + a9;                                  \
        out[x * 7 + 2] = a5;                                                            \
        out[x * 7 + 3] = a6;                                                            \
        out[x * 7 + 4] = a7;                                                            \
        out[x * 7 + 5] = a8;                                                            \
        out[x * 7 + 6] = a9;                                                            \
    }                                                                                   \
}

LW_PRIMARY(f32, float)
LW_PRIMARY(f64, double)
