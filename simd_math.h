#ifndef __SIMD_MATH_H__
#define __SIMD_MATH_H__

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

#if defined(_MSC_VER)
#define SIMD_MATH_API(type) static inline type __vectorcall
#define SIMD_MATH_ALIGN16_BEG __declspec(align(16))
#define SIMD_MATH_ALIGN16_END
#else
#define SIMD_MATH_API(type) static inline type
#define SIMD_MATH_ALIGN16_BEG
#define SIMD_MATH_ALIGN16_END __attribute__((aligned(16)))
#endif

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#define SIMD_MATH_CONST_INT(name, value) \
    static const SIMD_MATH_ALIGN16_BEG int name[4] SIMD_MATH_ALIGN16_END = {value, value, value, value}
#define SIMD_MATH_CONST_FLOAT(name, value) \
    static const SIMD_MATH_ALIGN16_BEG float name[4] SIMD_MATH_ALIGN16_END = {value, value, value, value}

#define SIMD_SHUFFLE(fp3,fp2,fp1,fp0) (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | ((fp0)))

typedef __m128 float4;
typedef __m128i int4;
typedef __m128d double2;
typedef __m256 float8;
typedef __m256i int8;
typedef __m256d double4;

typedef SIMD_MATH_ALIGN16_BEG struct
{
    float4 r[4];
} float4x4 SIMD_MATH_ALIGN16_END;

typedef struct
{
    float x;
    float y;
    float z;
} float3;

typedef struct
{
    int x;
    int y;
    int z;
} int3;

typedef struct
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
} uint3;

typedef struct
{
    uint16_t x;
    uint16_t y;
    uint16_t z;
} half3;

typedef struct
{
    uint16_t x;
    uint16_t y;
    uint16_t z;
    uint16_t w;
} half4;

typedef struct
{
    int16_t x;
    int16_t y;
    int16_t z;
} short3;

typedef struct
{
    uint16_t x;
    uint16_t y;
    uint16_t z;
} ushort3;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4201)
// C4201: nonstandard extension used : nameless struct/union
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
typedef struct
{
    union
    {
        struct
        {
            float _11, _12, _13;
            float _21, _22, _23;
            float _31, _32, _33;
        };
        float m[3][3];
    };
} float3x3;

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define float4_shuffle(v1, v2, shuffle) _mm_shuffle_ps(v1, v2, shuffle)

#define float4_permute(v, shuffle) _mm_shuffle_ps(v, v, shuffle)

SIMD_MATH_API(float4) float4_set(float x, float y, float z, float w)
{
    return _mm_set_ps(w, z, y, x);
}

SIMD_MATH_API(float4)float4_set1(float x)
{
    return _mm_set_ps1(x);
}

SIMD_MATH_API(float4) float4_zero()
{
    return _mm_setzero_ps();
}

SIMD_MATH_API(float4) float4_one()
{
    return _mm_set_ps1(1.0f);
}

SIMD_MATH_API(float4) float4_identity()
{
    return float4_set(0.f, 0.f, 0.f, 1.f);
}

SIMD_MATH_API(float4) float4_add(float4 v1, float4 v2)
{
    return _mm_add_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_sub(float4 v1, float4 v2)
{
    return _mm_sub_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_mul(float4 v1, float4 v2)
{
    return _mm_mul_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_div(float4 v1, float4 v2)
{
    return _mm_div_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_scale(float4 v, float s)
{
    float4 scale = float4_set1(s);
    return float4_mul(v, scale);
}

SIMD_MATH_API(float4) float4_min(float4 v1, float4 v2)
{
    return _mm_min_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_max(float4 v1, float4 v2)
{
    return _mm_max_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_clamp(float4 v, float4 min, float4 max)
{
    return float4_min(float4_max(v, min), max);
}

SIMD_MATH_API(float4) float4_and(float4 v1, float4 v2)
{
    return _mm_and_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_or(float4 v1, float4 v2)
{
    return _mm_or_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_cmple(float4 v1, float4 v2)
{
    return _mm_cmple_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_cmplt(float4 v1, float4 v2)
{
    return _mm_cmplt_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_cmpge(float4 v1, float4 v2)
{
    return _mm_cmpge_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_cmpgt(float4 v1, float4 v2)
{
    return _mm_cmpgt_ps(v1, v2);
}

SIMD_MATH_API(float4) float4_cmpeq(float4 v1, float4 v2)
{
    return _mm_cmpeq_ps(v1, v2);
}

SIMD_MATH_API(int4) float4_cast_int4(float4 v)
{
    return _mm_castps_si128(v);
}

SIMD_MATH_API(int4) float4_to_int4(float4 v)
{
    return _mm_cvtps_epi32(v);
}

SIMD_MATH_API(int4) float4_truncate(float4 v)
{
    return _mm_cvttps_epi32(v);
}

SIMD_MATH_API(float4) float4_fmadd(float4 v1, float4 v2, float4 v3)
{
    return _mm_fmadd_ps(v1, v2, v3);
}

SIMD_MATH_API(float) float4_hmax(float4 v)
{
    float4 tmp1 = float4_max(v, float4_permute(v, SIMD_SHUFFLE(2, 3, 0, 1)));
    float4 tmp2 = float4_max(tmp1, float4_permute(tmp1, SIMD_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(tmp2);
}

SIMD_MATH_API(float) float4_hmin(float4 v)
{
    float4 tmp1 = float4_min(v, float4_permute(v, SIMD_SHUFFLE(2, 3, 0, 1)));
    float4 tmp2 = float4_min(tmp1, float4_permute(tmp1, SIMD_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(tmp2);
}


SIMD_MATH_API(float4) int4_to_float4(int4 v)
{
    return _mm_cvtepi32_ps(v);
}

SIMD_MATH_API(float4) int4_cast_float4(int4 v)
{
    return _mm_castsi128_ps(v);
}

SIMD_MATH_API(int4) int4_set(int x, int y, int z, int w)
{
    return _mm_set_epi32(w, z, y, x);
}

SIMD_MATH_API(int4) int4_set1(int x)
{
    return _mm_set1_epi32(x);
}

SIMD_MATH_API(int4) int4_zero()
{
    return _mm_setzero_si128();
}

SIMD_MATH_API(int4) int4_one()
{
    return _mm_set1_epi32(1);
}

SIMD_MATH_API(int4) int4_add(int4 v1, int4 v2)
{
    return _mm_add_epi32(v1, v2);
}

SIMD_MATH_API(int4) int4_sub(int4 v1, int4 v2)
{
    return _mm_sub_epi32(v1, v2);
}

SIMD_MATH_API(int4) int4_and(int4 v1, int4 v2)
{
    return _mm_and_si128(v1, v2);
}

SIMD_MATH_API(int4) int4_xor(int4 v1, int4 v2)
{
    return _mm_xor_epi32(v1, v2);
}

SIMD_MATH_API(int4) int4_or(int4 v1, int4 v2)
{
    return _mm_or_epi32(v1, v2);
}

SIMD_MATH_API(int4) int4_max(int4 v1, int4 v2)
{
    return _mm_max_epi32(v1, v2);
}

SIMD_MATH_API(int4) int4_min(int4 v1, int4 v2)
{
    return _mm_min_epi32(v1, v2);
}

SIMD_MATH_API(int4) int4_clamp(int4 v, int4 min, int4 max)
{
    return int4_min(int4_max(v, min), max);
}

#define int4_shr(v, count) _mm_srli_epi32(v, count)

#define int4_shl(v, count) _mm_slli_epi32(v, count)

SIMD_MATH_API(float4) float4_log(float4 v)
{
    /* the smallest non denormalized float number */
    SIMD_MATH_CONST_INT(min_norm_pos, 0x00800000);
    SIMD_MATH_CONST_INT(inv_mant_mask, ~0x7f800000);
    SIMD_MATH_CONST_FLOAT(cephes_SQRTHF, 0.707106781186547524f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p0, 7.0376836292E-2f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p1, -1.1514610310E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p2, 1.1676998740E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p3, -1.2420140846E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p4, +1.4249322787E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p5, -1.6668057665E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p6, +2.0000714765E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p7, -2.4999993993E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_p8, +3.3333331174E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_log_q1, -2.12194440e-4f);
    SIMD_MATH_CONST_FLOAT(cephes_log_q2, 0.693359375f);

    float4 e;
    int4 emm0;
    float4 one = float4_one();
    float4 invalid_mask = float4_cmple(v, float4_zero());
    float4 ps0p5 = float4_set1(0.5f);

    v = float4_max(v, *(float4*)min_norm_pos);

    emm0 = int4_shr(float4_cast_int4(v), 23);

    /* keep only the fractional part */
    v = float4_and(v, *(float4*)inv_mant_mask);
    v = float4_or(v, ps0p5);

    emm0 = int4_sub(emm0, int4_set1(0x7f));
    e = int4_to_float4(emm0);
    e = float4_add(e, one);

    /* part2:
      if( x < SQRTHF ) {
      e -= 1;
      x = x + x - 1.0;
      } else { x = x - 1.0; }
    */
    {
        float4 y, z;
        float4 mask = float4_cmplt(v, *(float4*)cephes_SQRTHF);
        float4 tmp = float4_and(v, mask);
        v = float4_sub(v, one);
        e = float4_sub(e, float4_and(one, mask));
        v = float4_add(v, tmp);

        z = float4_mul(v, v);
        y = *(float4*)cephes_log_p0;

        y = float4_fmadd(y, v, *(float4*)cephes_log_p1);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p2);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p3);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p4);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p5);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p6);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p7);
        y = float4_fmadd(y, v, *(float4*)cephes_log_p8);
        y = float4_mul(y, v);

        y = float4_mul(y, z);

        y = float4_fmadd(e, *(float4*)cephes_log_q1, y);

        y = float4_fmadd(z, float4_set1(-0.5f), y);

        v = float4_add(v, y);
        v = float4_fmadd(e, *(float4*)cephes_log_q2, v);
        v = float4_or(v, invalid_mask); // negative arg will be NAN
    }
    return v;
}

SIMD_MATH_API(float4) float4_exp(float4 v)
{
    SIMD_MATH_CONST_FLOAT(exp_hi, 88.3762626647949f);
    SIMD_MATH_CONST_FLOAT(exp_lo, -88.3762626647949f);

    SIMD_MATH_CONST_FLOAT(cephes_LOG2EF, 1.44269504088896341f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_C1, 0.693359375f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_C2, -2.12194440e-4f);

    SIMD_MATH_CONST_FLOAT(cephes_exp_p0, 1.9875691500E-4f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_p1, 1.3981999507E-3f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_p2, 8.3334519073E-3f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_p3, 4.1665795894E-2f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_p4, 1.6666665459E-1f);
    SIMD_MATH_CONST_FLOAT(cephes_exp_p5, 5.0000001201E-1f);

    float4 fx, mask, y, z, pow2n;
    float4 tmp = float4_zero();
    int4 emm0;
    float4 one = float4_one();
    float4 ps0p5 = float4_set1(0.5f);

    v = float4_min(v, *(float4*)exp_hi);
    v = float4_max(v, *(float4*)exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = float4_fmadd(v, *(float4*)cephes_LOG2EF, ps0p5);

    emm0 = float4_truncate(fx);
    tmp = int4_to_float4(emm0);

    /* if greater, subtract 1 */
    mask = float4_cmpgt(tmp, fx);
    mask = float4_and(mask, one);
    fx = float4_sub(tmp, mask);

    tmp = float4_mul(fx, *(float4*)cephes_exp_C1);
    z = float4_mul(fx, *(float4*)cephes_exp_C2);
    v = float4_sub(v, tmp);
    v = float4_sub(v, z);

    z = float4_mul(v, v);

    y = *(float4*)cephes_exp_p0;
    y = float4_fmadd(y, v, *(float4*)cephes_exp_p1);
    y = float4_fmadd(y, v, *(float4*)cephes_exp_p2);
    y = float4_fmadd(y, v, *(float4*)cephes_exp_p3);
    y = float4_fmadd(y, v, *(float4*)cephes_exp_p4);
    y = float4_fmadd(y, v, *(float4*)cephes_exp_p5);
    y = float4_fmadd(y, z, v);
    y = float4_add(y, one);

    /* build 2^n */
    emm0 = float4_truncate(fx);
    emm0 = int4_add(emm0, int4_set1(0x7f));
    emm0 = int4_shl(emm0, 23);
    pow2n = int4_cast_float4(emm0);

    y = float4_mul(y, pow2n);

    return y;
}

SIMD_MATH_API(float4) float4_pow(float4 v1, float4 v2)
{
    return float4_exp(float4_mul(v2, float4_log(v1)));
}

SIMD_MATH_API(float4) float3_load(const float3* source)
{
    float4 xy = _mm_castpd_ps(_mm_load_sd((const double*)source));
    float4 z = _mm_load_ss(&source->z);
    return _mm_insert_ps(xy, z, 0x20);
}

SIMD_MATH_API(void) float3_store(float3* dest, float4 v)
{
    _mm_store_sd((double*)(dest), _mm_castps_pd(v));
    float4 z = float4_permute(v, SIMD_SHUFFLE(2, 2, 2, 2));
    _mm_store_ss(&dest->z, z);
}

SIMD_MATH_API(int4) int3_load(const int3* source)
{
    float4 xy = _mm_castpd_ps(_mm_load_sd((const double*)(source)));
    float4 z = _mm_load_ss((const float*)(&source->z));
    return float4_cast_int4(_mm_insert_ps(xy, z, 0x20));
}

SIMD_MATH_API(void) int3_store(int3* dest, int4 v)
{
    float4 z;
    float4 vf = int4_cast_float4(v);
    _mm_store_sd((double*)(dest), _mm_castps_pd(vf));
    z = float4_permute(vf, SIMD_SHUFFLE(2, 2, 2, 2));
    _mm_store_ss((float*)&dest[2], z);
}

SIMD_MATH_API(float4) half3_load(const half3* source)
{
    int4 xy = _mm_loadu_si32(source);
    return _mm_cvtph_ps(_mm_insert_epi32(xy, source->z, 1));
}

SIMD_MATH_API(void) half3_store(half3* dest, float4 v)
{
    int4 packed = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si32(dest, packed);
    dest->z = (uint16_t)_mm_extract_epi16(packed, 2);
}

SIMD_MATH_API(int4) ushort3_load(const ushort3* source)
{
    int4 xy = _mm_loadu_si32(source);
    xy = _mm_insert_epi32(xy, _mm_extract_epi16(xy, 1), 1);
    xy = int4_and(xy, int4_set(0xFFFF, 0xFFFF, 0, 0));
    return _mm_insert_epi32(xy, source->z, 2);
}

SIMD_MATH_API(void) ushort3_store(ushort3* dest, int4 v)
{
    v = int4_clamp(v, int4_zero(), int4_set1(65535));
    // SSE packing uses signed rules, so manually pack items instead
    dest->x = (uint16_t)_mm_extract_epi16(v, 0);
    dest->y = (uint16_t)_mm_extract_epi16(v, 2);
    dest->z = (uint16_t)_mm_extract_epi16(v, 4);
}

SIMD_MATH_API(float4x4) float3x3_load(const float3x3* source)
{
    float4 z = float4_zero();

    float4 v1 = _mm_loadu_ps(&source->m[0][0]);
    float4 v2 = _mm_loadu_ps(&source->m[1][1]);
    float4 v3 = _mm_load_ss(&source->m[2][2]);

    float4 t1 = _mm_unpackhi_ps(v1, z);
    float4 t2 = _mm_unpacklo_ps(v2, z);
    float4 t3 = float4_shuffle(v3, t2, SIMD_SHUFFLE(0, 1, 0, 0));
    float4 t4 = _mm_movehl_ps(t2, t3);
    float4 t5 = _mm_movehl_ps(z, t1);

    float4x4 m;
    m.r[0] = _mm_movelh_ps(v1, t1);
    m.r[1] = float4_add(t4, t5);
    m.r[2] = float4_shuffle(v2, v3, SIMD_SHUFFLE(1, 0, 3, 2));
    m.r[3] = float4_identity();
    return m;
}

SIMD_MATH_API(float4) float3_transform(float4 v, float4x4 m)
{
    float4 res = float4_permute(v, SIMD_SHUFFLE(2, 2, 2, 2)); // Z
    res = float4_fmadd(res, m.r[2], m.r[3]);
    float4 tmp = float4_permute(v, SIMD_SHUFFLE(1, 1, 1, 1)); // Y
    res = float4_fmadd(tmp, m.r[1], res);
    tmp = float4_permute(v, SIMD_SHUFFLE(0, 0, 0, 0)); // X
    res = float4_fmadd(tmp, m.r[0], res);
    return res;
}

SIMD_MATH_API(float) half_to_float(uint16_t half)
{
    int4 v1 = _mm_cvtsi32_si128(half);
    float4 v2 = _mm_cvtph_ps(v1);
    return _mm_cvtss_f32(v2);
}

SIMD_MATH_API(float4x4) float4x4_transpose(float4x4 m)
{
    float8 t0 = _mm256_castps128_ps256(m.r[0]);
    t0 = _mm256_insertf128_ps(t0, m.r[1], 1);
    float8 t1 = _mm256_castps128_ps256(m.r[2]);
    t1 = _mm256_insertf128_ps(t1, m.r[3], 1);

    float8 vTemp = _mm256_unpacklo_ps(t0, t1);
    float8 vTemp2 = _mm256_unpackhi_ps(t0, t1);
    float8 vTemp3 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x20);
    float8 vTemp4 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x31);
    vTemp = _mm256_unpacklo_ps(vTemp3, vTemp4);
    vTemp2 = _mm256_unpackhi_ps(vTemp3, vTemp4);
    t0 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x20);
    t1 = _mm256_permute2f128_ps(vTemp, vTemp2, 0x31);

    float4x4 res;
    res.r[0] = _mm256_castps256_ps128(t0);
    res.r[1] = _mm256_extractf128_ps(t0, 1);
    res.r[2] = _mm256_castps256_ps128(t1);
    res.r[3] = _mm256_extractf128_ps(t1, 1);
    return res;
}

SIMD_MATH_API(float4) float4_pq_inv_eotf(float4 v) {
    const float qM1 = 1305 / 8192.f;
    const float qM2 = 2523 / 32.f;
    const float qC1 = 107 / 128.f;
    const float qC2 = 2413 / 128.f;
    const float qC3 = 2392 / 128.f;

    float4 m1v = float4_set1(qM1);
    float4 m2v = float4_set1(qM2);
    float4 c1v = float4_set1(qC1);
    float4 idv = float4_set1(1.0f);
    return float4_pow(
        float4_div(
            float4_add(c1v, float4_scale(float4_pow(v, m1v), qC2)),
            float4_add(idv, float4_scale(float4_pow(v, m1v), qC3))),
        m2v);
}

SIMD_MATH_API(float4) float4_saturate(float4 v)
{
    return float4_clamp(v, float4_zero(), float4_one());
}

#ifdef __cplusplus
}
#endif

#endif // __SIMD_MATH_H__
