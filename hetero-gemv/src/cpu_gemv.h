// CPU-side int4 GEMV paths for the heterogeneous split.
//
// Two of them, because "use the CPU's matrix units" turns out to be a question
// rather than an instruction:
//
//   neon   - dequantise 32 weights at a time in NEON registers and FMA straight
//            against x. Reads only the int4 bytes; never materialises fp32.
//   amx    - dequantise a block of rows into an fp32 scratch that stays in L2,
//            then hand that block to Accelerate's cblas_sgemv, which is the
//            documented route to the CPU matrix/AMX path.
//
// The AMX route has to write and re-read K floats per row. That is 8x the bytes
// of the int4 row it came from, and only pays off if the scratch really does
// stay in cache.

#pragma once
#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include <cstdint>
#include <cstring>
#include <vector>

typedef uint16_t h16;
static inline float h2f_(h16 h) { __fp16 v; memcpy(&v, &h, 2); return (float)v; }

// Unpack 32 nibbles (16 bytes) and accumulate q[i] * x[i] into `acc`.
//
// Byte i holds weight 2i in its low nibble and weight 2i+1 in its high nibble,
// so masking gives the even weights and shifting gives the odd ones; zip1/zip2
// interleave them back into weight order without a single scalar extract.
static inline float32x4_t chunk32(const uint8_t* wp, const float* xp,
                                  float32x4_t acc) {
    uint8x16_t v  = vld1q_u8(wp);
    uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0F));
    uint8x16_t hi = vshrq_n_u8(v, 4);
    uint8x16_t z[2] = { vzip1q_u8(lo, hi), vzip2q_u8(lo, hi) };
    for (int h = 0; h < 2; ++h) {
        uint16x8_t a = vmovl_u8(vget_low_u8(z[h]));
        uint16x8_t b = vmovl_u8(vget_high_u8(z[h]));
        acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a))),  vld1q_f32(xp + h*16 + 0));
        acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a))), vld1q_f32(xp + h*16 + 4));
        acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_low_u16(b))),  vld1q_f32(xp + h*16 + 8));
        acc = vfmaq_f32(acc, vcvtq_f32_u32(vmovl_u16(vget_high_u16(b))), vld1q_f32(xp + h*16 + 12));
    }
    return acc;
}

// fp16 path. The weights are 4-bit and x is already fp16, so the whole product
// fits comfortably in half precision: a lane accumulates 4 products of at most
// 15 x 1, so |acc| <= 60 where fp16 still has 0.06 resolution. Staying in fp16
// halves the vector ops per weight -- 8 lanes per register instead of 4, and one
// uint16->fp16 convert instead of a widen-to-uint32 plus a convert-to-fp32.
// Each chunk's fp16 partial is folded into an fp32 accumulator so the row sum
// itself never accumulates in half precision.
static inline void cpu_gemv_neon16(const uint8_t* W, const h16* SB, const __fp16* xh,
                                   const float* xs, float* y,
                                   uint32_t K, uint32_t G, uint32_t m0, uint32_t m1) {
    const uint32_t chunks = K / 32, cpg = G / 32, groups = K / G;
    for (uint32_t m = m0; m < m1; ++m) {
        const uint8_t* wp = W + (size_t)m * (K / 2);
        const h16*     sb = SB + (size_t)m * groups * 2;
        float accf = 0.0f;
        for (uint32_t j = 0; j < chunks; ++j) {
            const uint8_t* p = wp + j * 16;
            const __fp16*  x = xh + j * 32;
            uint8x16_t v  = vld1q_u8(p);
            uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0F));
            uint8x16_t hi = vshrq_n_u8(v, 4);
            uint8x16_t z[2] = { vzip1q_u8(lo, hi), vzip2q_u8(lo, hi) };
            float16x8_t acc = vdupq_n_f16((__fp16)0.0f);
            for (int h = 0; h < 2; ++h) {
                float16x8_t a = vcvtq_f16_u16(vmovl_u8(vget_low_u8(z[h])));
                float16x8_t b = vcvtq_f16_u16(vmovl_u8(vget_high_u8(z[h])));
                acc = vfmaq_f16(acc, a, vld1q_f16(x + h * 16 + 0));
                acc = vfmaq_f16(acc, b, vld1q_f16(x + h * 16 + 8));
            }
            float d = (float)vaddvq_f32(vcvt_f32_f16(vget_low_f16(acc)))
                    + (float)vaddvq_f32(vcvt_f32_f16(vget_high_f16(acc)));
            uint32_t g = j / cpg;
            accf += h2f_(sb[2*g]) * d + h2f_(sb[2*g+1]) * xs[j];
        }
        y[m] = accf;
    }
}

// y[m] for m in [m0, m1), fp32 NEON path. Layout matches the Metal kernel exactly:
// W packed nibbles, SB interleaved (scale,bias) half2 per group, xs per 32.
static inline void cpu_gemv_neon(const uint8_t* W, const h16* SB, const float* xf,
                                 const float* xs, float* y,
                                 uint32_t K, uint32_t G, uint32_t m0, uint32_t m1) {
    const uint32_t chunks = K / 32, cpg = G / 32, groups = K / G;
    for (uint32_t m = m0; m < m1; ++m) {
        const uint8_t* wp = W + (size_t)m * (K / 2);
        const h16*     sb = SB + (size_t)m * groups * 2;
        float accf = 0.0f;
        for (uint32_t j = 0; j < chunks; ++j) {
            float32x4_t a = chunk32(wp + j * 16, xf + j * 32, vdupq_n_f32(0.0f));
            uint32_t g = j / cpg;
            accf += h2f_(sb[2*g]) * vaddvq_f32(a) + h2f_(sb[2*g+1]) * xs[j];
        }
        y[m] = accf;
    }
}

// Same rows, but dequantise into an fp32 block and let Accelerate do the matvec.
static inline void cpu_gemv_amx(const uint8_t* W, const h16* SB, const float* xf,
                                float* y, float* scratch, uint32_t block,
                                uint32_t K, uint32_t G, uint32_t m0, uint32_t m1) {
    const uint32_t chunks = K / 32, cpg = G / 32, groups = K / G;
    for (uint32_t base = m0; base < m1; base += block) {
        uint32_t n = std::min(block, m1 - base);
        for (uint32_t r = 0; r < n; ++r) {
            const uint8_t* wp = W + (size_t)(base + r) * (K / 2);
            const h16*     sb = SB + (size_t)(base + r) * groups * 2;
            float* out = scratch + (size_t)r * K;
            for (uint32_t j = 0; j < chunks; ++j) {
                uint32_t g = j / cpg;
                float32x4_t s = vdupq_n_f32(h2f_(sb[2*g]));
                float32x4_t b = vdupq_n_f32(h2f_(sb[2*g+1]));
                uint8x16_t v  = vld1q_u8(wp + j * 16);
                uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0F));
                uint8x16_t hi = vshrq_n_u8(v, 4);
                uint8x16_t z[2] = { vzip1q_u8(lo, hi), vzip2q_u8(lo, hi) };
                for (int h = 0; h < 2; ++h) {
                    uint16x8_t A = vmovl_u8(vget_low_u8(z[h]));
                    uint16x8_t B = vmovl_u8(vget_high_u8(z[h]));
                    uint32x4_t q[4] = { vmovl_u16(vget_low_u16(A)), vmovl_u16(vget_high_u16(A)),
                                        vmovl_u16(vget_low_u16(B)), vmovl_u16(vget_high_u16(B)) };
                    for (int t = 0; t < 4; ++t)
                        vst1q_f32(out + j*32 + h*16 + t*4,
                                  vfmaq_f32(b, s, vcvtq_f32_u32(q[t])));
                }
            }
        }
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)n, (int)K, 1.0f,
                    scratch, (int)K, xf, 1, 0.0f, y + base, 1);
    }
}
