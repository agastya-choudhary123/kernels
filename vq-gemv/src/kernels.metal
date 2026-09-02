// Fused codebook (vector-quantised) dequant-matmul for LLM decode, Apple M4.
//
// Scalar int4 stores one 4-bit index per weight into an implicit uniform
// lattice. Vector quantisation stores one 8-bit index per *pair* of weights
// into a learned 256-entry codebook of 2-vectors, so the codewords sit where
// the weights actually are instead of on a uniform grid.
//
// The two formats are deliberately the same size:
//     affine int4 : 4 bits/weight + fp16 scale + fp16 bias per 64  = 4.500 bits
//     VQ d=2 K=256: 4 bits/weight + fp16 scale         per 32      = 4.500 bits
// so any quality difference is a property of the quantiser, not of the budget.
// The codebook itself is 256 x half2 = 1 KB per matrix, which at these sizes is
// under 0.001 bits/weight.
//
// Layout:
//   IDX   [M, K/2]  uint8, index of the codeword for weights (2i, 2i+1)
//   C     [256][2]  half, one codebook per matrix
//   S     [M, K/32] half, one scale per 32 weights
//   w[m, 2i+t] = S[m, i/16] * C[IDX[m,i]][t]
//
// Note what is absent: there is no bias term and therefore no sum-of-x prepass.
// The codebook is free to be asymmetric, so it absorbs the group mean that the
// affine format has to carry in a separate fp16 per group.
//
// One 16-byte (uint4) load per lane covers 16 indices = 32 weights = exactly one
// scale group, so the scale index IS the chunk index and the inner loop needs no
// division at all.

#include <metal_stdlib>
using namespace metal;

struct VQParams {
    uint M;        // rows
    uint K;        // reduction length
    uint chunks_pr;// 32-weight chunks per row = K/32 = uint4 loads per row
    uint cb_size;  // codebook entries (256)
};

// ---------------------------------------------------------------------------
// 4 codebook lookups + 4 two-element dots, from one uint of packed indices.
// ---------------------------------------------------------------------------
inline float vqdot4(uint w, threadgroup const half2* C, const device half2* xp) {
    half2 c0 = C[ w        & 0xFF];
    half2 c1 = C[(w >>  8) & 0xFF];
    half2 c2 = C[(w >> 16) & 0xFF];
    half2 c3 = C[(w >> 24) & 0xFF];
    return float(dot(c0, xp[0])) + float(dot(c1, xp[1]))
         + float(dot(c2, xp[2])) + float(dot(c3, xp[3]));
}

// REP copies of the codebook, interleaved. Copy `l` of codeword `c` lives at
// word c*REP + l, so lane l reading codeword c hits bank (c*REP + l) % 32.
// At REP=32 that is exactly l for every c: conflict-free by construction.
// At REP=1 it degenerates to the plain shared codebook, c % 32, which collides.
constant uint REP [[function_constant(0)]];

inline float vqdot4_rep(uint w, threadgroup const half2* C, uint lane,
                        const device half2* xp) {
    uint l = lane & (REP - 1);
    half2 c0 = C[( w        & 0xFF) * REP + l];
    half2 c1 = C[((w >>  8) & 0xFF) * REP + l];
    half2 c2 = C[((w >> 16) & 0xFF) * REP + l];
    half2 c3 = C[((w >> 24) & 0xFF) * REP + l];
    return float(dot(c0, xp[0])) + float(dot(c1, xp[1]))
         + float(dot(c2, xp[2])) + float(dot(c3, xp[3]));
}

// ---------------------------------------------------------------------------
// v0: scalar reference. One thread per row.
// ---------------------------------------------------------------------------
kernel void gemv_vq_scalar(const device uint4*  IDX [[buffer(0)]],
                           const device half2*  C   [[buffer(1)]],
                           const device half*   S   [[buffer(2)]],
                           const device half2*  x   [[buffer(3)]],
                           device float*        y   [[buffer(4)]],
                           constant VQParams&   P   [[buffer(5)]],
                           uint m [[thread_position_in_grid]]) {
    if (m >= P.M) return;
    const device uint4* irow = IDX + (ulong)m * P.chunks_pr;
    const device half*  srow = S   + (ulong)m * P.chunks_pr;
    float acc = 0.0f;
    for (uint j = 0; j < P.chunks_pr; ++j) {
        uint4 p = irow[j];
        const device half2* xp = x + j * 16;
        float d = 0.0f;
        uint ws[4] = { p.x, p.y, p.z, p.w };
        for (uint t = 0; t < 4; ++t) {
            uint w = ws[t];
            for (uint b = 0; b < 4; ++b) {
                half2 c = C[(w >> (8 * b)) & 0xFF];
                half2 xv = xp[t * 4 + b];
                d += float(c.x) * float(xv.x) + float(c.y) * float(xv.y);
            }
        }
        acc += float(srow[j]) * d;
    }
    y[m] = acc;
}

// ---------------------------------------------------------------------------
// v1: one simdgroup per row, uint4 index loads, codebook shared in threadgroup
// memory (256 x half2 = 1 KB).
//
// This is the obvious kernel, and it has a bank-conflict problem. Apple's
// threadgroup memory is 32 banks of 4 bytes; a half2 codeword is exactly one
// 4-byte word, so lane l reading codeword c hits bank c % 32. The 32 lanes of a
// simdgroup read 32 unrelated codewords, so the accesses collide like balls in
// bins -- expected worst bank depth around 4, and the hardware serialises them.
// ---------------------------------------------------------------------------
kernel void gemv_vq_shared(const device uint4*  IDX [[buffer(0)]],
                           const device half2*  C   [[buffer(1)]],
                           const device half*   S   [[buffer(2)]],
                           const device half2*  x   [[buffer(3)]],
                           device float*        y   [[buffer(4)]],
                           constant VQParams&   P   [[buffer(5)]],
                           threadgroup half2*   Ct  [[threadgroup(0)]],
                           uint  tg_id [[threadgroup_position_in_grid]],
                           uint  tid   [[thread_index_in_threadgroup]],
                           uint  n_thr [[threads_per_threadgroup]],
                           uint  sg_in [[simdgroup_index_in_threadgroup]],
                           uint  n_sg  [[simdgroups_per_threadgroup]],
                           uint  lane  [[thread_index_in_simdgroup]]) {
    for (uint i = tid; i < P.cb_size; i += n_thr) Ct[i] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m = tg_id * n_sg + sg_in;
    if (m >= P.M) return;

    const device uint4* irow = IDX + (ulong)m * P.chunks_pr;
    const device half*  srow = S   + (ulong)m * P.chunks_pr;

    float acc = 0.0f;
    for (uint j = lane; j < P.chunks_pr; j += 32) {
        uint4 p = irow[j];
        const device half2* xp = x + j * 16;
        float d = vqdot4(p.x, Ct, xp)      + vqdot4(p.y, Ct, xp + 4)
                + vqdot4(p.z, Ct, xp + 8)  + vqdot4(p.w, Ct, xp + 12);
        acc += float(srow[j]) * d;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}

// ---------------------------------------------------------------------------
// v2: lane-replicated codebook -- bank conflicts removed by construction.
//
// Store 32 copies of the codebook interleaved so that copy `l` of codeword `c`
// lives at word index c*32 + l. Lane l always reads its own copy, so its bank is
// (c*32 + l) % 32 = l, whatever c happens to be. Every lane of the simdgroup
// therefore lands on a different bank on every lookup, for any index pattern:
// conflict-free by construction rather than by luck.
//
// The cost is 256 * 32 * 4 = 32768 bytes, exactly the M4's per-threadgroup
// maximum, and a staging cost of 32 KB of threadgroup writes per threadgroup.
// That staging is only affordable if threadgroups are persistent, so this kernel
// launches just enough threadgroups to fill the GPU and grid-strides over rows,
// paying the 32 KB once per threadgroup instead of once per 8 rows.
// ---------------------------------------------------------------------------
kernel void gemv_vq_replicated(const device uint4*  IDX [[buffer(0)]],
                               const device half2*  C   [[buffer(1)]],
                               const device half*   S   [[buffer(2)]],
                               const device half2*  x   [[buffer(3)]],
                               device float*        y   [[buffer(4)]],
                               constant VQParams&   P   [[buffer(5)]],
                               threadgroup half2*   Ct  [[threadgroup(0)]],
                               uint  tg_id [[threadgroup_position_in_grid]],
                               uint  n_tg  [[threadgroups_per_grid]],
                               uint  tid   [[thread_index_in_threadgroup]],
                               uint  n_thr [[threads_per_threadgroup]],
                               uint  sg_in [[simdgroup_index_in_threadgroup]],
                               uint  n_sg  [[simdgroups_per_threadgroup]],
                               uint  lane  [[thread_index_in_simdgroup]]) {
    uint total = P.cb_size * REP;
    for (uint i = tid; i < total; i += n_thr) Ct[i] = C[i / REP];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device half2* xb = x;
    for (uint m = tg_id * n_sg + sg_in; m < P.M; m += n_tg * n_sg) {
        const device uint4* irow = IDX + (ulong)m * P.chunks_pr;
        const device half*  srow = S   + (ulong)m * P.chunks_pr;
        float acc = 0.0f;
        for (uint j = lane; j < P.chunks_pr; j += 32) {
            uint4 p = irow[j];
            const device half2* xp = xb + j * 16;
            float d = vqdot4_rep(p.x, Ct, lane, xp)
                    + vqdot4_rep(p.y, Ct, lane, xp + 4)
                    + vqdot4_rep(p.z, Ct, lane, xp + 8)
                    + vqdot4_rep(p.w, Ct, lane, xp + 12);
            acc += float(srow[j]) * d;
        }
        acc = simd_sum(acc);
        if (lane == 0) y[m] = acc;
    }
}

// ---------------------------------------------------------------------------
// v3: codebook left in device memory. 1 KB read by every lane on every lookup,
// so it lives in L1 -- the question this variant answers is whether staging it
// into threadgroup memory is worth doing at all.
// ---------------------------------------------------------------------------
kernel void gemv_vq_device(const device uint4*  IDX [[buffer(0)]],
                           const device half2*  C   [[buffer(1)]],
                           const device half*   S   [[buffer(2)]],
                           const device half2*  x   [[buffer(3)]],
                           device float*        y   [[buffer(4)]],
                           constant VQParams&   P   [[buffer(5)]],
                           uint  tg_id [[threadgroup_position_in_grid]],
                           uint  sg_in [[simdgroup_index_in_threadgroup]],
                           uint  n_sg  [[simdgroups_per_threadgroup]],
                           uint  lane  [[thread_index_in_simdgroup]]) {
    uint m = tg_id * n_sg + sg_in;
    if (m >= P.M) return;
    const device uint4* irow = IDX + (ulong)m * P.chunks_pr;
    const device half*  srow = S   + (ulong)m * P.chunks_pr;
    float acc = 0.0f;
    for (uint j = lane; j < P.chunks_pr; j += 32) {
        uint4 p = irow[j];
        const device half2* xp = x + j * 16;
        float d = 0.0f;
        uint ws[4] = { p.x, p.y, p.z, p.w };
        for (uint t = 0; t < 4; ++t)
            for (uint b = 0; b < 4; ++b)
                d += float(dot(C[(ws[t] >> (8 * b)) & 0xFF], xp[t * 4 + b]));
        acc += float(srow[j]) * d;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}

// ---------------------------------------------------------------------------
// Empirical bandwidth ceiling, identical to the one in ../int4-gemv.
// ---------------------------------------------------------------------------
constant uint UNROLL [[function_constant(1)]];

kernel void stream_read(const device uint4* src [[buffer(0)]],
                        device uint*        sink[[buffer(1)]],
                        constant uint&      nquads [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint gsz [[threads_per_grid]]) {
    uint4 a[8];
    for (uint u = 0; u < UNROLL; ++u) a[u] = uint4(0);
    uint stride = gsz * UNROLL;
    for (uint base = gid; base + stride <= nquads + gid; base += stride) {
        uint4 v[8];
        for (uint u = 0; u < UNROLL; ++u) {
            uint i = base + u * gsz;
            v[u] = (i < nquads) ? src[i] : uint4(0);
        }
        for (uint u = 0; u < UNROLL; ++u) a[u] ^= v[u];
    }
    uint4 acc = uint4(0);
    for (uint u = 0; u < UNROLL; ++u) acc ^= a[u];
    uint r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0xFFFFFFFFu) sink[0] = r;
}

// ---------------------------------------------------------------------------
// The scalar-int4 opponent, at exactly the same bytes/param.
//
// This is the winning kernel from ../int4-gemv (v4: one simdgroup per row,
// uint4 loads, scale and bias interleaved as half2). It is compiled into this
// binary so that both formats are measured in one process, under the same clock
// state, the same cache-defeat rotation and the same dispatch batching. Timing
// them in separate runs would compare quantisation formats using numbers that
// differ mostly by GPU thermal drift.
// ---------------------------------------------------------------------------
struct Int4Params { uint M, K, G, words_pr, groups_pr, chunks_pg; };

inline float qdot8(uint p, half4 xa, half4 xb) {
    half4 v0 = half4( half(p        & 0xF), half((p >>  4) & 0xF),
                      half((p >>  8) & 0xF), half((p >> 12) & 0xF) );
    half4 v1 = half4( half((p >> 16) & 0xF), half((p >> 20) & 0xF),
                      half((p >> 24) & 0xF), half((p >> 28) & 0xF) );
    return float(dot(v0, xa)) + float(dot(v1, xb));
}

kernel void xsum32(const device half4*  x  [[buffer(0)]],
                   device float*        xs [[buffer(1)]],
                   constant Int4Params& P  [[buffer(2)]],
                   uint j [[thread_position_in_grid]]) {
    if (j >= P.K / 32) return;
    float s = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        half4 v = x[j * 8 + i];
        s += float(v.x) + float(v.y) + float(v.z) + float(v.w);
    }
    xs[j] = s;
}

kernel void gemv_int4_fused(const device uint4*  W  [[buffer(0)]],
                            const device half2*  SB [[buffer(1)]],
                            const device half4*  x  [[buffer(3)]],
                            device float*        y  [[buffer(4)]],
                            const device float*  xs [[buffer(5)]],
                            constant Int4Params& P  [[buffer(6)]],
                            uint  tg_id [[threadgroup_position_in_grid]],
                            uint  sg_in [[simdgroup_index_in_threadgroup]],
                            uint  n_sg  [[simdgroups_per_threadgroup]],
                            uint  lane  [[thread_index_in_simdgroup]]) {
    const uint chunks_pr = P.words_pr / 4;
    uint m = tg_id * n_sg + sg_in;
    if (m >= P.M) return;
    float acc = 0.0f;
    for (uint q = lane; q < chunks_pr; q += 32) {
        uint  k4 = q * 8;
        uint4 p  = W[(ulong)m * chunks_pr + q];
        float d  = qdot8(p.x, x[k4+0], x[k4+1]) + qdot8(p.y, x[k4+2], x[k4+3])
                 + qdot8(p.z, x[k4+4], x[k4+5]) + qdot8(p.w, x[k4+6], x[k4+7]);
        half2 sb = SB[(ulong)m * P.groups_pr + q / P.chunks_pg];
        acc += float(sb.x) * d + float(sb.y) * xs[q];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}

// ---------------------------------------------------------------------------
// v5: wide x loads + unrolled index loads.
//
// v1 reads x as sixteen `half2` loads per uint4 of indices -- 4-byte
// transactions, one per codeword. Reading it as four `uint4` and bit-casting
// gives the same bytes in a quarter of the instructions. The index loads are
// then unrolled U-deep so U independent 16-byte fetches are in flight before
// any is consumed. Same two levers that took ../int4-gemv from 89% to 94% of
// peak; the codebook lookups are unchanged.
// ---------------------------------------------------------------------------
constant uint UNR [[function_constant(2)]];

inline float vqdot4_wide(uint w, threadgroup const half2* C, uint4 xq) {
    half2 c0 = C[ w        & 0xFF];
    half2 c1 = C[(w >>  8) & 0xFF];
    half2 c2 = C[(w >> 16) & 0xFF];
    half2 c3 = C[(w >> 24) & 0xFF];
    return float(dot(c0, as_type<half2>(xq.x))) + float(dot(c1, as_type<half2>(xq.y)))
         + float(dot(c2, as_type<half2>(xq.z))) + float(dot(c3, as_type<half2>(xq.w)));
}

kernel void gemv_vq_wide(const device uint4*  IDX [[buffer(0)]],
                         const device half2*  C   [[buffer(1)]],
                         const device half*   S   [[buffer(2)]],
                         const device uint4*  x   [[buffer(3)]],
                         device float*        y   [[buffer(4)]],
                         constant VQParams&   P   [[buffer(5)]],
                         threadgroup half2*   Ct  [[threadgroup(0)]],
                         uint  tg_id [[threadgroup_position_in_grid]],
                         uint  tid   [[thread_index_in_threadgroup]],
                         uint  n_thr [[threads_per_threadgroup]],
                         uint  sg_in [[simdgroup_index_in_threadgroup]],
                         uint  n_sg  [[simdgroups_per_threadgroup]],
                         uint  lane  [[thread_index_in_simdgroup]]) {
    for (uint i = tid; i < P.cb_size; i += n_thr) Ct[i] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m = tg_id * n_sg + sg_in;
    if (m >= P.M) return;

    const device uint4* irow = IDX + (ulong)m * P.chunks_pr;
    const device half*  srow = S   + (ulong)m * P.chunks_pr;

    float acc = 0.0f;
    uint stride = 32 * UNR;
    for (uint base = lane; base < P.chunks_pr; base += stride) {
        uint4 p[8];
        for (uint u = 0; u < UNR; ++u) {
            uint j = base + u * 32;
            p[u] = (j < P.chunks_pr) ? irow[j] : uint4(0);
        }
        for (uint u = 0; u < UNR; ++u) {
            uint j = base + u * 32;
            if (j >= P.chunks_pr) break;
            const device uint4* xp = x + j * 4;      // 4 uint4 = 32 halves
            float d = vqdot4_wide(p[u].x, Ct, xp[0])
                    + vqdot4_wide(p[u].y, Ct, xp[1])
                    + vqdot4_wide(p[u].z, Ct, xp[2])
                    + vqdot4_wide(p[u].w, Ct, xp[3]);
            acc += float(srow[j]) * d;
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}

// The int4 opponent, upgraded to match: wide x loads, xsum in threadgroup
// memory, unrolled weight loads. Without this the equal-bytes comparison would
// pit a tuned VQ kernel against an untuned scalar one.
kernel void gemv_int4_wide_unroll(const device uint4*  W  [[buffer(0)]],
                                  const device half2*  SB [[buffer(1)]],
                                  const device uint4*  x  [[buffer(3)]],
                                  device float*        y  [[buffer(4)]],
                                  const device float*  xs [[buffer(5)]],
                                  constant Int4Params& P  [[buffer(6)]],
                                  threadgroup float*   xst [[threadgroup(0)]],
                                  uint  tg_id [[threadgroup_position_in_grid]],
                                  uint  tid   [[thread_index_in_threadgroup]],
                                  uint  n_thr [[threads_per_threadgroup]],
                                  uint  sg_in [[simdgroup_index_in_threadgroup]],
                                  uint  n_sg  [[simdgroups_per_threadgroup]],
                                  uint  lane  [[thread_index_in_simdgroup]]) {
    const uint chunks_pr = P.words_pr / 4;
    for (uint i = tid; i < chunks_pr; i += n_thr) xst[i] = xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m = tg_id * n_sg + sg_in;
    if (m >= P.M) return;
    const device uint4* wrow = W + (ulong)m * chunks_pr;
    const device half2* sbrow = SB + (ulong)m * P.groups_pr;

    float acc = 0.0f;
    uint stride = 32 * UNR;
    for (uint base = lane; base < chunks_pr; base += stride) {
        uint4 p[8];
        for (uint u = 0; u < UNR; ++u) {
            uint q = base + u * 32;
            p[u] = (q < chunks_pr) ? wrow[q] : uint4(0);
        }
        for (uint u = 0; u < UNR; ++u) {
            uint q = base + u * 32;
            if (q >= chunks_pr) break;
            uint4 xa = x[q*4+0], xb = x[q*4+1], xc = x[q*4+2], xd = x[q*4+3];
            float d = qdot8(p[u].x, as_type<half4>(uint2(xa.x, xa.y)),
                                    as_type<half4>(uint2(xa.z, xa.w)))
                    + qdot8(p[u].y, as_type<half4>(uint2(xb.x, xb.y)),
                                    as_type<half4>(uint2(xb.z, xb.w)))
                    + qdot8(p[u].z, as_type<half4>(uint2(xc.x, xc.y)),
                                    as_type<half4>(uint2(xc.z, xc.w)))
                    + qdot8(p[u].w, as_type<half4>(uint2(xd.x, xd.y)),
                                    as_type<half4>(uint2(xd.z, xd.w)));
            half2 sb = sbrow[q / P.chunks_pg];
            acc += float(sb.x) * d + float(sb.y) * xst[q];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}
