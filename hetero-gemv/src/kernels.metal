// int4 grouped-affine GEMV for LLM decode, Apple M4.
//
// Layout (bit-compatible with mlx.quantize(..., bits=4)):
//   W is [M, K] row-major, 4-bit unsigned, 8 values packed per uint32,
//   value j of a word sits at nibble j (low nibble = lowest k).
//   Per group of G consecutive k, one fp16 scale and one fp16 bias:
//       w[m,k] = scale[m, k/G] * q[m,k] + bias[m, k/G]
//   x is [K] fp16, y is [M] fp32.
//
// The whole point of this file is that decode GEMV is a pure streaming read of
// W. Every kernel below reads each byte of W exactly once; they differ only in
// how well they keep the memory system busy while doing it.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Dequant-and-accumulate primitive.
//
// The affine form factors, which is what makes int4 GEMV cheap in ALU terms:
//     sum_k (s*q_k + b) * x_k  =  s * sum_k q_k*x_k  +  b * sum_k x_k
// The second term needs only a partial sum of x, which is identical for every
// row, so it is precomputed once per token by `xsum32` below and costs nothing
// in the inner loop.
//
// The partial sums are kept per 32-element chunk, NOT per quant group. Every
// kernel here consumes W in 32-weight chunks (one uint4 per lane), so a
// per-group sum would have to be added exactly once per group while the loop
// runs G/32 times over it -- an easy way to double-count the bias. Per-chunk
// sums make the bias term land exactly once per chunk with no branch.
// ---------------------------------------------------------------------------

// dot product of the 8 nibbles of `p` with 8 halves of x, as float.
inline float qdot8(uint p, half4 xa, half4 xb) {
    half4 v0 = half4( half(p        & 0xF), half((p >>  4) & 0xF),
                      half((p >>  8) & 0xF), half((p >> 12) & 0xF) );
    half4 v1 = half4( half((p >> 16) & 0xF), half((p >> 20) & 0xF),
                      half((p >> 24) & 0xF), half((p >> 28) & 0xF) );
    return float(dot(v0, xa)) + float(dot(v1, xb));
}

struct GemvParams {
    uint M;         // rows / output length
    uint K;         // reduction length
    uint G;         // quant group size
    uint words_pr;  // uint32 words per row  = K/8
    uint groups_pr; // groups per row        = K/G
    uint chunks_pg; // 32-weight chunks per group = G/32
};

// ---------------------------------------------------------------------------
// Precompute sum(x) over each 32-element chunk. K/32 floats, once per token.
// ---------------------------------------------------------------------------
kernel void xsum32(const device half4*  x    [[buffer(0)]],
                   device float*        xs   [[buffer(1)]],
                   constant GemvParams& P    [[buffer(2)]],
                   uint j [[thread_position_in_grid]]) {
    if (j >= P.K / 32) return;
    float s = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        half4 v = x[j * 8 + i];
        s += float(v.x) + float(v.y) + float(v.z) + float(v.w);
    }
    xs[j] = s;
}

// ---------------------------------------------------------------------------
// v0: scalar baseline. One thread per output row, one uint32 (8 weights) per
// step. Correct, and the reference every other kernel is checked against.
// ---------------------------------------------------------------------------
kernel void gemv_int4_scalar(const device uint4*  W  [[buffer(0)]],
                             const device half*   S  [[buffer(1)]],
                             const device half*   B  [[buffer(2)]],
                             const device half4*  x  [[buffer(3)]],
                             device float*        y  [[buffer(4)]],
                             const device float*  xs [[buffer(5)]],
                             constant GemvParams& P  [[buffer(6)]],
                             uint m [[thread_position_in_grid]]) {
    if (m >= P.M) return;
    const uint chunks_pr = P.words_pr / 4;
    const device uint4* wrow = W + (ulong)m * chunks_pr;
    const device half*  srow = S + (ulong)m * P.groups_pr;
    const device half*  brow = B + (ulong)m * P.groups_pr;

    float acc = 0.0f;
    for (uint j = 0; j < chunks_pr; ++j) {
        uint4 p  = wrow[j];
        uint  k4 = j * 8;
        float d  = qdot8(p.x, x[k4+0], x[k4+1]) + qdot8(p.y, x[k4+2], x[k4+3])
                 + qdot8(p.z, x[k4+4], x[k4+5]) + qdot8(p.w, x[k4+6], x[k4+7]);
        uint g = j / P.chunks_pg;
        acc += float(srow[g]) * d + float(brow[g]) * xs[j];
    }
    y[m] = acc;
}

// ---------------------------------------------------------------------------
// v1: one simdgroup per row, 128-bit (uint4) loads.
//
// 32 lanes x uint4 = 32 x 32 weights = 1024 weights consumed per simdgroup
// step, issued as 32 fully-coalesced 16-byte loads covering one contiguous
// 512-byte span of the row. Reduction is a single simd_sum.
// ---------------------------------------------------------------------------
kernel void gemv_int4_simd(const device uint4*  W  [[buffer(0)]],
                           const device half*   S  [[buffer(1)]],
                           const device half*   B  [[buffer(2)]],
                           const device half4*  x  [[buffer(3)]],
                           device float*        y  [[buffer(4)]],
                           const device float*  xs [[buffer(5)]],
                           constant GemvParams& P  [[buffer(6)]],
                           uint  sg_id  [[threadgroup_position_in_grid]],
                           uint  sg_in  [[simdgroup_index_in_threadgroup]],
                           uint  n_sg   [[simdgroups_per_threadgroup]],
                           uint  lane   [[thread_index_in_simdgroup]]) {
    uint m = sg_id * n_sg + sg_in;
    if (m >= P.M) return;

    const uint quads_pr = P.words_pr / 4;         // uint4 per row
    const device uint4* wrow = W + (ulong)m * quads_pr;
    const device half*  srow = S + (ulong)m * P.groups_pr;
    const device half*  brow = B + (ulong)m * P.groups_pr;

    float acc = 0.0f;
    for (uint q = lane; q < quads_pr; q += 32) {
        uint4 p  = wrow[q];
        uint  k4 = q * 8;                          // index into half4-view of x
        float d  = qdot8(p.x, x[k4+0], x[k4+1])
                 + qdot8(p.y, x[k4+2], x[k4+3])
                 + qdot8(p.z, x[k4+4], x[k4+5])
                 + qdot8(p.w, x[k4+6], x[k4+7]);
        uint g = q / P.chunks_pg;
        acc += float(srow[g]) * d + float(brow[g]) * xs[q];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}

// ---------------------------------------------------------------------------
// v2: one simdgroup per R rows.
//
// Same access pattern as v1, but the x values and the group index computed for
// a lane are reused across R rows, so per-byte-of-W overhead drops by R and R
// independent row streams are in flight at once. More outstanding loads per
// thread is the main lever for hiding DRAM latency on Apple GPUs.
// R is templated by specialisation constant so the loops fully unroll.
// ---------------------------------------------------------------------------
constant uint RPS [[function_constant(0)]];

kernel void gemv_int4_multirow(const device uint4*  W  [[buffer(0)]],
                               const device half*   S  [[buffer(1)]],
                               const device half*   B  [[buffer(2)]],
                               const device half4*  x  [[buffer(3)]],
                               device float*        y  [[buffer(4)]],
                               const device float*  xs [[buffer(5)]],
                               constant GemvParams& P  [[buffer(6)]],
                               uint  tg_id  [[threadgroup_position_in_grid]],
                               uint  sg_in  [[simdgroup_index_in_threadgroup]],
                               uint  n_sg   [[simdgroups_per_threadgroup]],
                               uint  lane   [[thread_index_in_simdgroup]]) {
    const uint quads_pr = P.words_pr / 4;
    uint m0 = (tg_id * n_sg + sg_in) * RPS;

    float acc[8];
    for (uint r = 0; r < RPS; ++r) acc[r] = 0.0f;

    for (uint q = lane; q < quads_pr; q += 32) {
        uint  k4 = q * 8;
        half4 x0 = x[k4+0], x1 = x[k4+1], x2 = x[k4+2], x3 = x[k4+3];
        half4 x4 = x[k4+4], x5 = x[k4+5], x6 = x[k4+6], x7 = x[k4+7];
        uint  g  = q / P.chunks_pg;
        float sx = xs[q];

        // issue all R loads before consuming any of them
        uint4 p[8];
        for (uint r = 0; r < RPS; ++r)
            p[r] = W[(ulong)min(m0 + r, P.M - 1) * quads_pr + q];

        for (uint r = 0; r < RPS; ++r) {
            float d = qdot8(p[r].x, x0, x1) + qdot8(p[r].y, x2, x3)
                    + qdot8(p[r].z, x4, x5) + qdot8(p[r].w, x6, x7);
            uint  mg = min(m0 + r, P.M - 1) * P.groups_pr + g;
            acc[r] += float(S[mg]) * d + float(B[mg]) * sx;
        }
    }
    // rows past the end are computed against the clamped last row and dropped;
    // that costs one partial threadgroup at the tail and lets any M run
    for (uint r = 0; r < RPS; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && m0 + r < P.M) y[m0 + r] = v;
    }
}

// ---------------------------------------------------------------------------
// v3: v2 with x staged in threadgroup memory.
//
// x is read by every simdgroup in the threadgroup; staging it once into the
// 32 KB threadgroup memory removes those redundant device-side reads and the
// pressure they put on the same cache lines W is streaming through.
// Requires K*2 bytes <= 32 KB, i.e. K <= 16384.
// ---------------------------------------------------------------------------
kernel void gemv_int4_multirow_tgx(const device uint4*  W  [[buffer(0)]],
                                   const device half*   S  [[buffer(1)]],
                                   const device half*   B  [[buffer(2)]],
                                   const device half4*  x  [[buffer(3)]],
                                   device float*        y  [[buffer(4)]],
                                   const device float*  xs [[buffer(5)]],
                                   constant GemvParams& P  [[buffer(6)]],
                                   threadgroup half4*   xt [[threadgroup(0)]],
                                   uint  tg_id  [[threadgroup_position_in_grid]],
                                   uint  tid    [[thread_index_in_threadgroup]],
                                   uint  n_thr  [[threads_per_threadgroup]],
                                   uint  sg_in  [[simdgroup_index_in_threadgroup]],
                                   uint  n_sg   [[simdgroups_per_threadgroup]],
                                   uint  lane   [[thread_index_in_simdgroup]]) {
    const uint quads_pr = P.words_pr / 4;
    const uint x_half4 = P.K / 4;

    for (uint i = tid; i < x_half4; i += n_thr) xt[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m0 = (tg_id * n_sg + sg_in) * RPS;

    float acc[8];
    for (uint r = 0; r < RPS; ++r) acc[r] = 0.0f;

    for (uint q = lane; q < quads_pr; q += 32) {
        uint  k4 = q * 8;
        half4 x0 = xt[k4+0], x1 = xt[k4+1], x2 = xt[k4+2], x3 = xt[k4+3];
        half4 x4 = xt[k4+4], x5 = xt[k4+5], x6 = xt[k4+6], x7 = xt[k4+7];
        uint  g  = q / P.chunks_pg;
        float sx = xs[q];

        uint4 p[8];
        for (uint r = 0; r < RPS; ++r)
            p[r] = W[(ulong)min(m0 + r, P.M - 1) * quads_pr + q];

        for (uint r = 0; r < RPS; ++r) {
            float d = qdot8(p[r].x, x0, x1) + qdot8(p[r].y, x2, x3)
                    + qdot8(p[r].z, x4, x5) + qdot8(p[r].w, x6, x7);
            uint  mg = min(m0 + r, P.M - 1) * P.groups_pr + g;
            acc[r] += float(S[mg]) * d + float(B[mg]) * sx;
        }
    }
    // rows past the end are computed against the clamped last row and dropped;
    // that costs one partial threadgroup at the tail and lets any M run
    for (uint r = 0; r < RPS; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && m0 + r < P.M) y[m0 + r] = v;
    }
}

// ---------------------------------------------------------------------------
// Empirical bandwidth ceiling: pure streaming read, no arithmetic worth the
// name, same 16-byte-per-lane access pattern as the GEMV kernels. This is the
// honest denominator -- what this machine will actually hand a compute kernel.
// ---------------------------------------------------------------------------
// UNROLL independent loads are issued before any is consumed: a single
// accumulator would make each load wait on the previous XOR and measure
// latency rather than bandwidth.
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
    if (r == 0xFFFFFFFFu) sink[0] = r;   // never taken; defeats DCE
}

// ---------------------------------------------------------------------------
// v4: fused metadata stream.
//
// v1-v3 read the scale and the bias for a group from two separate arrays, so
// every 32 weights costs two 2-byte loads from two distant addresses. Storing
// them interleaved as a half2 makes it one 4-byte load from one address: the
// same bytes off DRAM, but half the memory instructions and one metadata
// stream per row instead of two. At ~90% of peak the remaining cost is
// transactions, not bytes, so this is the lever that is left.
// ---------------------------------------------------------------------------
kernel void gemv_int4_fused(const device uint4*  W  [[buffer(0)]],
                            const device half2*  SB [[buffer(1)]],
                            const device half4*  x  [[buffer(3)]],
                            device float*        y  [[buffer(4)]],
                            const device float*  xs [[buffer(5)]],
                            constant GemvParams& P  [[buffer(6)]],
                            uint  tg_id  [[threadgroup_position_in_grid]],
                            uint  sg_in  [[simdgroup_index_in_threadgroup]],
                            uint  n_sg   [[simdgroups_per_threadgroup]],
                            uint  lane   [[thread_index_in_simdgroup]]) {
    const uint chunks_pr = P.words_pr / 4;
    uint m0 = (tg_id * n_sg + sg_in) * RPS;

    float acc[8];
    for (uint r = 0; r < RPS; ++r) acc[r] = 0.0f;

    for (uint q = lane; q < chunks_pr; q += 32) {
        uint  k4 = q * 8;
        half4 x0 = x[k4+0], x1 = x[k4+1], x2 = x[k4+2], x3 = x[k4+3];
        half4 x4 = x[k4+4], x5 = x[k4+5], x6 = x[k4+6], x7 = x[k4+7];
        uint  g  = q / P.chunks_pg;
        float sx = xs[q];

        uint4 p[8];
        for (uint r = 0; r < RPS; ++r)
            p[r] = W[(ulong)min(m0 + r, P.M - 1) * chunks_pr + q];

        for (uint r = 0; r < RPS; ++r) {
            float d = qdot8(p[r].x, x0, x1) + qdot8(p[r].y, x2, x3)
                    + qdot8(p[r].z, x4, x5) + qdot8(p[r].w, x6, x7);
            half2 sb = SB[(ulong)min(m0 + r, P.M - 1) * P.groups_pr + g];
            acc[r] += float(sb.x) * d + float(sb.y) * sx;
        }
    }
    // rows past the end are computed against the clamped last row and dropped;
    // that costs one partial threadgroup at the tail and lets any M run
    for (uint r = 0; r < RPS; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0 && m0 + r < P.M) y[m0 + r] = v;
    }
}

// ---------------------------------------------------------------------------
// v5: wider x loads + xsum staged in threadgroup memory.
//
// Two instruction-side costs remain in v1/v4 that the profile-by-subtraction
// above does not remove:
//
//   * x is read as eight `half4` loads per 32 weights -- 64 bytes of x per 16
//     bytes of W, in 8-byte transactions. Reading it as four `uint4` and
//     bit-casting to `half4` pairs halves the load count for the same bytes.
//     x is small and cache-resident, so this is purely about issue slots.
//   * xs[] is a K/32-element array re-read by every row. It is 2 KB and lives
//     in L1, but it still costs a device load per iteration; staging it once
//     per threadgroup makes it a threadgroup-memory read instead.
//
// Both are pure instruction-count reductions on a kernel that is already at
// ~95% of the achievable read bandwidth, which is the only place left to look.
// ---------------------------------------------------------------------------
kernel void gemv_int4_wide(const device uint4*  W  [[buffer(0)]],
                           const device half2*  SB [[buffer(1)]],
                           const device uint4*  x  [[buffer(3)]],
                           device float*        y  [[buffer(4)]],
                           const device float*  xs [[buffer(5)]],
                           constant GemvParams& P  [[buffer(6)]],
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

    float acc = 0.0f;
    for (uint q = lane; q < chunks_pr; q += 32) {
        uint4 p  = W[(ulong)m * chunks_pr + q];
        uint  k4 = q * 4;                       // uint4 index into x
        uint4 xa = x[k4+0], xb = x[k4+1], xc = x[k4+2], xd = x[k4+3];
        float d  = qdot8(p.x, as_type<half4>(uint2(xa.x, xa.y)),
                              as_type<half4>(uint2(xa.z, xa.w)))
                 + qdot8(p.y, as_type<half4>(uint2(xb.x, xb.y)),
                              as_type<half4>(uint2(xb.z, xb.w)))
                 + qdot8(p.z, as_type<half4>(uint2(xc.x, xc.y)),
                              as_type<half4>(uint2(xc.z, xc.w)))
                 + qdot8(p.w, as_type<half4>(uint2(xd.x, xd.y)),
                              as_type<half4>(uint2(xd.z, xd.w)));
        half2 sb = SB[(ulong)m * P.groups_pr + q / P.chunks_pg];
        acc += float(sb.x) * d + float(sb.y) * xst[q];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc;
}

// v6: v5 with the weight loads unrolled by U, so U independent 16-byte loads
// are issued before any is consumed. Same lever that took the pure-read ceiling
// kernel from latency-bound to bandwidth-bound.
constant uint UNR [[function_constant(2)]];

kernel void gemv_int4_wide_unroll(const device uint4*  W  [[buffer(0)]],
                                  const device half2*  SB [[buffer(1)]],
                                  const device uint4*  x  [[buffer(3)]],
                                  device float*        y  [[buffer(4)]],
                                  const device float*  xs [[buffer(5)]],
                                  constant GemvParams& P  [[buffer(6)]],
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
            uint k4 = q * 4;
            uint4 xa = x[k4+0], xb = x[k4+1], xc = x[k4+2], xd = x[k4+3];
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
