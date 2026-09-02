// Fully-fused MoE decode block: router + top-k + expert gather + int4 matmul.
//
// The unfused shape of this block is ~30 kernel launches. Fused it is 3, and 3
// is the floor rather than a stopping point: the block has exactly two
// grid-wide data dependencies, and Metal gives no way to synchronise across
// threadgroups inside one dispatch.
//
//   1. router + softmax + top-k        ->  idx[4], wgt[4]
//   2. gather + gate + up + SwiGLU     ->  h[4][inter]     (needs idx from 1)
//   3. gather + down + weighted combine->  y[hidden]       (needs all of h)
//
// The expert gather is genuinely inside the kernels: they read `idx` from
// device memory and compute the weight addresses themselves, so the host binds
// one pool buffer once and never re-binds per expert.
//
// Weights are gpt-oss-120b experts exactly as repacked by moe-stream: int4
// group 64, BF16 scales/biases, one 14,024,704-byte blob per expert.

#include <metal_stdlib>
using namespace metal;

struct MoEParams {
    uint hidden;      // 2880
    uint inter;       // 2880
    uint n_experts;   // 128
    uint top_k;       // 4
    uint quads_hid;   // hidden/32
    uint quads_int;   // inter/32
    uint groups_hid;  // hidden/64
    uint groups_int;  // inter/64
    uint chunks_pg;   // 64/32 = 2
    ulong blob;       // 14024704
    ulong off_gate, off_up, off_down;
    ulong sub_w, sub_s, sub_b, sub_bi;
};

inline float qdot8(uint p, float4 xa, float4 xb) {
    float4 v0 = float4( float( p        & 0xF), float((p >>  4) & 0xF),
                        float((p >>  8) & 0xF), float((p >> 12) & 0xF) );
    float4 v1 = float4( float((p >> 16) & 0xF), float((p >> 20) & 0xF),
                        float((p >> 24) & 0xF), float((p >> 28) & 0xF) );
    return dot(v0, xa) + dot(v1, xb);
}

// Weight loads per row can be unrolled QUNR-deep. Unlike ../int4-gemv, where
// 4x unrolling was worth +4%, it LOSES here and QUNR=1 is best: K is 2880, which
// is 90 uint4 per row and under 3 loop iterations per lane, so the unrolled
// staging array spills more than the extra loads in flight buy. Kept as a
// constant because the negative result is the point.
constant uint QUNR [[function_constant(0)]];

// Activations are fp16. They are read once per 16 bytes of weights and reread by
// every row, so at fp32 they cost 128 bytes of loads per 16 bytes of weight --
// eight float4 fetches. At fp16 the same values are four uint4 fetches
// bit-cast to half4 pairs: half the transactions for identical arithmetic, and x
// is cache-resident so it is issue slots that are being saved, not bandwidth.
inline void unpack8(uint4 v, thread float4& a, thread float4& b) {
    half4 h0 = as_type<half4>(uint2(v.x, v.y));
    half4 h1 = as_type<half4>(uint2(v.z, v.w));
    a = float4(h0); b = float4(h1);
}

// One quantised row dot: y = sum_k w[m,k]*x[k], weights gathered from `blobbase`.
inline float qrow(const device uchar* blobbase, ulong proj, uint m,
                  const device uint4* x, threadgroup const float* xs,
                  uint quads, uint groups, uint chunks_pg, uint lane,
                  constant MoEParams& P) {
    const device uint4*  W = (const device uint4*)(blobbase + proj + P.sub_w)
                             + (ulong)m * quads;
    const device bfloat* S = (const device bfloat*)(blobbase + proj + P.sub_s)
                             + (ulong)m * groups;
    const device bfloat* B = (const device bfloat*)(blobbase + proj + P.sub_b)
                             + (ulong)m * groups;
    float acc = 0.0f;
    uint stride = 32 * QUNR;
    for (uint base = lane; base < quads; base += stride) {
        uint4 p[8];
        for (uint u = 0; u < QUNR; ++u) {
            uint q = base + u * 32;
            p[u] = (q < quads) ? W[q] : uint4(0);
        }
        for (uint u = 0; u < QUNR; ++u) {
            uint q = base + u * 32;
            if (q >= quads) break;
            uint  k4 = q * 4;
            float4 a0,a1,b0,b1,c0,c1,d0,d1;
            unpack8(x[k4+0], a0, a1); unpack8(x[k4+1], b0, b1);
            unpack8(x[k4+2], c0, c1); unpack8(x[k4+3], d0, d1);
            float d = qdot8(p[u].x, a0, a1) + qdot8(p[u].y, b0, b1)
                    + qdot8(p[u].z, c0, c1) + qdot8(p[u].w, d0, d1);
            uint g = q / chunks_pg;
            acc += float(S[g]) * d + float(B[g]) * xs[q];
        }
    }
    return simd_sum(acc);
}

// ---------------------------------------------------------------------------
// 1. router + softmax + top-k, one threadgroup.
//
// The router is [n_experts, hidden] and n_experts is 128, so the whole logit
// vector fits in threadgroup memory and the selection is a 4-pass argmax over
// 128 values -- cheaper than any sort and trivially correct. Softmax is taken
// over the selected k only, matching gpt-oss.
// ---------------------------------------------------------------------------
// One simdgroup per expert with float4 loads and a simd_sum reduction. The
// first version gave each expert a single thread walking `hidden` scalars, which
// left one threadgroup pulling the whole 1.47 MB router at one core's share of
// bandwidth -- about 16% of the whole block. The unfused path uses the identical
// routine, so the comparison below measures fusion and not router quality.
inline float router_logit(const device float4* r4, const device float4* x4,
                          uint hid4, uint lane) {
    float a = 0.0f;
    for (uint k = lane; k < hid4; k += 32) {
        float4 v = r4[k] * x4[k];
        a += v.x + v.y + v.z + v.w;
    }
    return simd_sum(a);
}

kernel void router_topk(const device float*  R    [[buffer(0)]],  // [n_experts, hidden]
                        const device float*  x    [[buffer(1)]],
                        device uint*         idx  [[buffer(2)]],
                        device float*        wgt  [[buffer(3)]],
                        constant MoEParams&  P    [[buffer(4)]],
                        device atomic_float* xsh  [[buffer(5)]],
                        device half*         xh   [[buffer(6)]],
                        threadgroup float*   lg   [[threadgroup(0)]],
                        uint tid   [[thread_index_in_threadgroup]],
                        uint nthr  [[threads_per_threadgroup]]) {
    // Clear the sum(h) accumulator the atomic variant of pass 2 adds into. The
    // clear must itself be atomic: a plain store here and an atomic RMW there is
    // a mixed-access hazard, and it showed up as ~1/7 of the expected sum -- the
    // plain zeroes were landing after some of the next dispatch's adds.
    for (uint j = tid; j < P.top_k * P.quads_int; j += nthr)
        atomic_store_explicit(&xsh[j], 0.0f, memory_order_relaxed);
    for (uint j = tid; j < P.hidden; j += nthr) xh[j] = half(x[j]);

    uint sg_in = tid / 32, n_sg = nthr / 32, lane = tid % 32;
    const device float4* x4 = (const device float4*)x;
    uint hid4 = P.hidden / 4;
    for (uint e = sg_in; e < P.n_experts; e += n_sg) {
        float v = router_logit((const device float4*)(R + (ulong)e * P.hidden),
                               x4, hid4, lane);
        if (lane == 0) lg[e] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float chosen[8];
        for (uint j = 0; j < P.top_k; ++j) {
            uint  bi = 0; float bv = -INFINITY;
            for (uint e = 0; e < P.n_experts; ++e)
                if (lg[e] > bv) { bv = lg[e]; bi = e; }
            idx[j] = bi; chosen[j] = bv;
            lg[bi] = -INFINITY;          // remove and take the next
        }
        float mx = chosen[0], sum = 0.0f;
        for (uint j = 1; j < P.top_k; ++j) mx = max(mx, chosen[j]);
        for (uint j = 0; j < P.top_k; ++j) { chosen[j] = exp(chosen[j] - mx); sum += chosen[j]; }
        for (uint j = 0; j < P.top_k; ++j) wgt[j] = chosen[j] / sum;
    }
}

// ---------------------------------------------------------------------------
// 2. gather + gate + up + SwiGLU, for all top_k experts in one dispatch.
//
// grid is (top_k * inter) rows, one simdgroup per row. Both projections for a
// row are computed back to back so x is read once and the SwiGLU is applied
// before anything leaves registers -- gate and up never reach device memory.
//
// sum(x) per 32-chunk is recomputed redundantly by every threadgroup into
// threadgroup memory. That is 90 floats from an x that is already in cache, and
// it removes a dispatch: a separate reduction kernel would be a grid-wide
// dependency of its own.
// ---------------------------------------------------------------------------
kernel void fused_gate_up_swiglu(const device uchar*  pool [[buffer(0)]],
                                 const device uint*   idx  [[buffer(1)]],
                                 const device uint4*  x    [[buffer(2)]],
                                 device half*         h    [[buffer(3)]],
                                 constant MoEParams&  P    [[buffer(4)]],
                                 threadgroup float*   xs   [[threadgroup(0)]],
                                 uint tg_id [[threadgroup_position_in_grid]],
                                 uint tid   [[thread_index_in_threadgroup]],
                                 uint nthr  [[threads_per_threadgroup]],
                                 uint sg_in [[simdgroup_index_in_threadgroup]],
                                 uint n_sg  [[simdgroups_per_threadgroup]],
                                 uint lane  [[thread_index_in_simdgroup]]) {
    for (uint j = tid; j < P.quads_hid; j += nthr) {
        float4 t = float4(0);
        for (uint u = 0; u < 4; ++u) {
            float4 a, b; unpack8(x[j*4+u], a, b); t += a + b;
        }
        xs[j] = t.x + t.y + t.z + t.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg_id * n_sg + sg_in;
    uint e   = row / P.inter;
    uint m   = row % P.inter;
    if (e >= P.top_k) return;

    const device uchar* bb = pool + (ulong)idx[e] * P.blob;   // <- the gather
    float g = qrow(bb, P.off_gate, m, x, xs, P.quads_hid, P.groups_hid,
                   P.chunks_pg, lane, P);
    float u = qrow(bb, P.off_up,   m, x, xs, P.quads_hid, P.groups_hid,
                   P.chunks_pg, lane, P);
    if (lane == 0) {
        const device bfloat* bg = (const device bfloat*)(bb + P.off_gate + P.sub_bi);
        const device bfloat* bu = (const device bfloat*)(bb + P.off_up   + P.sub_bi);
        g += float(bg[m]); u += float(bu[m]);
        float gg = min(g, 7.0f), ll = clamp(u, -7.0f, 7.0f);
        h[(ulong)e * P.inter + m] = half((gg / (1.0f + exp(-1.702f * gg))) * (ll + 1.0f));
    }
}

// ---------------------------------------------------------------------------
// 3. gather + down + weighted combine.
//
// One simdgroup per output row, looping over the top_k experts and accumulating
// wgt[e] * down_e(h_e) directly, so the per-expert partial outputs never exist
// in memory and the combine costs nothing.
// ---------------------------------------------------------------------------
kernel void fused_down_combine(const device uchar*  pool [[buffer(0)]],
                               const device uint*   idx  [[buffer(1)]],
                               const device float*  wgt  [[buffer(2)]],
                               const device uint4*  h    [[buffer(3)]],
                               device float*        y    [[buffer(4)]],
                               constant MoEParams&  P    [[buffer(5)]],
                               threadgroup float*   xs   [[threadgroup(0)]],
                               uint tg_id [[threadgroup_position_in_grid]],
                               uint tid   [[thread_index_in_threadgroup]],
                               uint nthr  [[threads_per_threadgroup]],
                               uint sg_in [[simdgroup_index_in_threadgroup]],
                               uint n_sg  [[simdgroups_per_threadgroup]],
                               uint lane  [[thread_index_in_simdgroup]]) {
    // sum(h_e) per 32-chunk for every selected expert: top_k * inter/32 floats
    uint tot = P.top_k * P.quads_int;
    for (uint j = tid; j < tot; j += nthr) {
        const device uint4* he = h + (ulong)(j / P.quads_int) * (P.inter / 8);
        uint c = j % P.quads_int;
        float4 t = float4(0);
        for (uint u = 0; u < 4; ++u) {
            float4 a, b; unpack8(he[c*4+u], a, b); t += a + b;
        }
        xs[j] = t.x + t.y + t.z + t.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m = tg_id * n_sg + sg_in;
    if (m >= P.hidden) return;

    float acc = 0.0f;
    for (uint e = 0; e < P.top_k; ++e) {
        const device uchar* bb = pool + (ulong)idx[e] * P.blob;      // <- the gather
        const device uint4* he = h + (ulong)e * (P.inter / 8);
        float d = qrow(bb, P.off_down, m, he, xs + e * P.quads_int,
                       P.quads_int, P.groups_int, P.chunks_pg, lane, P);
        if (lane == 0) {
            const device bfloat* bd = (const device bfloat*)(bb + P.off_down + P.sub_bi);
            acc += wgt[e] * (d + float(bd[m]));
        }
    }
    if (lane == 0) y[m] = acc;
}

// ===========================================================================
// Unfused reference path.
//
// The same block decomposed the way an eager framework runs it: one kernel per
// operation, every intermediate through device memory. This exists to make the
// launch-count comparison a measurement rather than an assertion -- both paths
// are dispatched by the same harness, which counts its own dispatches, and both
// are checked against the same reference output.
// ===========================================================================

struct GemvJob {
    uint slot_i;   // which of the top_k slots -> idx[slot_i]
    uint M, K, quads, groups;
    ulong proj;
    uint has_bias;
};

kernel void u_router(const device float* R [[buffer(0)]],
                     const device float* x [[buffer(1)]],
                     device float*       lg[[buffer(2)]],
                     constant MoEParams& P [[buffer(3)]],
                     device half*        xh[[buffer(4)]],
                     device atomic_float* xsh[[buffer(5)]],
                     uint gid   [[thread_position_in_grid]],
                     uint gsz   [[threads_per_grid]],
                     uint tg_id [[threadgroup_position_in_grid]],
                     uint sg_in [[simdgroup_index_in_threadgroup]],
                     uint n_sg  [[simdgroups_per_threadgroup]],
                     uint lane  [[thread_index_in_simdgroup]]) {
    // same fp16 activation copy and accumulator clear the fused router does, so
    // the unfused path stands alone rather than depending on the fused path
    // having run first
    for (uint j = gid; j < P.hidden; j += gsz) xh[j] = half(x[j]);
    for (uint j = gid; j < P.top_k * P.quads_int; j += gsz)
        atomic_store_explicit(&xsh[j], 0.0f, memory_order_relaxed);
    uint e = tg_id * n_sg + sg_in;
    if (e >= P.n_experts) return;
    float v = router_logit((const device float4*)(R + (ulong)e * P.hidden),
                           (const device float4*)x, P.hidden / 4, lane);
    if (lane == 0) lg[e] = v;
}

kernel void u_topk(device float*       lg  [[buffer(0)]],
                   device uint*        idx [[buffer(1)]],
                   device float*       sel [[buffer(2)]],
                   constant MoEParams& P   [[buffer(3)]],
                   uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    for (uint j = 0; j < P.top_k; ++j) {
        uint bi = 0; float bv = -INFINITY;
        for (uint e = 0; e < P.n_experts; ++e)
            if (lg[e] > bv) { bv = lg[e]; bi = e; }
        idx[j] = bi; sel[j] = bv; lg[bi] = -INFINITY;
    }
}

kernel void u_softmax_k(const device float* sel [[buffer(0)]],
                        device float*       wgt [[buffer(1)]],
                        constant MoEParams& P   [[buffer(2)]],
                        uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    float mx = sel[0], sum = 0.0f;
    for (uint j = 1; j < P.top_k; ++j) mx = max(mx, sel[j]);
    for (uint j = 0; j < P.top_k; ++j) sum += exp(sel[j] - mx);
    for (uint j = 0; j < P.top_k; ++j) wgt[j] = exp(sel[j] - mx) / sum;
}

kernel void u_xsum(const device uint4*  v  [[buffer(0)]],
                   device float*        xs [[buffer(1)]],
                   constant uint&       nq [[buffer(2)]],
                   uint j [[thread_position_in_grid]]) {
    if (j >= nq) return;
    float4 t = float4(0);
    for (uint u = 0; u < 4; ++u) { float4 a, b; unpack8(v[j*4+u], a, b); t += a + b; }
    xs[j] = t.x + t.y + t.z + t.w;
}

kernel void u_gemv(const device uchar*  pool [[buffer(0)]],
                   const device uint*   idx  [[buffer(1)]],
                   const device uint4*  x    [[buffer(2)]],
                   const device float*  xs   [[buffer(3)]],
                   device float*        y    [[buffer(4)]],
                   constant MoEParams&  P    [[buffer(5)]],
                   constant GemvJob&    J    [[buffer(6)]],
                   uint tg_id [[threadgroup_position_in_grid]],
                   uint sg_in [[simdgroup_index_in_threadgroup]],
                   uint n_sg  [[simdgroups_per_threadgroup]],
                   uint lane  [[thread_index_in_simdgroup]]) {
    uint m = tg_id * n_sg + sg_in;
    if (m >= J.M) return;
    const device uchar* bb = pool + (ulong)idx[J.slot_i] * P.blob;
    const device uint4*  W = (const device uint4*)(bb + J.proj + P.sub_w) + (ulong)m * J.quads;
    const device bfloat* S = (const device bfloat*)(bb + J.proj + P.sub_s) + (ulong)m * J.groups;
    const device bfloat* B = (const device bfloat*)(bb + J.proj + P.sub_b) + (ulong)m * J.groups;
    float acc = 0.0f;
    for (uint q = lane; q < J.quads; q += 32) {
        uint4 p  = W[q];
        uint  k4 = q * 4;
        float4 a0,a1,b0,b1,c0,c1,d0,d1;
        unpack8(x[k4+0], a0, a1); unpack8(x[k4+1], b0, b1);
        unpack8(x[k4+2], c0, c1); unpack8(x[k4+3], d0, d1);
        float d  = qdot8(p.x, a0, a1) + qdot8(p.y, b0, b1)
                 + qdot8(p.z, c0, c1) + qdot8(p.w, d0, d1);
        acc += float(S[q / P.chunks_pg]) * d + float(B[q / P.chunks_pg]) * xs[q];
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        const device bfloat* bi = (const device bfloat*)(bb + J.proj + P.sub_bi);
        y[m] = acc + float(bi[m]);
    }
}

kernel void u_swiglu(const device float* g [[buffer(0)]],
                     const device float* u [[buffer(1)]],
                     device half*        o [[buffer(2)]],
                     constant uint&      n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float gg = min(g[i], 7.0f), ll = clamp(u[i], -7.0f, 7.0f);
    o[i] = half((gg / (1.0f + exp(-1.702f * gg))) * (ll + 1.0f));
}

kernel void u_axpy(const device float* v   [[buffer(0)]],
                   const device float* wgt [[buffer(1)]],
                   device float*       y   [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   constant uint&      e   [[buffer(4)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    y[i] = (e == 0 ? 0.0f : y[i]) + wgt[e] * v[i];
}


// ===========================================================================
// Atomic variant of passes 2 and 3.
//
// Pass 3 needs sum(h_e) per 32-chunk. Recomputing it in every threadgroup means
// each of the 360 threadgroups reads all 46 KB of h -- 16.6 MB of extra reads on
// top of 53 MB of weights, cache-resident but not free. Instead pass 2 folds
// each h value into a device atomic_float as it is produced (11520 adds over 360
// counters), and pass 3 reads 1.4 KB. The accumulator is zeroed by pass 1, so
// this still costs no extra dispatch.
// ===========================================================================
kernel void fused_gate_up_swiglu_at(const device uchar*   pool [[buffer(0)]],
                                    const device uint*    idx  [[buffer(1)]],
                                    const device uint4*   x    [[buffer(2)]],
                                    device half*          h    [[buffer(3)]],
                                    constant MoEParams&   P    [[buffer(4)]],
                                    device atomic_float*  xsh  [[buffer(5)]],
                                    threadgroup float*    xs   [[threadgroup(0)]],
                                    uint tg_id [[threadgroup_position_in_grid]],
                                    uint tid   [[thread_index_in_threadgroup]],
                                    uint nthr  [[threads_per_threadgroup]],
                                    uint sg_in [[simdgroup_index_in_threadgroup]],
                                    uint n_sg  [[simdgroups_per_threadgroup]],
                                    uint lane  [[thread_index_in_simdgroup]]) {
    for (uint j = tid; j < P.quads_hid; j += nthr) {
        float4 t = float4(0);
        for (uint u = 0; u < 4; ++u) {
            float4 a, b; unpack8(x[j*4+u], a, b); t += a + b;
        }
        xs[j] = t.x + t.y + t.z + t.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg_id * n_sg + sg_in;
    uint e   = row / P.inter;
    uint m   = row % P.inter;
    if (e >= P.top_k) return;

    const device uchar* bb = pool + (ulong)idx[e] * P.blob;
    float g = qrow(bb, P.off_gate, m, x, xs, P.quads_hid, P.groups_hid,
                   P.chunks_pg, lane, P);
    float u = qrow(bb, P.off_up,   m, x, xs, P.quads_hid, P.groups_hid,
                   P.chunks_pg, lane, P);
    if (lane == 0) {
        const device bfloat* bg = (const device bfloat*)(bb + P.off_gate + P.sub_bi);
        const device bfloat* bu = (const device bfloat*)(bb + P.off_up   + P.sub_bi);
        g += float(bg[m]); u += float(bu[m]);
        float gg = min(g, 7.0f), ll = clamp(u, -7.0f, 7.0f);
        float hv = (gg / (1.0f + exp(-1.702f * gg))) * (ll + 1.0f);
        h[(ulong)e * P.inter + m] = half(hv);
        atomic_fetch_add_explicit(&xsh[e * P.quads_int + m / 32], hv,
                                  memory_order_relaxed);
    }
}

kernel void fused_down_combine_at(const device uchar*  pool [[buffer(0)]],
                                  const device uint*   idx  [[buffer(1)]],
                                  const device float*  wgt  [[buffer(2)]],
                                  const device uint4*  h    [[buffer(3)]],
                                  device float*        y    [[buffer(4)]],
                                  constant MoEParams&  P    [[buffer(5)]],
                                  const device float*  xsh  [[buffer(6)]],
                                  threadgroup float*   xs   [[threadgroup(0)]],
                                  uint tg_id [[threadgroup_position_in_grid]],
                                  uint tid   [[thread_index_in_threadgroup]],
                                  uint nthr  [[threads_per_threadgroup]],
                                  uint sg_in [[simdgroup_index_in_threadgroup]],
                                  uint n_sg  [[simdgroups_per_threadgroup]],
                                  uint lane  [[thread_index_in_simdgroup]]) {
    uint tot = P.top_k * P.quads_int;
    for (uint j = tid; j < tot; j += nthr) xs[j] = xsh[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m = tg_id * n_sg + sg_in;
    if (m >= P.hidden) return;

    float acc = 0.0f;
    for (uint e = 0; e < P.top_k; ++e) {
        const device uchar* bb = pool + (ulong)idx[e] * P.blob;
        const device uint4* he = h + (ulong)e * (P.inter / 8);
        float d = qrow(bb, P.off_down, m, he, xs + e * P.quads_int,
                       P.quads_int, P.groups_int, P.chunks_pg, lane, P);
        if (lane == 0) {
            const device bfloat* bd = (const device bfloat*)(bb + P.off_down + P.sub_bi);
            acc += wgt[e] * (d + float(bd[m]));
        }
    }
    if (lane == 0) y[m] = acc;
}
