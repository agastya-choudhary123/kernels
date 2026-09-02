// GPU side of the stream-and-compute MoE pipeline.
//
// One gpt-oss-120b expert, exactly as it sits in moe-stream's repacked
// experts.bin: gate_proj, up_proj and down_proj, each [2880, 2880] in MLX int4
// with group_size 64, and each carrying its own BF16 scales, BF16 biases and a
// BF16 output bias. One expert is 14,024,704 bytes and the three GEMVs below
// read essentially all of it, so GPU compute and SSD traffic cover the same
// bytes -- which is the whole point: the pipeline is trying to make one hide
// the other.
//
// Scales and biases are BF16 here rather than the FP16 of ../int4-gemv, because
// that is how gpt-oss ships. MSL has a native `bfloat`, so no conversion pass is
// needed on the streamed bytes: they are read in place, in the layout the SSD
// delivered them.

#include <metal_stdlib>
using namespace metal;

struct ExpertParams {
    uint M;         // output rows      (2880)
    uint K;         // reduction length (2880)
    uint G;         // quant group      (64)
    uint quads_pr;  // uint4 per row    = K/32
    uint groups_pr; // groups per row   = K/G
    uint chunks_pg; // chunks per group = G/32
};

inline float qdot8(uint p, float4 xa, float4 xb) {
    float4 v0 = float4( float( p        & 0xF), float((p >>  4) & 0xF),
                        float((p >>  8) & 0xF), float((p >> 12) & 0xF) );
    float4 v1 = float4( float((p >> 16) & 0xF), float((p >> 20) & 0xF),
                        float((p >> 24) & 0xF), float((p >> 28) & 0xF) );
    return dot(v0, xa) + dot(v1, xb);
}

// sum(x) over each 32-element chunk, once per projection input.
kernel void xsum32(const device float4*  x  [[buffer(0)]],
                   device float*         xs [[buffer(1)]],
                   constant ExpertParams& P [[buffer(2)]],
                   uint j [[thread_position_in_grid]]) {
    if (j >= P.K / 32) return;
    float s = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        float4 v = x[j * 8 + i];
        s += v.x + v.y + v.z + v.w;
    }
    xs[j] = s;
}

// One simdgroup per output row, 16-byte weight loads -- the shape that won in
// ../int4-gemv. Scales and biases live in separate arrays here (that is the
// on-disk layout), so this cannot use the fused half2 trick from that project.
kernel void gemv_int4_bf16(const device uint4*    W  [[buffer(0)]],
                           const device bfloat*   S  [[buffer(1)]],
                           const device bfloat*   B  [[buffer(2)]],
                           const device bfloat*   BI [[buffer(3)]],  // output bias
                           const device float4*   x  [[buffer(4)]],
                           device float*          y  [[buffer(5)]],
                           const device float*    xs [[buffer(6)]],
                           constant ExpertParams& P  [[buffer(7)]],
                           uint tg_id [[threadgroup_position_in_grid]],
                           uint sg_in [[simdgroup_index_in_threadgroup]],
                           uint n_sg  [[simdgroups_per_threadgroup]],
                           uint lane  [[thread_index_in_simdgroup]]) {
    uint m = tg_id * n_sg + sg_in;
    if (m >= P.M) return;

    const device uint4*  wrow = W + (ulong)m * P.quads_pr;
    const device bfloat* srow = S + (ulong)m * P.groups_pr;
    const device bfloat* brow = B + (ulong)m * P.groups_pr;

    float acc = 0.0f;
    for (uint q = lane; q < P.quads_pr; q += 32) {
        uint4 p  = wrow[q];
        uint  k4 = q * 8;
        float d  = qdot8(p.x, x[k4+0], x[k4+1]) + qdot8(p.y, x[k4+2], x[k4+3])
                 + qdot8(p.z, x[k4+4], x[k4+5]) + qdot8(p.w, x[k4+6], x[k4+7]);
        uint g = q / P.chunks_pg;
        acc += float(srow[g]) * d + float(brow[g]) * xs[q];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[m] = acc + float(BI[m]);
}

// gpt-oss SwiGLU, matching mlx_lm.models.gpt_oss.swiglu exactly:
//   glu    = clip(gate, max=limit); lin = clip(up, -limit, limit)
//   out    = glu * sigmoid(alpha*glu) * (lin + 1)
kernel void swiglu_k(const device float* gate [[buffer(0)]],
                     const device float* up   [[buffer(1)]],
                     device float*       out  [[buffer(2)]],
                     constant uint&      n    [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float alpha = 1.702f, limit = 7.0f;
    float g = min(gate[i], limit);
    float l = clamp(up[i], -limit, limit);
    out[i] = (g / (1.0f + exp(-alpha * g))) * (l + 1.0f);
}
