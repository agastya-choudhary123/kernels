#!/usr/bin/env python3
"""Vector (product) quantiser matching the kernel's on-disk format exactly.

    d = 2, codebook of 256 two-dimensional codewords, one fp16 scale per 32
    weights.  ->  8 bits / 2 weights + 16 bits / 32 weights = 4.500 bits/weight

which is byte-for-byte the same budget as affine int4 with group_size 64
(4 bits/weight + fp16 scale + fp16 bias per 64 = 4.500 bits/weight). The
codebook is 256 x 2 fp16 = 1 KB per matrix, under 0.001 bits/weight at these
sizes and counted in the reported figure anyway.

Reconstruction, identical to what the Metal kernel computes:

    w[m, 2i+t] = scale[m, i//16] * codebook[idx[m, i]][t]
"""
import mlx.core as mx

GROUP = 32          # weights per fp16 scale
K_CB = 256          # codebook entries
DIM = 2             # subvector width


def bits_per_weight(group=GROUP, kcb=K_CB, dim=DIM):
    import math
    return math.log2(kcb) / dim + 16.0 / group


def _kmeans(P, k, iters=25, seed=0, sub=200_000):
    """Lloyd on a subsample of 2-D points P [N, 2]. Returns [k, 2] centroids.

    Assignment is done as a matmul: argmin_c ||p-c||^2 = argmax_c (p.c - |c|^2/2),
    so each pass is one [N,2] x [2,k] product instead of an [N,k,2] tensor.
    """
    N = P.shape[0]
    key = mx.random.key(seed)
    S = P if N <= sub else P[mx.random.randint(0, N, (sub,), key=key)]
    C = S[mx.random.randint(0, S.shape[0], (k,), key=key)].astype(mx.float32)
    for _ in range(iters):
        score = S @ C.T - 0.5 * mx.sum(C * C, axis=1)[None, :]
        a = mx.argmax(score, axis=1)
        oh = (a[:, None] == mx.arange(k)[None, :]).astype(mx.float32)   # [n, k]
        cnt = oh.sum(0)
        new = (oh.T @ S) / mx.maximum(cnt, 1)[:, None]
        # keep empty clusters where they were rather than collapsing them to 0
        C = mx.where((cnt > 0)[:, None], new, C)
        mx.eval(C)
    return C


def quantize_matrix(W, group=GROUP, kcb=K_CB, seed=0):
    """W [M, K] float -> (idx uint8 [M, K/2], codebook [kcb, 2] f16, scale f16 [M, K/32])."""
    M, K = W.shape
    assert K % group == 0, f"K={K} not divisible by group={group}"
    Wf = W.astype(mx.float32)

    g = Wf.reshape(M, K // group, group)
    scale = mx.max(mx.abs(g), axis=2)                     # [M, K/group]
    safe = mx.where(scale > 0, scale, mx.array(1.0, mx.float32))
    U = (g / safe[:, :, None]).reshape(M, K // 2, 2)      # normalised pairs

    C = _kmeans(U.reshape(-1, 2), kcb, seed=seed)

    # assign in row blocks so the [n, kcb] score block stays small
    cn = 0.5 * mx.sum(C * C, axis=1)[None, :]
    idx = []
    step = max(1, (1 << 22) // max(1, K // 2))
    for a in range(0, M, step):
        P = U[a:a + step].reshape(-1, 2)
        idx.append(mx.argmax(P @ C.T - cn, axis=1).astype(mx.uint8))
        mx.eval(idx[-1])
    idx = mx.concatenate(idx).reshape(M, K // 2)

    # round-trip the codebook and scales through fp16, since that is what the
    # kernel actually reads -- quality must be measured on the stored values
    return idx, C.astype(mx.float16), scale.astype(mx.float16)


def dequantize(idx, C, scale, group=GROUP):
    M, half = idx.shape
    K = half * 2
    cw = C.astype(mx.float32)[idx.astype(mx.uint32)]       # [M, K/2, 2]
    W = cw.reshape(M, K // group, group)
    return (W * scale.astype(mx.float32)[:, :, None]).reshape(M, K)
