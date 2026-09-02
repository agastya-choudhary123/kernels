#!/usr/bin/env python3
"""Bandwidth of mlx.quantized_matmul on this project's shapes, same protocol.

MLX has no vector-quantised matmul, so there is nothing to compare the VQ kernel
against directly. What can be compared is bytes moved per second at identical
bytes/param: this project's VQ format and MLX's affine int4 both spend 4.5
bits/weight, so GB/s is an apples-to-apples number even though the arithmetic
differs.

Protocol matches ../int4-gemv and the Metal harness: rotate over enough distinct
weight copies that nothing is served from the 8 MB SLC, and batch enough calls
per timed span that per-call dispatch is amortised the same way.
"""
import sys, time
import mlx.core as mx

PEAK = 120.0


def gemv_bytes(M, K, G):
    return (M * K // 2 + 2 * M * (K // G) * 2 + K * 2 + (K // 32) * 4 + M * 4)


def bandwidth(M, K, G=64, iters=15, rot_mb=200, span_ms=30.0):
    nbytes = gemv_bytes(M, K, G)
    wbytes = M * K // 2 + 2 * M * (K // G) * 2
    ncopy = max(1, min(64, -(-int(rot_mb * 2**20) // wbytes)))
    x = mx.random.normal((1, K), key=mx.random.key(0)).astype(mx.float16)
    copies = []
    for i in range(ncopy):
        w = mx.random.normal((M, K), key=mx.random.key(i)).astype(mx.float16)
        q, s, b = mx.quantize(w, group_size=G, bits=4)
        mx.eval(q, s, b)
        copies.append((q, s, b)); del w
    mx.eval(x)

    def batch(R, off=0):
        ys = []
        for r in range(R):
            q, s, b = copies[(off + r) % ncopy]
            ys.append(mx.quantized_matmul(x, q, scales=s, biases=b,
                                          transpose=True, group_size=G, bits=4))
        mx.eval(ys)

    batch(1)
    t0 = time.perf_counter(); batch(1); probe = time.perf_counter() - t0
    R = max(1, min(64, int(-(-span_ms * 1e-3 // probe))))
    for _ in range(3):
        batch(R)
    ts = []
    for i in range(iters):
        t0 = time.perf_counter(); batch(R, off=i * R)
        ts.append((time.perf_counter() - t0) / R)
    best = min(ts)
    print(f"  mlx.quantized_matmul  M={M} K={K} g={G}  [{ncopy} copies, {R}/span]"
          f"  {best*1e3:7.3f} ms  {nbytes/best/1e9:7.2f} GB/s  "
          f"{100*nbytes/best/1e9/PEAK:.1f}% of peak")
    return nbytes / best / 1e9


if __name__ == "__main__":
    print("MLX affine int4 at the same 0.5625 bytes/param:")
    for (M, K) in [(8192, 8192), (16384, 16384), (32768, 16384)]:
        bandwidth(M, K)
