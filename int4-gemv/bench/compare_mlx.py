#!/usr/bin/env python3
"""Independent cross-check and bandwidth comparison against MLX.

Two separate jobs:
  1. Correctness. Load the tensors the Metal harness dumped and run them
     through mlx.core.quantized_matmul, which is a completely independent
     implementation of the same layout. If the kernel and MLX disagree, the
     kernel is wrong regardless of what the CPU reference said.
  2. Bandwidth. Time MLX's own int4 GEMV on the benchmark shape and convert to
     GB/s with the identical byte formula the harness uses, so the two numbers
     are directly comparable.
"""
import glob, os, subprocess, sys, time
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
PEAK = 120.0  # GB/s, M4 base: LPDDR5X-7500 on a 128-bit bus


def gemv_bytes(M, K, G):
    """Exactly what one decode GEMV must pull through the memory system."""
    return (M * K // 2          # int4 weights
            + 2 * M * (K // G) * 2  # fp16 scale + fp16 bias
            + K * 2             # x, fp16
            + (K // 32) * 4     # per-chunk x partial sums
            + M * 4)            # y, fp32


def correctness():
    M, K, G = map(int, open(os.path.join(RES, "shape.txt")).read().split())
    W = np.fromfile(f"{RES}/W.u32", dtype=np.uint32).reshape(M, K // 8)
    S = np.fromfile(f"{RES}/S.f16", dtype=np.float16).reshape(M, K // G)
    B = np.fromfile(f"{RES}/B.f16", dtype=np.float16).reshape(M, K // G)
    x = np.fromfile(f"{RES}/x.f16", dtype=np.float16)

    y = mx.quantized_matmul(mx.array(x).reshape(1, K).astype(mx.float32),
                            mx.array(W), scales=mx.array(S).astype(mx.float32),
                            biases=mx.array(B).astype(mx.float32),
                            transpose=True, group_size=G, bits=4)
    mx.eval(y)
    ymlx = np.array(y).reshape(-1)
    scale = np.abs(ymlx).max()

    print(f"cross-check vs mlx.quantized_matmul   M={M} K={K} group={G}")
    print(f"  {'variant':<36} {'max|d|/|mlx|inf':>16}")
    ok = True
    for f in sorted(glob.glob(f"{RES}/y_*.f32")):
        name = os.path.basename(f)[2:-4].replace("_", " ").strip()
        g = np.fromfile(f, dtype=np.float32)
        rel = np.abs(g - ymlx).max() / scale
        # both sides accumulate int4*fp16 products; 2e-3 is well outside fp16
        # rounding noise and well inside anything a real bug would produce
        flag = "ok" if rel < 2e-3 else "MISMATCH"
        if rel >= 2e-3:
            ok = False
        print(f"  {name:<36} {rel:>16.2e}  {flag}")
    return ok


def bandwidth(M, K, G, iters=15, rot_mb=200, span_ms=5.0):
    """Same protocol as the Metal harness, or the comparison is meaningless.

    Two things have to match. (1) Rotate over enough distinct weight copies that
    no call re-reads what the last one left in the 8 MB SLC. (2) Batch enough
    calls per timed span that per-call dispatch overhead is amortised the same
    way the harness amortises it across dispatches in one command buffer.
    """
    nbytes = gemv_bytes(M, K, G)
    wbytes = M * K // 2 + 2 * M * (K // G) * 2
    ncopy = max(1, min(64, -(-int(rot_mb * 2**20) // wbytes)))

    key = mx.random.key(0)
    x = mx.random.normal((1, K), key=key).astype(mx.float16)
    copies = []
    for i in range(ncopy):
        w = mx.random.normal((M, K), key=mx.random.key(i)).astype(mx.float16)
        q, s, b = mx.quantize(w, group_size=G, bits=4)
        mx.eval(q, s, b)
        copies.append((q, s, b))
        del w
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
        t0 = time.perf_counter()
        batch(R, off=i * R)
        ts.append((time.perf_counter() - t0) / R)
    best = min(ts)
    print(f"\nmlx.quantized_matmul   M={M} K={K} group={G}"
          f"  [{ncopy} weight copies, {R} calls/timed span]")
    print(f"  best {best*1e3:.3f} ms   {nbytes/best/1e9:7.2f} GB/s   "
          f"{100*nbytes/best/1e9/PEAK:.1f}% of {PEAK:.0f} GB/s peak")
    return nbytes / best / 1e9


if __name__ == "__main__":
    ok = correctness()
    for (M, K) in [(8192, 8192), (16384, 16384), (32768, 16384)]:
        bandwidth(M, K, 64)
    sys.exit(0 if ok else 1)
