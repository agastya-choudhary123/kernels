#!/usr/bin/env python3
"""Close the loop: run the Metal VQ kernel on a real Qwen weight matrix.

The perplexity numbers come from dequantising to fp16 and running the stock
forward pass, which measures the quantiser. This script checks the other half of
the claim -- that the kernel computes that same reconstruction -- by quantising a
real weight matrix from Qwen2.5-0.5B-Instruct, running gemv_vq_shared on it via
./verify_real, and comparing against the dequantise-then-matmul reference
computed independently in MLX.
"""
import os, subprocess, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "quant"))

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
import pq_quantize as pq

OUT = os.path.join(ROOT, "results", "real")


def main():
    os.makedirs(OUT, exist_ok=True)
    model, _ = load("Qwen/Qwen2.5-0.5B-Instruct")
    params = dict(tree_flatten(model.parameters()))
    # widest projection in the model, so the kernel sees a realistic K
    name = "model.layers.12.mlp.down_proj.weight"
    W = params[name].astype(mx.float32)
    M, K = W.shape
    print(f"{name}  M={M} K={K}")

    idx, C, S = pq.quantize_matrix(W)
    mx.eval(idx, C, S)
    mx.random.seed(0)
    x = mx.random.normal((K,)).astype(mx.float16)
    mx.eval(x)

    np.array(idx).astype(np.uint8).tofile(f"{OUT}/idx.u8")
    np.array(C).astype(np.float16).tofile(f"{OUT}/cb.f16")
    np.array(S).astype(np.float16).tofile(f"{OUT}/scale.f16")
    np.array(x).astype(np.float16).tofile(f"{OUT}/x.f16")
    open(f"{OUT}/meta.txt", "w").write(f"{M} {K}\n")

    subprocess.run([os.path.join(ROOT, "verify_real"), OUT], cwd=ROOT, check=True)

    D = pq.dequantize(idx, C, S)                      # independent reconstruction
    ref = np.array((D @ x.astype(mx.float32)))
    got = np.fromfile(f"{OUT}/y_metal.f32", dtype=np.float32)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"  metal kernel vs MLX dequant-matmul: max|d|/|ref|inf = {rel:.2e}  "
          f"{'ok' if rel < 2e-3 else 'MISMATCH'}")

    # and the quantiser's own error on this real matrix, for context
    Wf = np.array(W)
    err = np.sqrt(((np.array(D) - Wf) ** 2).mean()) / np.sqrt((Wf ** 2).mean())
    q, s, b = mx.quantize(W, group_size=64, bits=4)
    D4 = np.array(mx.dequantize(q, s, b, group_size=64, bits=4))
    err4 = np.sqrt(((D4 - Wf) ** 2).mean()) / np.sqrt((Wf ** 2).mean())
    print(f"  relative RMS reconstruction error on this tensor: "
          f"VQ {err:.5f}   int4 {err4:.5f}")
    return 0 if rel < 2e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
