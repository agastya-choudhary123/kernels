#!/usr/bin/env python3
"""Cross-check the Metal expert kernel against MLX on real gpt-oss-120b bytes.

Reads the same expert blob out of experts.bin that ./moepipe --verify just
computed, rebuilds the three projections and the SwiGLU with mlx.core, and
compares. Nothing synthetic: these are the bytes the SSD actually delivered.
"""
import os, sys
import numpy as np
import mlx.core as mx

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "verify")
STORE = os.path.expanduser("~/Desktop/moe-stream/model-120b/experts.bin")
OFF = {"gate": 0, "up": 4671360, "down": 9342720}
SUB_W, SUB_S, SUB_B, SUB_BI = 0, 4147200, 4406400, 4665600


def bf16(buf):
    """bf16 bytes -> fp32, exactly (bf16 is the top 16 bits of fp32)."""
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def main():
    off0, HID, INTER, G = (int(v) for v in open(f"{D}/meta.txt").read().split())
    x = np.fromfile(f"{D}/x.f32", dtype=np.float32)
    with open(STORE, "rb") as f:
        f.seek(off0); blob = f.read(14024704)

    def proj(name, vec, M, K):
        b = OFF[name]
        W = np.frombuffer(blob[b+SUB_W: b+SUB_W+M*(K//8)*4], dtype=np.uint32).reshape(M, K//8)
        S = bf16(blob[b+SUB_S: b+SUB_S+M*(K//G)*2]).reshape(M, K//G)
        B = bf16(blob[b+SUB_B: b+SUB_B+M*(K//G)*2]).reshape(M, K//G)
        BI = bf16(blob[b+SUB_BI: b+SUB_BI+M*2])
        y = mx.quantized_matmul(mx.array(vec).reshape(1, K), mx.array(W),
                                scales=mx.array(S), biases=mx.array(B),
                                transpose=True, group_size=G, bits=4)
        mx.eval(y)
        return np.array(y).reshape(-1) + BI

    gate = proj("gate", x, INTER, HID)
    up   = proj("up",   x, INTER, HID)
    g = np.minimum(gate, 7.0); l = np.clip(up, -7.0, 7.0)
    h = (g / (1.0 + np.exp(-1.702 * g))) * (l + 1.0)
    y = proj("down", h, HID, INTER)

    ok = True
    for name, ref in (("gate", gate), ("up", up), ("h", h), ("y", y)):
        got = np.fromfile(f"{D}/{name}.f32", dtype=np.float32)
        rel = np.abs(got - ref).max() / max(1e-30, np.abs(ref).max())
        flag = "ok" if rel < 2e-3 else "MISMATCH"
        if rel >= 2e-3: ok = False
        print(f"  {name:<6} vs MLX: max|d|/|ref|inf = {rel:.2e}  {flag}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
