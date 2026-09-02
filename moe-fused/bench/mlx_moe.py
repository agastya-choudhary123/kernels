#!/usr/bin/env python3
"""The same MoE decode block written with MLX ops, for wall-clock context.

Uses mx.gather_qmm, which is MLX's own fused-gather quantised matmul, so this is
not a strawman: it is the fastest way to express this block in MLX today. The
weights are the same real gpt-oss-120b experts the Metal harness uses, restacked
into the [n_experts, out, in/8] layout gather_qmm wants.

Caveat stated plainly: MLX's kernel launch count cannot be measured from here.
mx.metal.start_capture writes a .gputrace whose store is compressed and needs
Xcode tooling to read, and no Xcode is installed on this machine. Only wall time
is reported for MLX; the launch counts in the README are counted by the Metal
harness for its own two paths.
"""
import os, sys, time
import numpy as np
import mlx.core as mx

STORE = os.path.expanduser("~/Desktop/moe-stream/model-120b/experts.bin")
BLOB = 14024704
OFF = {"gate": 0, "up": 4671360, "down": 9342720}
SUB_W, SUB_S, SUB_B, SUB_BI = 0, 4147200, 4406400, 4665600
HID = INTER = 2880
G, BITS, TOPK = 64, 4, 4


def load(n_experts):
    W = {k: np.empty((n_experts, HID, HID // 8), np.uint32) for k in OFF}
    S = {k: np.empty((n_experts, HID, HID // G), np.uint16) for k in OFF}
    B = {k: np.empty((n_experts, HID, HID // G), np.uint16) for k in OFF}
    BI = {k: np.empty((n_experts, HID), np.uint16) for k in OFF}
    with open(STORE, "rb") as f:
        for e in range(n_experts):
            f.seek(e * BLOB)
            blob = f.read(BLOB)
            for k, o in OFF.items():
                W[k][e] = np.frombuffer(blob[o+SUB_W:o+SUB_W+HID*(HID//8)*4],
                                        np.uint32).reshape(HID, HID//8)
                S[k][e] = np.frombuffer(blob[o+SUB_S:o+SUB_S+HID*(HID//G)*2],
                                        np.uint16).reshape(HID, HID//G)
                B[k][e] = np.frombuffer(blob[o+SUB_B:o+SUB_B+HID*(HID//G)*2],
                                        np.uint16).reshape(HID, HID//G)
                BI[k][e] = np.frombuffer(blob[o+SUB_BI:o+SUB_BI+HID*2], np.uint16)
    to_bf = lambda a: mx.array(a).view(mx.bfloat16)
    return ({k: mx.array(v) for k, v in W.items()},
            {k: to_bf(v) for k, v in S.items()},
            {k: to_bf(v) for k, v in B.items()},
            {k: to_bf(v) for k, v in BI.items()})


VARIANT = "gather"


def main():
    global VARIANT
    n_experts = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    VARIANT = sys.argv[2] if len(sys.argv) > 2 else "gather"
    layers = 36
    print(f"loading {n_experts} experts into MLX gather_qmm layout ...", flush=True)
    t0 = time.perf_counter()
    W, S, B, BI = load(n_experts)
    mx.eval(list(W.values()), list(S.values()), list(B.values()), list(BI.values()))
    print(f"  {time.perf_counter()-t0:.1f} s, {mx.metal.get_active_memory()/1e9:.2f} GB active")

    # must match the C++ harness bit for bit: uint32 wrap-around multiply
    idx_i = np.arange(n_experts * HID, dtype=np.uint64)
    h = (((idx_i * np.uint64(1103515245) + np.uint64(12345)) & np.uint64(0xFFFFFFFF))
         >> np.uint64(16)).astype(np.uint32)
    R = mx.array((0.02 * ((h % np.uint32(2001)).astype(np.int64) - 1000) / 1000.0)
                 .reshape(n_experts, HID).astype(np.float32))
    # fp16 activations, matching the Metal kernel: the comparison has to run both
    # sides on the same activation precision or it is measuring dtype, not code
    x = mx.array((0.05 * ((np.arange(HID) % 17) - 8)).astype(np.float32)).reshape(1, HID)
    x16 = x.astype(mx.float16)
    mx.eval(R, x)

    def block(x):
        lg = (x @ R.T)[0]        # router stays fp32; it is 1.5 MB and needs the range
        idx = mx.argpartition(lg, kth=-TOPK)[-TOPK:]
        sel = lg[idx]
        order = mx.argsort(-sel)
        idx, sel = idx[order], sel[order]
        w = mx.softmax(sel)
        ri = idx
        xb = mx.broadcast_to(x16, (TOPK, HID)).reshape(TOPK, 1, HID)
        arange4 = mx.arange(TOPK, dtype=mx.uint32)
        def proj(name, v):
            if VARIANT == "take":
                # Slice the k experts out first, then gather_qmm over a k-expert
                # tensor. Costs an explicit 16.6 MB copy but avoids whatever in
                # gather_qmm scales with the *total* expert count -- 2.8x faster
                # per projection at 128 experts, so this is MLX's best showing.
                ws = mx.take(W[name], idx, axis=0)
                ss = mx.take(S[name], idx, axis=0)
                bs = mx.take(B[name], idx, axis=0)
                y = mx.gather_qmm(v, ws, scales=ss, biases=bs,
                                  rhs_indices=arange4, transpose=True,
                                  group_size=G, bits=BITS)
            else:
                y = mx.gather_qmm(v, W[name], scales=S[name], biases=B[name],
                                  rhs_indices=ri, transpose=True,
                                  group_size=G, bits=BITS)
            return y.reshape(TOPK, HID) + BI[name][idx].astype(mx.float32)
        g = proj("gate", xb)
        u = proj("up", xb)
        gg = mx.minimum(g, 7.0); ll = mx.clip(u, -7.0, 7.0)
        hh = ((gg * mx.sigmoid(1.702 * gg)) * (ll + 1.0)).astype(mx.float16)
        y = proj("down", hh.reshape(TOPK, 1, HID))
        return (w[:, None] * y).sum(0)

    for _ in range(10):
        mx.eval(block(x))

    # Each block must be evaluated. Building n graphs and evaluating only the
    # last one measures Python graph construction, not the GPU -- the first
    # version of this loop did exactly that and reported 155 GB/s, above the
    # machine's 120 GB/s peak, which is how the mistake showed itself.
    n = 20
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        for _ in range(n):
            mx.eval(block(x))
        ts.append((time.perf_counter() - t0) / n)
    best = min(ts)

    # and the same work with all n blocks submitted before a single eval, which
    # is the friendliest case for MLX's scheduling
    tb = []
    for _ in range(10):
        t0 = time.perf_counter()
        outs = [block(x) for _ in range(n)]
        mx.eval(outs)
        tb.append((time.perf_counter() - t0) / n)
    best_b = min(tb)
    print(f"MLX, {n} blocks per eval : {best_b*1e3:.4f} ms  "
          f"{(TOPK*3*4147200 + TOPK*6*259200)/best_b/1e9:.2f} GB/s")
    out = block(x); mx.eval(out)
    mb = TOPK * 3 * 4147200 + TOPK * 6 * 259200
    print(f"\nMLX MoE block [{VARIANT}]: {best*1e3:.4f} ms  "
          f"{mb/best/1e9:.2f} GB/s  {1.0/(layers*best):.1f} tok/s ({layers} layers)")
    np.array(out).astype(np.float32).tofile(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "y_mlx.f32"))
    print("wrote results/y_mlx.f32")


if __name__ == "__main__":
    main()
