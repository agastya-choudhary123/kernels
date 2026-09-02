#!/usr/bin/env python3
"""Real perplexity of Qwen2.5-0.5B-Instruct on WikiText-2, fp16 vs affine-int4
vs codebook-VQ, at identical bytes/param.

No proxy metric and no synthetic weights: the model is loaded, the transformer
block projections are replaced by their quantise->dequantise round trip, and the
whole WikiText-2 test split is scored. Both schemes touch exactly the same set of
tensors and spend exactly 4.500 bits/weight, so the difference is the quantiser.

Dequantising to fp16 and running the stock forward pass is the right way to
measure a *quantiser*: it isolates representation error from kernel arithmetic.
That the Metal kernel reproduces this same reconstruction is established
separately, on these same real tensors, by bench/verify_real.py.
"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quant"))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
import pq_quantize as pq

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
WIKI = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
    "test-00000-of-00001.parquet")
CTX = 2048


def target_keys(model):
    """Every 2-D projection weight inside a transformer block."""
    return [k for k, v in tree_flatten(model.parameters())
            if k.startswith("model.layers.") and k.endswith(".weight") and v.ndim == 2]


def quantize_all(orig, scheme):
    out, nbits, nparam = {}, 0.0, 0
    for k, W in orig.items():
        if scheme == "int4":
            q, s, b = mx.quantize(W, group_size=64, bits=4)
            D = mx.dequantize(q, s, b, group_size=64, bits=4)
            bits = W.size * 4 + s.size * 16 + b.size * 16
        elif scheme == "vq":
            idx, C, s = pq.quantize_matrix(W)
            D = pq.dequantize(idx, C, s)
            bits = idx.size * 8 + s.size * 16 + C.size * 16
        else:
            raise ValueError(scheme)
        mx.eval(D)
        out[k] = D.astype(W.dtype)
        nbits += bits
        nparam += W.size
    return out, nbits / nparam, nparam


def perplexity(model, ids, ctx=CTX):
    n = (len(ids) - 1) // ctx
    tot, cnt = 0.0, 0
    for i in range(n):
        chunk = ids[i * ctx: i * ctx + ctx + 1]
        inp = mx.array(chunk[:-1])[None]
        tgt = mx.array(chunk[1:])[None]
        logits = model(inp).astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        picked = mx.take_along_axis(logits, tgt[..., None], axis=-1).squeeze(-1)
        tot += float(mx.sum(lse - picked))
        cnt += tgt.size
        mx.eval(tot)
    import math
    return math.exp(tot / cnt), cnt


def main():
    import pandas as pd
    model, tok = load(MODEL)
    text = "\n\n".join(pd.read_parquet(WIKI)["text"].tolist())
    ids = tok.encode(text)
    print(f"model {MODEL}")
    print(f"wikitext-2 test: {len(text)} chars -> {len(ids)} tokens, "
          f"{(len(ids)-1)//CTX} windows of {CTX}\n")

    keys = target_keys(model)
    orig = {k: v for k, v in tree_flatten(model.parameters()) if k in keys}
    nq = sum(v.size for v in orig.values())
    tot = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"quantising {len(keys)} tensors, {nq/1e6:.1f}M of {tot/1e6:.1f}M params\n")

    rows = []
    t0 = time.time()
    ppl, ntok = perplexity(model, ids)
    rows.append(("fp16 (unquantised)", 16.0, ppl))
    print(f"  {'fp16 (unquantised)':<24} bits/w 16.000   ppl {ppl:8.4f}   "
          f"[{time.time()-t0:.0f}s, {ntok} tokens]")

    for scheme, label in [("int4", "affine int4 g=64"), ("vq", "codebook VQ d=2 K=256")]:
        t0 = time.time()
        new, bpw, _ = quantize_all(orig, scheme)
        tq = time.time() - t0
        model.update(tree_unflatten(list(new.items())))
        mx.eval(model.parameters())
        t0 = time.time()
        ppl, _ = perplexity(model, ids)
        rows.append((label, bpw, ppl))
        print(f"  {label:<24} bits/w {bpw:6.3f}   ppl {ppl:8.4f}   "
              f"[quant {tq:.0f}s, eval {time.time()-t0:.0f}s]")
        model.update(tree_unflatten(list(orig.items())))
        mx.eval(model.parameters())

    base = rows[0][2]
    print(f"\n  {'scheme':<24} {'bits/w':>7} {'ppl':>9} {'d vs fp16':>10}")
    for label, bpw, ppl in rows:
        print(f"  {label:<24} {bpw:7.3f} {ppl:9.4f} {ppl-base:+10.4f}")
    i4 = next(r for r in rows if r[0].startswith("affine"))
    vq = next(r for r in rows if r[0].startswith("codebook"))
    print(f"\n  at equal {vq[1]:.3f} bits/weight: VQ ppl {vq[2]:.4f} vs int4 {i4[2]:.4f} "
          f"({vq[2]-i4[2]:+.4f}, {100*(vq[2]-i4[2])/i4[2]:+.2f}%)")


if __name__ == "__main__":
    main()
