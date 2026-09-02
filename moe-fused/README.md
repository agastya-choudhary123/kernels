# moe-fused — router + top-k + gather + int4 matmul in one pass

A gpt-oss-120b MoE decode block collapsed from 28 kernel launches to 3, with the
expert gather done inside the kernels: they read the top-k indices out of device
memory and compute the weight addresses themselves, so the host binds one pool
buffer once and never re-binds per expert.

128 real experts held resident (1.80 GB, read from moe-stream's `experts.bin`),
hidden 2880, top-4, int4 group 64 with BF16 scales.

## Result

| path | launches | ms/block | GB/s | tok/s |
|---|---|---|---|---|
| unfused (one kernel per op) | 28 | 0.650 – 0.670 | 83.6 – 86.1 | 41.5 – 42.7 |
| fused | **3** | 0.663 – 0.670 | 83.6 – 84.4 | 41.5 – 41.9 |
| **fused + atomic sum(h)** | **3** | **0.660 – 0.670** | **83.6 – 84.9** | **41.5 – 42.1** |
| `mlx.gather_qmm`, best formulation | — | 2.265 | 24.7 | 12.3 |

**9.3x fewer launches. Identical time. 3.4x faster than MLX.**

The first two of those are the point, and they do not match. Removing 25 of 28
kernel launches bought **nothing at all** — across three runs the fused and
unfused paths land in the same 0.650 – 0.670 ms band, and which one wins depends
on the run. Kernel launches inside a single command buffer are essentially free
on this hardware.

`tok/s` counts 36 MoE blocks per token with the experts already resident. It is
the MoE compute rate, not end-to-end decode, and not comparable to
[`../moe-pipeline`](../moe-pipeline), which is bound by streaming those experts
off the SSD in the first place.

## Why 3 and not 1

The block has exactly two grid-wide data dependencies, and Metal offers no
synchronisation across threadgroups inside one dispatch:

```
1. router + softmax + top-k        ->  idx[4], wgt[4]
2. gather + gate + up + SwiGLU     ->  h[4][inter]   (needs idx from 1)
3. gather + down + weighted combine->  y[hidden]     (needs all of h)
```

`h` cannot be consumed until every row of gate/up exists, and `y` cannot be
accumulated until every row of `h` exists. Three is the floor for this block
shape, not a stopping point. Getting to 1 would need a grid-wide barrier built
out of atomics and a persistent grid, which is only correct if every threadgroup
is co-resident — a property Metal does not guarantee and that hangs the GPU when
it does not hold. Not worth it for an op that is already bandwidth-bound.

Inside those three, the fusion is real: gate and up are computed back to back in
registers and SwiGLU is applied before anything reaches memory, so the
per-expert `gate` and `up` vectors never exist in device memory at all; and pass
3 accumulates `wgt[e] * down_e(h_e)` directly, so the per-expert output vectors
never exist either.

## What did move the number: bytes, twice

**fp16 activations.** x and `h` were fp32, read once per 16 bytes of weights and
re-read by every row — 128 bytes of activation loads per 16 bytes of weight, in
eight `float4` fetches. At fp16 the same values are four `uint4` fetches
bit-cast to `half4` pairs: half the transactions for identical arithmetic. x is
cache-resident, so this is issue slots, not bandwidth. Worth **0.710 → 0.660 ms**
across every path, and applied to the unfused baseline too.

**Atomic `sum(h)`.** Pass 3 needs `sum(h_e)` per 32-element chunk — the term
that lets the affine dequant factor. Computing it redundantly in every
threadgroup means each of the 360 threadgroups reads all of `h`. Folding each
`h` value into a `device atomic_float` as pass 2 produces it (11520 adds over 360
counters) drops that to a 1.4 KB read, and pass 1 zeroes the accumulator so it
costs no extra dispatch. This was worth ~2.5% when `h` was fp32; now that `h` is
fp16 and the redundant read has halved, it is worth well under 1% and sits inside
the noise. Kept because it is strictly less work, not because it is measurable.

The lesson is the one the other projects here keep producing. This block reads
56 MB and runs at ~85 GB/s; it is limited by bytes, so only removing bytes helps.

Unrolling the weight loads, which was worth +4% in
[`../int4-gemv`](../int4-gemv), **loses** here — 0.716 ms at unroll 1 against
0.769 at unroll 4. K is 2880, which is 90 `uint4` per row and under three loop
iterations per lane, so the unrolled staging array spills more than the extra
loads in flight buy. Same lever, opposite sign, decided by the reduction length.

## Against MLX

Same weights, same router, bit-identical test data:

| | ms/block | GB/s | tok/s |
|---|---|---|---|
| this fused kernel | **0.660** | 84.9 | 42.1 |
| MLX, `take` then `gather_qmm` | 2.265 | 24.7 | 12.3 |
| MLX, plain `mx.gather_qmm` | 6.928 | 8.1 | 4.0 |

**3.4x faster than the best MLX formulation I could find**, 10.5x against the
natural one. Both sides run fp16 activations; MLX's time is unchanged by that,
so it is not activation-bound. But most of that gap is not fusion, and saying so matters:

**`mx.gather_qmm` in MLX 0.32 scales with the total expert count, not with
top-k.** One projection gathering 4 experts, measured:

```
  8 experts   0.69 ms
 32 experts   1.05 ms
128 experts   2.53 ms
```

16.6 MB is needed in every case. Slicing the four experts out with `mx.take`
first — paying an explicit 16.6 MB copy — is 2.8x faster than letting
`gather_qmm` do the gather, which is why the `take` row above is MLX's best
showing and the one the 3.2x is quoted against. Quoting the 9.8x instead would
be reporting an MLX indexing behaviour as if it were this kernel's merit.

**MLX's own launch count is not measured here.** `mx.metal.start_capture` writes
a `.gputrace` whose store is compressed and needs Xcode tooling to read, and no
Xcode is installed on this machine. The 28 and 3 in the table above are counted
by this harness for its own two paths — it increments a counter on every
`dispatchThreads` call — and no launch count is claimed for MLX at all.

## Correctness

Every path is checked against an fp64 CPU reference that recomputes the whole
block — router logits, top-k selection, both quantised matmuls, SwiGLU, and the
weighted combine — before anything is timed, and the top-k indices are compared
exactly, not just the output:

```
  fused (3 kernels)      top-k match   max|d|/|y|inf = 2.15e-05  ok
  fused, atomic sum(h)   top-k match   max|d|/|y|inf = 4.60e-05  ok
  unfused (27 kernels)   top-k match   max|d|/|y|inf = 2.15e-05  ok
```

(2e-05 rather than the earlier 5e-08 because the activations are fp16 now; the
weights and accumulation are unchanged.) And independently against MLX on the
same inputs, both at fp16 activations: **2.23e-07**.

Two things this caught. The first router weights were generated with `% 2000 -
1000` in unsigned arithmetic, which wraps to ~4e9 and produced logits around
9e13 — large enough that fp32 could not resolve the 3rd and 4th ranked experts,
so the GPU top-k disagreed with the fp64 reference. The kernels were right and
the test data was wrong, which is exactly the failure an output-only check would
have hidden, since the mis-ranked experts had softmax weights of ~0.

The third: `h` stayed declared `device float*` in one of the two pass-2 variants
after the switch to fp16, so it wrote 4-byte elements into a 2-byte buffer. The
giveaway was that the accumulator it fed was *correct* while `h` itself came back
with impossible +-512 spikes in a tensor a clamped SwiGLU bounds at ~56 — a
value-range sanity check found it faster than any amount of staring at indices.

The second: the first router kernel gave each expert a single thread walking
2880 scalars, leaving one threadgroup to pull the entire 1.47 MB router at one
core's share of bandwidth. Fixing it to one simdgroup per expert with `float4`
loads and a `simd_sum` — **in both paths**, since a slow router in the baseline
would have flattered the fused one — moved the fused-vs-unfused ratio from an
apparent 1.11x down to its real 1.03x.

## Build and run

```sh
make
./moefused --iters 25 --warmup 6
python3 bench/mlx_moe.py 128 take      # MLX comparison (also: gather)
```

Flags: `--store --experts --iters --warmup --repeat --layers`.
Needs `~/Desktop/moe-stream/model-120b/experts.bin`. Captured runs in `results/`.
