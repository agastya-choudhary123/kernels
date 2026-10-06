# moe-fused

A gpt-oss-120b MoE decode block (router, top-k, expert gather, int4 matmuls,
SwiGLU, and the weighted combine) done in 3 kernel launches instead of 28.
The kernels read the top-k indices from device memory and compute the
expert weight addresses on the GPU. The host binds a single expert pool
buffer once and never rebinds it per expert.

The test setup keeps 128 real experts resident (1.80 GB, read from
moe-stream's `experts.bin`): hidden size 2880, top-4, int4 with group size 64,
and BF16 scales.

## Result

| path | launches | ms/block | GB/s | tok/s |
|---|---|---|---|---|
| unfused (one kernel per op) | 28 | 0.650–0.670 | 83.6–86.1 | 41.5–42.7 |
| fused | 3 | 0.663–0.670 | 83.6–84.4 | 41.5–41.9 |
| fused + atomic sum(h) | 3 | 0.660–0.670 | 83.6–84.9 | 41.5–42.1 |
| `mlx.gather_qmm`, fastest version I found | — | 2.265 | 24.7 | 12.3 |

Going from 28 launches to 3 made **no difference to speed**. Across three runs,
the fused and unfused paths both land at 0.650–0.670 ms, and which one wins
changes from run to run. Kernel launches within one command buffer are nearly
free on this hardware. The block reads 56 MB at about 85 GB/s, so it's limited
by bandwidth.

Both of my paths are about 3.4x faster than MLX, but most of that gap comes
from how MLX does the gather, not from fusion (see below).

`tok/s` assumes 36 MoE blocks per token with every expert already in memory.
That makes it a compute rate, not real decode speed. In real decode, the
experts would have to be streamed from the SSD first (see
[moe-pipeline](../moe-pipeline/)).

## Why 3 kernels and not 1

The block has two dependencies that span the whole grid, and Metal has no way
to synchronize threadgroups within a single dispatch:

```
1. router + softmax + top-k          ->  idx[4], wgt[4]
2. gather + gate + up + SwiGLU       ->  h[4][inter]   (needs idx)
3. gather + down + weighted combine  ->  y[hidden]     (needs all of h)
```

Getting down to one kernel would need a grid-wide barrier built from atomics,
which only works if every threadgroup is resident at the same time. Metal
doesn't guarantee that, and if it doesn't hold, the GPU hangs.

Each of the three kernels is fused internally. Gate and up are computed back
to back in registers, and SwiGLU is applied before anything is written, so the
per-expert gate and up vectors never touch device memory. Pass 3 accumulates
`wgt[e] * down_e(h_e)` directly, so the per-expert outputs never touch device
memory either.

## What actually helped

- **fp16 activations.** x and `h` were fp32, which meant 8 `float4` loads of
  activations per 16 bytes of weights. In fp16 it's 4 `uint4` loads. That took
  every path from 0.710 to 0.660 ms. The unfused baseline got the same change.
- **Atomic `sum(h)`.** Pass 3 needs `sum(h_e)` for each 32-element chunk.
  Previously, each of the 360 threadgroups recomputed it by reading all of
  `h`. Now pass 2 adds into a `device atomic_float` as it produces `h`, and
  pass 1 zeroes it. This was worth about 2.5% when `h` was fp32. With fp16 it's
  within noise, but I kept it because it does strictly less work.
- **Unrolling didn't help here.** Unrolling the weight loads 4x gained 4% in
  [int4-gemv](../int4-gemv/), but it's slower here: 0.716 ms with no unroll
  vs. 0.769 ms unrolled 4x. With K = 2880, each lane runs fewer than three
  loop iterations, so the unrolled staging array spills registers.

## Compared with MLX

Same weights, same router, same inputs, fp16 activations on both sides:

| | ms/block | GB/s | tok/s |
|---|---|---|---|
| this fused kernel | 0.660 | 84.9 | 42.1 |
| MLX, `mx.take` then `gather_qmm` | 2.265 | 24.7 | 12.3 |
| MLX, plain `mx.gather_qmm` | 6.928 | 8.1 | 4.0 |

That's 3.4x faster than the fastest MLX version, and 10.5x faster than the
straightforward one. Most of that gap comes from how MLX is used, not from
fusion. In MLX 0.32, `gather_qmm`'s cost grows with the *total* number of
experts, not with top-k. Here's one projection gathering 4 experts:

```
  8 experts   0.69 ms
 32 experts   1.05 ms
128 experts   2.53 ms
```

Every case needs the same 16.6 MB. Copying the four experts out with
`mx.take` first is 2.8x faster, which is why the 3.4x figure is measured
against that version.

I didn't measure MLX's launch count. Reading a `.gputrace` requires Xcode,
which isn't installed. The 28 and 3 above come from my harness counting its
own `dispatchThreads` calls.

## Correctness

Before anything is timed, every path is checked against an fp64 CPU
reference of the whole block (router, top-k, both matmuls, SwiGLU, and the
combine). The top-k indices have to match exactly, not just the outputs:

```
  fused (3 kernels)      top-k match   max|d|/|y|inf = 2.15e-05  ok
  fused, atomic sum(h)   top-k match   max|d|/|y|inf = 4.60e-05  ok
  unfused                top-k match   max|d|/|y|inf = 2.15e-05  ok
```

The fused output also matches MLX on the same inputs to within 2.23e-07.

Bugs this caught:

- **Bad test data.** The router weights were generated with `% 2000 - 1000`
  in unsigned arithmetic, which wraps around to about 4e9. The resulting logits
  were around 9e13, too large for fp32 to tell the 3rd and 4th experts
  apart, so the GPU's top-k didn't match the reference. The kernels were
  right and the test data was wrong. Because those experts had near-zero
  softmax weight, a check that only compared outputs would have missed it.
- **Wrong type after the fp16 switch.** One of the two pass-2 variants still
  declared `h` as `device float*`, so it wrote 4-byte values into a buffer of
  2-byte values. The accumulator was still correct, but `h` showed ±512 spikes,
  even though a clamped SwiGLU can't go above about 56. A range check on the
  values caught it.
- **Slow router.** The first router kernel gave each expert one thread
  looping over 2880 scalars. After switching to one simdgroup per expert
  with `float4` loads and `simd_sum` (in both paths), the fused path's
  apparent 1.11x advantage over unfused shrank to about 1.0x.

## Build and run

```sh
make
./moefused --iters 25 --warmup 6
python3 bench/mlx_moe.py 128 take      # MLX comparison (or: gather)
```

Flags: `--store --experts --iters --warmup --repeat --layers`. You'll need
moe-stream's `model-120b/experts.bin`. By default it's read from
`~/Desktop/moe-stream/model-120b/experts.bin`; use `--store` to point
somewhere else. Saved runs are in `results/`.
