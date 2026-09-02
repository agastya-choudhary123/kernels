# vq-gemv — fused codebook (vector-quant) dequant-matmul

Scalar int4 spends one 4-bit index per weight on a uniform lattice. Vector
quantisation spends one 8-bit index per *pair* of weights on a learned 256-entry
codebook of 2-vectors, so the codewords sit where the weights actually are.

The two formats are deliberately the same size:

```
affine int4   : 4 bits/weight + fp16 scale + fp16 bias per 64  = 4.500 bits/weight
VQ d=2 K=256  : 4 bits/weight + fp16 scale         per 32      = 4.500 bits/weight
```

Both are 0.5625 bytes/param, so any difference is the quantiser, not the budget.
VQ additionally carries a 256 x half2 = 1 KB codebook per matrix, which is the
`4.504` vs `4.500` in the table below — 0.09% more bits.

## Quality — real perplexity, real model

Qwen2.5-0.5B-Instruct, all 168 transformer-block projections quantised
(357.8M of 494.0M params), scored on the **entire** WikiText-2 test split
(299,078 tokens, 146 non-overlapping windows of 2048):

| scheme | bits/weight | ppl | Δ vs fp16 |
|---|---|---|---|
| fp16 (unquantised) | 16.000 | 14.2499 | — |
| affine int4 g=64 | 4.500 | 16.2559 | +2.0060 |
| **codebook VQ d=2 K=256** | 4.504 | **15.4926** | **+1.2427** |

**At equal bytes/param, VQ is 4.70% better perplexity** — it gives back 38% of
the quality that 4-bit quantisation costs, for 0.09% more bits. Reproduce with
`python3 bench/perplexity.py`.

## Speed — the codebook lookup is free

`M=16384, K=16384`, both formats timed in the same process, round-robin, under
the same clock settle / cache-defeat rotation / dispatch batching:

| variant | GB/s | % of 120 GB/s peak | % of measured ceiling |
|---|---|---|---|
| pure streaming read (ceiling) | 111.1 – 111.7 | 92.6 – 93.1 | 100 |
| **VQ, wide loads + unroll 4** | **111.2 – 111.6** | **92.7 – 93.0** | 99.8 – 100.3 |
| int4 affine, same bytes/param, same tuning | 111.7 – 112.5 | 93.1 – 93.7 | 100.0 – 101.2 |
| `mlx.quantized_matmul`, same bytes/param | 103.9 – 105.0 | 86.5 – 87.5 | — |
| VQ, shared codebook, untuned | 108.0 | 90.0 | 97.0 |
| VQ scalar (1 thread/row) | 88.3 | 73.6 | 79.3 |

**Both kernels sit on the streaming-read ceiling**, and both are **~6% faster
than `mlx.quantized_matmul`** at identical bytes/param (111.2 – 111.6 against
103.9 – 105.0). MLX has no vector-quantised matmul at all, so GB/s at equal
bytes/param is the only apples-to-apples comparison available; the arithmetic
differs, the bytes do not.

**VQ / int4 throughput: 0.99 – 1.00x** across runs with both kernels tuned
identically. The claim this supports is **parity**: whichever is ahead in a given
run is ahead by less than the run-to-run spread.

Getting here needed the same two levers as [`../int4-gemv`](../int4-gemv). The
first VQ kernel read x as sixteen `half2` loads per uint4 of indices — 4-byte
transactions, one per codeword. Reading it as four `uint4` and bit-casting gives
identical bytes in a quarter of the instructions, and unrolling the index loads
4-deep keeps four 16-byte fetches in flight. That took VQ from 108.0 to 111.5
GB/s. The int4 opponent got exactly the same treatment, because pitting a tuned
VQ kernel against an untuned baseline would have manufactured a VQ win.

Parity is the result worth having, and it is a stronger statement now than it was
at 108 GB/s: sixteen codebook lookups per 16 bytes streamed cost nothing
measurable even when the kernel is running at 100% of the achievable read
bandwidth and has no stall left to hide them in.

So the combined claim is: **VQ costs the same time as scalar int4, beats MLX's
int4 by ~6%, and is 4.70% better perplexity — all at the same bytes/param.**

Implied decode rate for Qwen2.5-0.5B's quantised projections — 357.8M params at
0.5625 bytes/param is 201.3 MB streamed per token, so 107.6 GB/s is **535 tok/s**
against int4's 529. This is arithmetic on the measured bandwidth and the model's
parameter count, not an end-to-end generation measurement.

## The negative result: bank conflicts cost nothing

Apple's threadgroup memory is 32 banks of 4 bytes, and a `half2` codeword is
exactly one 4-byte word, so lane `l` reading codeword `c` hits bank `c % 32`. The
32 lanes of a simdgroup read 32 unrelated codewords, which collide like balls in
bins — expected worst bank depth about 4, serialised by the hardware. That looks
like the obvious thing to fix.

The fix is clean: store `REP` copies of the codebook interleaved, copy `l` of
codeword `c` at word `c*REP + l`. At `REP=32` lane `l` always lands on bank
`(c*32 + l) % 32 = l`, whatever `c` is — conflict-free by construction rather
than by luck. It costs 256 x 32 x 4 = 32768 bytes, exactly the M4's
per-threadgroup maximum, so the kernel uses persistent threadgroups and
grid-strides over rows to pay that staging once per threadgroup instead of once
per 8 rows.

It loses, monotonically, and the sweep says exactly why:

| codebook copies | threadgroup memory | GB/s |
|---|---|---|
| 1 (persistent) | 1 KB | 106.8 |
| 4 | 4 KB | 105.9 |
| 8 | 8 KB | 101.5 |
| 32 (conflict-free) | 32 KB | 85.4 |

The `REP=1` persistent row ties the non-persistent `v1` (106.8 vs 107.9), so
running persistently is neutral and the whole decline is the threadgroup memory
itself crowding out occupancy. Two other measurements say the same thing: leaving
the codebook in **device memory** ties the staged version (106.8 vs 107.9), and
the scalar kernel — which does identical lookups with no simdgroup parallelism —
is slow for entirely unrelated reasons. The bank conflicts were real and were
never costing anything, because a memory-bound kernel has stall cycles to spend.

Optimising the arithmetic of a kernel running at 97% of its achievable bandwidth
cannot help, and buying that optimisation with occupancy actively hurts. That is
worth more than a speedup would have been.

## Correctness

Nothing is reported for a kernel that has not been verified.

- Every variant is checked against a multithreaded fp64 CPU reference in-process
  before any timing runs; a mismatch aborts the benchmark rather than printing a
  number. All agree to ~3e-04 relative to `‖y‖∞`, which is fp16 dot rounding.
- `bench/verify_real.py` closes the loop between the speed and quality halves: it
  quantises a **real** Qwen2.5-0.5B weight matrix (`layers.12.mlp.down_proj`,
  896x4864), runs `gemv_vq_shared` on it through `./verify_real`, and compares
  against MLX's independent dequantise-then-matmul. Agreement `1.99e-04`. On that
  real tensor the reconstruction error is VQ 0.08301 vs int4 0.09716.

That check matters because the perplexity numbers come from dequantising to fp16
and running the stock forward pass — the right way to measure a *quantiser*,
since it isolates representation error from kernel arithmetic, but only a valid
stand-in if the kernel computes the same reconstruction. It does.

## Build and run

```sh
make                        # vqgemv (bandwidth) and verify_real
./vqgemv --M 16384 --K 16384 --iters 15 --warmup 5
python3 bench/compare_mlx.py     # MLX int4 at the same bytes/param
python3 bench/perplexity.py     # ~5 min: fp16 / int4 / VQ over all of wikitext-2
python3 bench/verify_real.py    # kernel vs MLX on a real Qwen matrix
```

Flags: `--M --K --tg --iters --warmup --repeat --src --dump`. Shaders compile at
launch through `MTLDevice newLibraryWithSource:`; there is no Xcode toolchain on
this machine and none is needed. Captured runs are in `results/`.

## Files

- `src/kernels.metal` — VQ variants (scalar, shared codebook, device codebook,
  lane-replicated persistent) plus the int4 opponent and the streaming ceiling
- `src/main.mm` — benchmark harness; `src/verify_real.mm` — real-weight runner
- `quant/pq_quantize.py` — the quantiser, k-means in MLX, matching the kernel's
  format exactly
- `bench/perplexity.py`, `bench/verify_real.py`
