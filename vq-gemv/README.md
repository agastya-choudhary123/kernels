# vq-gemv

A fused codebook (vector quantization) dequant-matmul kernel in Metal,
compared against affine int4 at exactly the same size.

Scalar int4 gives each weight its own 4-bit index on a uniform grid. This VQ
format gives each *pair* of weights one 8-bit index into a learned codebook
of 256 two-element vectors, so the codewords are placed where the weights
actually are. The two formats are the same size:

```
affine int4   : 4 bits/weight + fp16 scale + fp16 bias per 64  = 4.500 bits/weight
VQ d=2 K=256  : 4 bits/weight + fp16 scale         per 32      = 4.500 bits/weight
```

Both are 0.5625 bytes per parameter. VQ also needs a 1 KB codebook per
matrix, which makes it 4.504 bits per weight.

## Quality

Qwen2.5-0.5B-Instruct with all 168 transformer-block projections quantized
(357.8M of its 494.0M parameters). Perplexity is measured on the full
WikiText-2 test split (299,078 tokens, 146 windows of 2048 tokens):

| scheme | bits/weight | ppl | Δ vs fp16 |
|---|---|---|---|
| fp16 | 16.000 | 14.2499 | — |
| affine int4, g=64 | 4.500 | 16.2559 | +2.0060 |
| codebook VQ, d=2, K=256 | 4.504 | 15.4926 | +1.2427 |

At the same size, VQ's perplexity is 4.70% lower than int4's. That recovers
about 38% of the quality lost to 4-bit quantization. To reproduce, run
`python3 bench/perplexity.py`.

## Speed

`M=16384, K=16384`. Every variant was timed in the same process, round-robin,
using the same method as [int4-gemv](../int4-gemv/):

| variant | GB/s | % of 120 GB/s | % of read ceiling |
|---|---|---|---|
| plain streaming read | 111.1–111.7 | 92.6–93.1 | 100 |
| VQ, wide loads + unroll 4 | 111.2–111.6 | 92.7–93.0 | 99.8–100.3 |
| int4 affine, same size, same tuning | 111.7–112.5 | 93.1–93.7 | 100.0–101.2 |
| `mlx.quantized_matmul`, same size | 103.9–105.0 | 86.5–87.5 | — |
| VQ, shared codebook, untuned | 108.0 | 90.0 | 97.0 |
| VQ scalar (1 thread/row) | 88.3 | 73.6 | 79.3 |

The VQ and int4 kernels both run at the plain-read ceiling, and both are
about 6% faster than MLX's int4 at the same size. MLX doesn't have a VQ
matmul, so GB/s at equal size is the only fair comparison. Between VQ and
int4, the ratio is 0.99–1.00x, which is within run-to-run noise. In other
words, the codebook lookups cost nothing measurable.

The speedup from 108.0 to 111.5 GB/s came from the same two changes as in
int4-gemv. x is read as four `uint4` loads instead of sixteen `half2` loads,
and the index loads are unrolled 4x. The int4 kernel I compared against got
the same tuning, so the comparison is fair.

At 0.5625 bytes per parameter, Qwen2.5-0.5B's quantized projections are
201.3 MB per token. At 107.6 GB/s that works out to about 535 tok/s, compared
with 529 for int4. This is calculated from bandwidth, not measured end to end.

## Fixing bank conflicts made it slower

Threadgroup memory has 32 banks of 4 bytes, and each `half2` codeword is
exactly one 4-byte word. The 32 lanes of a simdgroup look up 32 unrelated
codewords, so some of them collide on the same bank and get serialized.

The textbook fix is to store `REP` interleaved copies of the codebook so each
lane reads its own copy. With `REP=32`, lane `l` always hits bank `l`. That
takes 32 KB, which is the per-threadgroup maximum, so the kernel uses
persistent threadgroups to stage it only once. It got slower as `REP`
increased:

| codebook copies | threadgroup memory | GB/s |
|---|---|---|
| 1 (persistent) | 1 KB | 106.8 |
| 4 | 4 KB | 105.9 |
| 8 | 8 KB | 101.5 |
| 32 (no conflicts) | 32 KB | 85.4 |

With `REP=1`, the persistent kernel performs about the same as the regular one
(106.8 vs. 107.9), so all of the slowdown comes from the extra threadgroup
memory reducing occupancy. Keeping the codebook in device memory instead also
performs about the same (106.8 vs. 107.9). The bank conflicts were real, but
they didn't cost anything, because a memory-bound kernel already has idle
cycles to absorb them.

## Correctness

- Every variant is checked against a multithreaded fp64 CPU reference before
  it's timed. A mismatch stops the benchmark. All variants agree to within
  about 3e-4 relative to `‖y‖∞`.
- `bench/verify_real.py` quantizes a real Qwen2.5-0.5B matrix
  (`layers.12.mlp.down_proj`, 896×4864), runs it through the kernel, and
  compares the result with MLX's dequantize-then-matmul. They agree to
  1.99e-4. On that matrix, the reconstruction error is 0.08301 for VQ and
  0.09716 for int4.

The perplexity numbers come from dequantizing to fp16 and running the normal
forward pass. That only measures the kernel if the kernel computes the same
reconstruction, and `verify_real` checks that it does.

## Build and run

```sh
make                             # builds vqgemv and verify_real
./vqgemv --M 16384 --K 16384 --iters 15 --warmup 5
python3 bench/compare_mlx.py     # MLX int4 at the same size
python3 bench/perplexity.py      # about 5 minutes
python3 bench/verify_real.py
```

Flags: `--M --K --tg --iters --warmup --repeat --src --dump`. Shaders are
compiled at launch, so you don't need Xcode. Saved runs are in `results/`.

## Files

```
src/kernels.metal       VQ variants, the int4 comparison kernel, and the read ceiling
src/main.mm             benchmark harness
src/verify_real.mm      runs the kernel on a real weight matrix
quant/pq_quantize.py    the quantizer (k-means in MLX)
bench/                  perplexity, MLX comparison, real-weight check
```
