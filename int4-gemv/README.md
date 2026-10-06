# int4-gemv

A Metal int4 matrix-vector kernel for LLM decode. Decode GEMV reads every
weight exactly once, so its speed is limited by memory bandwidth, not
compute. Because of that, results here are given as a percentage of measured
memory bandwidth instead of FLOP/s.

Apple M4 (10 GPU cores, LPDDR5X-7500, 128-bit bus, 120 GB/s theoretical).
Shape `M=32768, K=16384`, 4-bit, group size 64 (256 MB of weights). The
ranges are across three runs:

| | GB/s | % of 120 GB/s |
|---|---|---|
| plain streaming read (no math) | 111.3–113.5 | 92.7–94.6% |
| this kernel (`v6`) | 111.9–113.4 | 93.2–94.5% |
| `mlx.quantized_matmul`, same protocol | 102.6–104.7 | 85.5–87.2% |

The kernel runs as fast as a plain read of the same bytes (98.6–101.0% of the
measured read ceiling), so dequantization costs nothing extra. The last ~6%
of the theoretical bandwidth isn't reachable from software at all (refresh
and memory-controller overhead). That's why the measured read speed is used
as the ceiling instead of the spec number.

Other sizes: 93.5% of peak at 32 MB, 93.3% at 128 MB, 93.7% at 256 MB, and
87.7% at 8 MB. At 8 MB a single call takes only about 80 µs, so dispatch
overhead starts to show.

Compared with MLX, this kernel is 5–10% faster, with dispatch overhead
spread over about 10 calls on both sides (110.0–112.4 GB/s vs. 102.6–104.7).
That comparison is between two separate processes, which drift by about 8%
on this machine, so it isn't as clean as the comparisons between variants
below.

## Measurement

Four things made early results look better than they really were. Each one
produced a number above 100% of the bus, which is physically impossible:

1. **The cache.** At 32 MB, the M4's 8 MB system-level cache was absorbing
   some of the re-reads, so the kernel showed 109% of the bus. Now the
   benchmark rotates between ~200 MB of identical weight copies, so no call
   re-reads data the previous call left in cache. That also matches real
   decode, where each layer reads a different matrix.
2. **Timer resolution.** Very short kernels (about 70 µs) over-report
   throughput from GPU timestamps. Now each timed span batches enough
   dispatches to take about 5 ms. With rotation plus batching, the 8 MB
   shape reads 94.5% of the ceiling instead of 114%.
3. **GPU clock ramp-up.** A cold GPU reads at about 75 GB/s and a warm one
   at about 113 GB/s. The benchmark keeps the GPU under load until the rate
   stops climbing, and only then starts measuring.
4. **Drift between processes.** Results vary by about 8% from one process
   to the next. So every variant, including the ceiling, is timed
   round-robin inside a single process. This changed two conclusions: `v4`
   and `tg=512` had looked like wins when tested one after another, but
   they're within noise when interleaved.

`bench/compare_mlx.py` uses the same rotation and batching for MLX.

## The kernel

The weight layout matches `mlx.quantize(..., bits=4)`: `W` is `[M,K]`
row-major with 8 nibbles per uint32, plus one fp16 scale and one fp16 bias
per group of 64. That works out to 0.5625 bytes per parameter.

The affine form can be factored:

```
sum_k (s*q_k + b) * x_k  =  s * (sum_k q_k*x_k)  +  b * (sum_k x_k)
```

`sum(x)` doesn't depend on the row, so a small prepass (`xsum32`) computes it
once per token.

| | change | GB/s |
|---|---|---|
| `v0` | scalar, one thread per row | 80.9 |
| `v1` | one simdgroup per row, 16-byte loads, `simd_sum` | 106.2–107.1 |
| `v2` | one simdgroup per R rows | 104.3–106.6 |
| `v3` | `v2` + x staged in threadgroup memory | 100.0–104.6 |
| `v4` | scale and bias interleaved as `half2` | 106.0–106.9 |
| `v5` | `v4` + x read as `uint4`, xsum in threadgroup memory | 107.3 |
| `v6` | `v5` + weight loads unrolled 4x | 111.9–113.4 |

`v1` through `v4` all land at roughly the same speed. Each simdgroup already
reads a contiguous 512-byte span per step, which keeps DRAM busy, so the
extra rows in `v2` and the threadgroup staging in `v3` only cost registers
and occupancy. `v4` cuts out real work and still doesn't move, because at
that point the kernel is waiting on memory, not on instructions.

`v6` helps by reducing instruction issue:

- x is read as four `uint4` loads per 32 weights instead of eight `half4`
  loads. It's the same bytes in half as many loads.
- `xs[]` is staged once per threadgroup, instead of being re-read from
  device memory by every row.
- Weight loads are unrolled 4x, so four independent 16-byte loads are in
  flight before any of them is used. This alone is worth about 4%. Unrolling
  by 2 or 8 is slower.

## Correctness

Every variant is checked against two independent implementations before
any timing is reported:

- a CPU reference (fp64 accumulation, multithreaded), which is itself
  checked against NumPy
- `mlx.quantized_matmul` (`bench/compare_mlx.py`)

All variants match to within 2.4e-4 relative to `‖y‖∞`, which is ordinary
fp16 rounding.

The cross-check caught a real bug. The first vectorized version added the
bias term twice per group: it processed weights in 32-element chunks, but
the bias applies per 64-element group. The scalar version looped per group
and was correct, so only comparing against a differently structured
implementation exposed the bug. The fix keeps x partial sums per 32-element
chunk, so the bias is added exactly once.

## Build and run

This only needs the Command Line Tools, not Xcode. The `.metal` source is
compiled at runtime with `newLibraryWithSource:`.

```sh
make
./int4gemv --M 32768 --K 16384 --iters 25 --warmup 6     # headline run
./int4gemv --M 16384 --K 16384 --dump results            # dump tensors, then
python3 bench/compare_mlx.py                             # cross-check against MLX
```

Flags: `--M --K --G --tg --iters --warmup --repeat --src --dump`. `--repeat`
is set automatically so each timed span is about 5 ms. The kernel has been
tested with group sizes 32, 64, and 128, and with `M` values that don't
divide evenly into the threadgroup row count.

The saved runs are in `results/`. `results/mlx_compare.txt` is currently a
failed run (it was missing `shape.txt`), so the MLX numbers above aren't
backed by a saved file yet.
