# hetero-gemv

One int4 decode GEMV split between the GPU and the CPU at the same time. The
GPU computes rows `[0, split)` and the CPU computes rows `[split, M)`. Because
memory is unified, the CPU reads `buffer.contents`, the same physical memory
the GPU is reading, so nothing has to be copied.

Both processors share the same 120 GB/s memory bus, so splitting a
bandwidth-bound kernel can't add bandwidth. It only helps if the second
processor is held back by something other than the bus. For int4, the CPU's
bottleneck is unpacking nibbles, which is compute rather than memory.

## Result

`M=16384, K=16384`, int4 with group size 64 (128 MB of weights, 144 MB read
per GEMV). Wall-clock times, all from the same process with the same
batching:

| | ms | GB/s | % of 120 GB/s |
|---|---|---|---|
| CPU only (NEON fp16, 10 threads) | 2.57–2.60 | 58.2–58.9 | 48.5–49.1 |
| GPU only (`v6` from [int4-gemv](../int4-gemv/)) | 1.37–1.38 | 109.6–110.6 | 91.3–92.2 |
| GPU + CPU, best split (0.55–0.60) | 1.28–1.30 | 116.3–117.7 | 96.9–98.1 |
| `mlx.quantized_matmul` (GPU only) | ~1.43 | 103.9–105.0 | 86.5–87.5 |

The split is 1.046–1.072x faster than the GPU alone and 11–13% faster than
MLX. (It's 1.046x at `M=32768`, 256 MB, and 1.057–1.072x at 128 MB.)

Against an older, slower GPU kernel, the same split measured 1.10–1.11x. About
half of that gain was really just headroom the GPU kernel wasn't using. The
numbers above use the best GPU kernel I have.

Even a pure streaming read on the GPU alone tops out at 92.7–94.6% of
peak. With the CPU reading at the same time, the bus reaches 97–98%. One
processor can't keep enough requests in flight to saturate this memory
system, but two can. At 98% the bus is essentially full, so further gains
have to come from reading fewer bytes.

For Qwen2.5-0.5B's quantized projections (201.3 MB per token), that bandwidth
works out to 585 tok/s for the split, 549 for GPU only, and 520 for MLX. These
are calculated from bandwidth, not measured end to end.

## Choosing the split

If each processor's rows-per-second rate is measured on its own, the split
where both finish at the same time comes out to 0.651–0.655. The best split
actually measured is 0.55–0.60. Running both at once slows the GPU down more
than the CPU, because the GPU was already close to the bus limit. A split
based on solo throughput gets close to the best one, but finding the true
optimum takes a sweep. A fine-grained sweep only takes a few seconds.

## CPU paths

| CPU path | GB/s | best split | speedup vs GPU only |
|---|---|---|---|
| NEON fp16, dequantize in registers | 58.2–58.9 | 0.55–0.60 | 1.057–1.069x |
| NEON fp32, dequantize in registers | 43.1 | 0.70 | 1.079x |
| Accelerate `cblas_sgemv` | 17.3 | 1.000 | 1.000x |

**Accelerate.** On the same operand, `src/amx_probe.c` measured `sgemv` at
46.6 GFLOP/s and `sgemm` (n=64) at 877.2 GFLOP/s. That 19x gap means
Accelerate's `sgemv` isn't using the matrix coprocessor. That makes sense:
GEMV uses each weight only once, so there's nothing for a matrix unit to
reuse. On top of that, Accelerate can't read int4, so each block has to be
dequantized into an fp32 scratch buffer first, which is 8x the bytes. With
this path, the best split gives the CPU zero rows. The `--cpu amx` flag
selects it, but despite the name it doesn't actually use AMX.

**fp16 vs. fp32.** Keeping the inner loop in fp16 makes the CPU path 35%
faster (43.1 → 58.1 GB/s). Each lane only accumulates four products of at
most 15 × 1, so it never goes above 60, and fp16 is precise enough in that
range. fp16 fits eight values per register instead of four, and converting
`uint16 → fp16` is a single instruction. Each chunk's fp16 partial sum is
added into an fp32 accumulator, so the full row sum is never accumulated in
half precision.

## Measurement issues I hit

- **Equal chunks per thread.** The M4 has 4 performance cores and 6
  efficiency cores. With equal chunks, the P-cores finished early and sat
  idle while the E-cores were still working. Splitting the work into about 6
  chunks per core and letting GCD hand them out took the CPU path from 35.6
  to 42.3 GB/s.
- **Command-buffer submission.** Each submission costs 0.25–0.8 ms, which is
  longer than the GEMV itself. Timed naively, the GPU-only run showed
  1.145 ms for a 0.36 ms kernel, and the split looked like a 1.31x win. Now
  the GPU dispatches are batched into one command buffer while the CPU runs
  the same number of passes.
- **Caches.** The CPU's share of the rows fits in its 4 MB L2. Without the
  cache-defeat rotation from int4-gemv, the speedup measured 1.31x. With
  rotation, it was 1.08x.

## Correctness

Before any timing, splits 0.00, 0.50, 0.85, and 1.00 are each checked
against a multithreaded fp64 reference, so CPU-only, GPU-only, and mixed
outputs are all verified. If anything doesn't match, the program exits.
The results agree to within 2.1e-4 relative to `‖y‖∞` whenever the GPU
computes any of the rows, and to within 2.4e-7 to 2.6e-4 for CPU-only
(depending on whether the fp32 or fp16 loop is used).

## Build and run

```sh
make
./hetero --M 16384 --K 16384 --steps 21 --cpu neon16 --repeat 12
./hetero --cpu amx --block 16          # the Accelerate path, for comparison
```

Flags: `--M --K --iters --warmup --tg --block --steps --repeat --cpu
{neon16,neon,amx} --src`. Shaders are compiled at launch. Saved sweeps are in
`results/`.

## Files

```
src/cpu_gemv.h      the three CPU paths
src/kernels.metal   the GPU int4 kernel (same as int4-gemv)
src/main.mm         split sweep, concurrent launch, verification, split prediction
src/amx_probe.c     sgemv vs sgemm comparison
```
