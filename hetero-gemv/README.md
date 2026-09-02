# hetero-gemv — one matmul, Metal GPU and CPU vector units at the same time

The GPU takes rows `[0, split)` of an int4 decode GEMV and the CPU takes rows
`[split, M)`, concurrently, over unified memory. The CPU reads `buffer.contents`
— literally the same physical addresses the GPU is reading — so the split costs
no transfer, no staging, no copy of any kind.

What makes it a real question rather than free parallelism: **both engines share
one 120 GB/s memory system.** Splitting a bandwidth-bound kernel cannot add
bandwidth. It can only help if the second engine is limited by something other
than the bus — which for int4 the CPU is, because unpacking nibbles is
arithmetic and the CPU's share of the rows costs it far fewer bytes than the
GPU is already extracting.

## Result

`M=16384, K=16384`, int4 group 64, 0.5625 bytes/param (128 MB of weights,
144 MB moved per GEMV). All numbers wall-clock, same process, same batching:

| | ms | GB/s | % of 120 GB/s peak |
|---|---|---|---|
| CPU only (NEON fp16, 10 threads) | 2.57 – 2.60 | 58.2 – 58.9 | 48.5 – 49.1 |
| GPU only (`v6` from `../int4-gemv`) | 1.37 – 1.38 | 109.6 – 110.6 | 91.3 – 92.2 |
| **GPU + CPU, best split (0.55 – 0.60)** | **1.28 – 1.30** | **116.3 – 117.7** | **96.9 – 98.1** |
| `mlx.quantized_matmul` (GPU only) | 2.91 | 103.9 – 105.0 | 86.5 – 87.5 |

**Speedup over GPU-only: 1.046 – 1.072x**, and **+11% to +13% over MLX**
(1.046x at `M=32768, K=16384` / 256 MB, 1.057 – 1.072x at 128 MB).

That speedup is *smaller* than it used to be, on purpose. The GPU half now runs
the `v6` kernel from [`../int4-gemv`](../int4-gemv) — wide x loads, staged xsum,
4x-unrolled weight loads — which took GPU-only from 104.8 to 110.6 GB/s. A
heterogeneous split has to be measured against the best GPU kernel available, or
the CPU gets credited for headroom the GPU simply was not using. Against the
older fused-half2 kernel the same split reported 1.100 – 1.111x; roughly half of
that was the GPU leaving performance on the table.

The interesting part is not the 6%. It is *where* it lands: even a GPU kernel
running at the streaming-read ceiling tops out at ~92% of peak, and adding the
CPU pushes the same op to **~97 – 98% of peak** — past the 92.7 – 94.6% that a
*pure streaming read with no arithmetic at all* reaches on this machine
([`../int4-gemv`](../int4-gemv)). Two engines pulling on the bus together extract
more of it than one engine can, even one doing nothing but reading. A single
client cannot keep enough requests in flight to saturate this memory system; two
can.

That also bounds what is left. At 98% of 120 GB/s the bus is finished, so the win
is 6% and not 60%, and no further split, kernel or scheduling change moves it.
Only fewer bytes would.

Derived decode rate for Qwen2.5-0.5B's quantised projections (357.8M params at
0.5625 bytes/param = 201.3 MB/token): **585 tok/s combined, 549 GPU-only,
520 for `mlx.quantized_matmul`**.
Arithmetic on measured bandwidth and parameter count, not an end-to-end
generation measurement.

## Load balance: measured throughput is a starting point, not the answer

If the GPU clears rows at `r_g = M/t_gpu` and the CPU at `r_c = M/t_cpu`, the
split that finishes both at the same instant is `M · r_g/(r_g + r_c)`. On these
measurements that predicts **0.651 – 0.655**. The measured optimum is
**0.55 – 0.60**.

The prediction is biased high by about 9 points of split, and the reason is the
same one that makes the whole thing work: the two rates do not survive being run
at once. The GPU loses more throughput to contention than the CPU does, because
the GPU was the one already near the bus limit, so the true optimum sits further
toward the CPU than independent measurements suggest.

The practical reading: calibrating from solo throughput lands at 0.65, a few
points of split past the measured optimum. Close, and free, but the
last third of the win needs a real sweep — the curve is flat enough that a coarse
sweep will miss it and shallow enough that a fine one finds it in a few seconds.

## The CPU path: Accelerate never reaches the matrix units here

"Use the CPU's matrix/AMX units" is the obvious instruction, and on this workload
it is not merely slower — it does not happen at all. Three CPU paths, same rows,
same format:

| CPU path | GB/s | best split | speedup vs GPU-only |
|---|---|---|---|
| **NEON fp16, dequant in registers** | **58.2 – 58.9** | 0.55 – 0.60 | **1.057 – 1.069x** |
| NEON fp32, dequant in registers | 43.1 | 0.70 | 1.079x |
| Accelerate `cblas_sgemv` | 17.3 | **1.000** | **1.000x** |

Two separate things go wrong, and it is worth keeping them apart.

**Accelerate does not use the matrix coprocessor for `sgemv`.** There is no
userspace way to ask, so `src/amx_probe.c` compares it against `cblas_sgemm` on
the same operand:

```
sgemv  0.180 ms    46.6 GFLOP/s
sgemm  0.612 ms   877.2 GFLOP/s   (n=64)
```

19x. The matrix hardware is there and it is plainly not being used for the
matrix-vector case — which is the expected answer, because GEMV reads every
weight exactly once and a matrix engine has nothing to reuse. So this row is not
really "the AMX path"; it is Accelerate's SIMD `sgemv`, and calling it anything
else would be describing hardware that never ran.

**And the route to it costs extra bytes.** Accelerate cannot read int4, so a
block of rows has to be dequantised into an fp32 scratch first — 8x the bytes of
the int4 rows it came from — and writing plus re-reading that scratch is pure
overhead. Together these put it 3.4x behind simply unpacking nibbles into NEON
registers and FMA-ing them against `x`. The load balancer's verdict is
unambiguous: with this path the optimal split is **1.000**, i.e. *give the CPU
nothing*.

The honest generalisation is not "AMX is bad" but "batch-1 decode has no work for
a matrix engine". A matrix coprocessor needs reuse; a matrix-vector product has
none to give.

Staying in fp16 rather than fp32 for the inner loop is worth 35% on the CPU side
(43.1 → 58.1 GB/s). The weights are 4-bit and `x` is already fp16, so a lane
accumulates at most four products of 15 × 1 — |acc| ≤ 60, where fp16 still has
0.06 resolution. Eight lanes per register instead of four, and one `uint16→fp16`
convert instead of a widen-to-`uint32` plus a convert-to-fp32. Each chunk's fp16
partial folds into an fp32 accumulator, so the row sum never accumulates in half
precision.

## Two bugs that were costing real throughput

**Equal-size CPU chunks.** This machine has 4 performance and 6 efficiency cores.
Handing each thread one equal share makes the E-cores the critical path: the
P-cores finish and idle while the slowest E-core still holds a full block.
Over-decomposing into ~6 chunks per core and letting GCD hand the fast cores more
of them took the CPU path from 35.6 to 42.3 GB/s — a 19% gain from scheduling
alone, before any SIMD work.

**Command-buffer submission.** One submission costs ~0.25 – 0.8 ms wall, which is
larger than the entire GEMV being measured, and it is charged to the GPU side of
every split. Timed naively, GPU-only reads 1.145 ms for a kernel that takes 0.36 ms,
and the split sweep measures mostly overhead — it reported a fake 1.31x. The R
GPU dispatches now go into one command buffer while the CPU makes R passes, so
the overlap is preserved and submission is amortised the way it would be inside a
real decode loop.

The cache-defeat rotation from [`../int4-gemv`](../int4-gemv) matters more here
than there, because the CPU has its own 4 MB L2 per cluster and its slice of the
rows is small enough to sit in it entirely. Without rotation the measured
speedup was 1.31x; with it, 1.08x. The difference was the caches, not the engines.

## Correctness

Every split is checked against a multithreaded fp64 reference before anything is
timed, and a mismatch aborts rather than printing a number — splits 0.00, 0.50,
0.85 and 1.00, so the pure-CPU path, the pure-GPU path and two genuinely mixed
outputs are all verified. Agreement is 2.1e-04 relative to `‖y‖∞` for any split
containing GPU rows (fp16 dot rounding) and 2.4e-07 – 2.6e-04 for CPU-only,
depending on whether the fp32 or fp16 inner loop is in use.

## Build and run

```sh
make
./hetero --M 16384 --K 16384 --steps 21 --cpu neon16 --repeat 12
./hetero --cpu amx --block 16          # the matrix-unit path, for comparison
```

Flags: `--M --K --iters --warmup --tg --block --steps --repeat --cpu {neon16,neon,amx} --src`.
Shaders compile at launch via `MTLDevice newLibraryWithSource:`; no Xcode
toolchain is installed or needed. Captured sweeps are in `results/`.

## Files

- `src/cpu_gemv.h` — the three CPU paths (NEON fp16, NEON fp32, Accelerate/AMX)
- `src/kernels.metal` — the GPU int4 kernel, shared with `../int4-gemv`
- `src/main.mm` — split sweep, concurrent launch, verification, load-balance predictor
