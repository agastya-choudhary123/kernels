# int4-gemv — bandwidth-saturating int4 GEMV for decode

LLM decode is matrix-vector, and matrix-vector is not a compute problem. Every
weight is read once and used once, so the only thing that matters is how much
of the machine's memory bandwidth the kernel can actually keep busy. The metric
here is therefore **percent of peak DRAM bandwidth**, not FLOP/s.

Measured on this machine (Apple M4 base, 10 GPU cores, 16 GB LPDDR5X-7500,
128-bit bus → **120.0 GB/s** theoretical peak):

| | GB/s | % of 120 GB/s peak |
|---|---|---|
| pure streaming read (empirical ceiling) | 111.3 – 113.5 | 92.7 – 94.6 % |
| **this int4 GEMV** (`v6`) | **111.9 – 113.4** | **93.2 – 94.5 %** |
| `mlx.quantized_matmul`, same protocol | 102.6 – 104.7 | 85.5 – 87.2 % |

Shape `M=32768, K=16384`, group 64, 4-bit — 256 MB of weights, 288 MB moved per
call. Ranges are across three separate runs. The result holds across shapes:
93.5% of peak at 32 MB, 93.3% at 128 MB, 93.7% at 256 MB, and 87.7% at 8 MB
where a single GEMV is only ~80 us and dispatch starts to show.

**The honest reading of that table:** the kernel reaches ~94% of the M4's peak
bandwidth, which is **the same rate a pure read with no arithmetic at all
achieves** — `v6` lands at 98.6 – 101.0% of the measured streaming-read ceiling
across runs. The dequantisation is now completely free: reading these weights and
multiplying by them costs what reading them costs. The remaining ~6% of nominal
peak is not available to software at all — it is refresh, turnaround and
controller overhead — which is exactly why the measured ceiling, not the spec
sheet, is the denominator that matters.

Against MLX's own int4 GEMV the margin is **+5% to +10%** — wall clock with
dispatch amortised over ~10 calls on both sides: 110.0 – 112.4 GB/s here against
102.6 – 104.7 GB/s for MLX, over three runs each. That margin is now comfortably
wider than the run-to-run spread, which it was not before `v6`.

One caveat that cuts against it: unlike the variant-to-variant numbers above,
this compares two *separate processes*, which is exactly the setup section 4
below warns produces drift-driven differences. It is the best available
comparison, not a clean one. (Batching shallowly also makes MLX look faster —
per-call command-buffer submission is ~0.25 ms — which is why the protocol has
to match on both sides.)

The point of the project is that both numbers are stated against a measured
ceiling, under a protocol identical on both sides, instead of being quoted as a
bare percentage of a spec sheet.

## Why the ceiling is measured, not assumed

Almost nobody reports the denominator. Four things here would each have
produced a flattering but wrong number, and every one of them showed up as a
measured result above 100% of a bus that physically cannot exceed 120 GB/s:

1. **A working set that fits in cache.** At `M=K=8192` (32 MB) the *same*
   kernel appeared to hit 130 GB/s — 109% of the bus. The M4's 8 MB system
   level cache was absorbing part of every re-read. Fixed properly rather than
   by only benchmarking huge matrices: the harness allocates ~200 MB of
   *identical* weight copies and rotates to the next copy on every dispatch, so
   no call ever re-reads what the previous one left behind. That is also the
   physically honest model of decode, where a real model streams a different
   weight matrix at every layer and no call finds its weights warm.
2. **Kernels short enough to outrun the timer.** A 4096x4096 GEMV runs in ~70 us
   and the command buffer's GPU timestamps systematically over-report throughput
   at that scale — the 8 MB shape still read 114% of ceiling after rotation
   alone. The harness now packs enough dispatches into one command buffer that
   every timed span is ~5 ms, and divides. Rotation continues *between* those
   dispatches so the repeats do not become cache hits. With both fixes the 8 MB
   shape reports 94.5% of ceiling instead of 114%.
3. **GPU clock ramp.** A cold GPU reads ~75 GB/s and a warm one ~113 GB/s on a
   byte-identical kernel. The harness holds the memory system at sustained load
   until the achieved rate stops improving before it measures anything.
4. **Comparing variants across processes.** Process-to-process drift on this
   machine is ~8%, larger than the differences between kernels. Every variant —
   including the ceiling — is timed round-robin inside one process, one sample
   each per round, so all of them see the same thermal and clock state. This
   single change reversed two conclusions: `v4` and `tg=512` both looked like
   clear wins in sequential testing and are within noise when interleaved.

`bench/compare_mlx.py` reproduces the same rotation and batching for MLX, so the
comparison is not this kernel's amortised number against MLX's per-call number.

## The kernels

Layout is bit-compatible with `mlx.quantize(..., bits=4)`: `W` is `[M,K]`
row-major, 8 nibbles per uint32 (low nibble = lowest `k`), one fp16 scale and
one fp16 bias per group of 64 — **0.5625 bytes/param including metadata**.

The affine form factors, which is what makes int4 GEMV nearly free in ALU terms:

```
sum_k (s*q_k + b) * x_k  =  s * (sum_k q_k*x_k)  +  b * (sum_k x_k)
```

`sum(x)` over a chunk does not depend on the row, so a `K/32`-element prepass
(`xsum32`) computes it once per token and the inner loop never touches it again.

| | idea | GB/s |
|---|---|---|
| `v0` | scalar, one thread per row | 80.9 |
| `v1` | **one simdgroup per row, 16-byte (`uint4`) loads, `simd_sum` reduce** | **106.2 – 107.1** |
| `v2` | one simdgroup per R rows, R row-streams in flight | 104.3 – 106.6 |
| `v3` | `v2` + `x` staged in threadgroup memory | 100.0 – 104.6 |
| `v4` | scale+bias interleaved as `half2`: one metadata stream, not two | 106.0 – 106.9 |
| `v5` | `v4` + x read as `uint4` and bit-cast, xsum staged in threadgroup memory | 107.3 |
| `v6` | **`v5` + weight loads unrolled 4x** | **111.9 – 113.4** |

`v1` through `v4` are tied at the top of the *bandwidth-bound* regime and
everything clever loses there. `v6` is what finally moves it, and it does so by
attacking issue slots rather than bytes:

- x was read as eight `half4` loads per 32 weights — 64 bytes of x per 16 bytes
  of W, in 8-byte transactions. Reading it as four `uint4` and bit-casting to
  `half4` pairs halves that load count for identical bytes. x is small and
  cache-resident, so this is purely about instruction issue.
- `xs[]` is re-read by every row; staging it once per threadgroup turns a device
  load per iteration into a threadgroup-memory read.
- Unrolling the weight loads 4x issues four independent 16-byte loads before any
  is consumed. This is the same lever that took the pure-read ceiling kernel from
  latency-bound to bandwidth-bound, and it is worth +4% here on its own.

Unroll 4 is the peak; 2 and 8 both measure slower.

The reason this worked when the earlier "clever" variants did not: `v2`/`v3`
tried to buy latency tolerance with more rows or threadgroup staging, paying
registers and occupancy for it. `v6` buys the same tolerance with unrolling,
which costs neither. Each simdgroup
issues 32 lanes x 16 bytes = one contiguous 512-byte span per step, which is
already enough outstanding traffic to cover DRAM latency; adding rows (`v2`) or
threadgroup staging (`v3`) buys nothing and costs registers and occupancy. `v4`
removes real work — one 4-byte metadata load instead of two 2-byte loads from
distant arrays — and still lands within noise of `v1`, which is itself the
finding: at ~94% of the achievable ceiling the kernel is waiting on DRAM, not on
instructions, so removing instructions cannot help.

## Correctness

No number is printed for a kernel that has not been verified. Two independent
checks, both of which caught real bugs:

- **CPU reference** (`cpuRef`, fp64 accumulate, multithreaded), checked itself
  against numpy.
- **`mlx.quantized_matmul`**, a completely independent implementation of the
  same layout — `bench/compare_mlx.py`.

All variants agree with both to `2.4e-04` relative to `‖y‖∞`, which is fp16
dot-product rounding and nothing else.

The check earned its keep: the first vectorized version double-counted the bias
term. Each kernel consumes W in 32-weight chunks but the bias is per 64-weight
group, so `b*sum(x_g)` was added twice per group. The scalar kernel looped per
group and was correct, so only a cross-check across differently-structured
implementations exposed it. Fix: keep the x partial sums per 32-chunk rather
than per group, so the bias lands exactly once per chunk with no branch.

## Build and run

No Xcode on this machine, only Command Line Tools — so there is no offline
`metal` compiler and no `.metallib`. Metal.framework's runtime shader compiler
(`newLibraryWithSource:`) is part of the OS, needs no developer tools, and is
what this project uses. Kernel iteration is a file edit plus a rerun.

```sh
make
./int4gemv --M 32768 --K 16384 --iters 25 --warmup 6      # headline number
./int4gemv --M 16384 --K 16384 --dump results             # dump tensors, then:
python3 bench/compare_mlx.py                              # cross-check + MLX
```

Flags: `--M --K --G --tg --iters --warmup --repeat --src --dump`.
`--repeat` is chosen automatically so each timed span is ~5 ms; the cache-defeat
rotation sizes itself to ~200 MB of distinct weights. Both mean small shapes now
report honestly, so no minimum matrix size is required. Verified at group sizes
32/64/128 and at `M` not divisible by the threadgroup row count.
Captured runs are in `results/`.
