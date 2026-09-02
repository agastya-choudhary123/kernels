kernels
-------

Five Metal compute kernels for LLM inference on Apple silicon, each measured
against MLX on the same machine, shapes and protocol.

| project | what it does | result |
|---|---|---|
| [int4-gemv](int4-gemv/) | int4 GEMV for decode | ~94% of peak DRAM bandwidth, 5-10% over `mlx.quantized_matmul` |
| [vq-gemv](vq-gemv/) | fused codebook dequant-matmul | ~6% over `mlx.quantized_matmul`, 4.70% better perplexity at equal bytes/param |
| [hetero-gemv](hetero-gemv/) | one GEMV split across GPU and CPU vector units | ~97% of the 120 GB/s bus, 11-13% over MLX |
| [moe-pipeline](moe-pipeline/) | SSD→GPU expert streaming for gpt-oss-120b | 1.45-1.51x over stop-and-wait, ~99.8% of the pure-IO ceiling |
| [moe-fused](moe-fused/) | a full MoE block in 3 kernels instead of 28 | 3.4x over MLX's best formulation, verified to 2.2e-07 |

Every project is faster than MLX except `moe-pipeline`, where stock MLX cannot
load the model at all.

Each directory has its own README with the reproduction command.

### Requirements

    Apple M4 (base, Mac16,1) — 10 CPU cores (4P/6E), 10 GPU cores, 16 GB LPDDR5X-7500
    128-bit memory bus, 120.0 GB/s theoretical peak DRAM bandwidth
    macOS 15.3.1 (24D70), Metal 3, simd width 32, 32 KB threadgroup memory

Every number here was measured on that machine. The kernels are not portable to
other Apple silicon without retuning, and the bandwidth-percentage figures are
meaningless on different memory geometry.

No Xcode is required, only the Command Line Tools. There is therefore no
offline `metal` compiler and no `.metallib` to build; every project compiles
its `.metal` sources at process start through Metal.framework's runtime shader
compiler, which also makes kernel iteration a file edit and a rerun.

### Ground rules

These held for every number above:

- No fallbacks, no simulated paths, no CPU stand-ins for GPU work.
- Correctness is verified against an independent implementation before any
  performance number is reported.
- Bandwidth-bound kernels are scored as a percentage of measured peak, not in
  FLOP/s, because FLOP/s on a GEMV says nothing.

### Notes

For `moe-fused`, the 9.3x reduction in kernel launches is worth nothing on its
own. The block is bound by bytes, not by launches, and the speedup comes from
what fusion lets you avoid reading.
