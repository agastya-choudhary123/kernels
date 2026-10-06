# kernels

Five Metal compute kernels for LLM inference on Apple silicon. Each one is
compared against MLX on the same machine, with the same shapes and the same
measurement method.

| project | what it does | result |
|---|---|---|
| [int4-gemv](int4-gemv/) | int4 matrix-vector product for decode | ~94% of peak DRAM bandwidth, 5–10% faster than `mlx.quantized_matmul` |
| [vq-gemv](vq-gemv/) | fused codebook (VQ) dequant + matmul | ~6% faster than `mlx.quantized_matmul`, 4.70% lower perplexity than int4 at the same size |
| [hetero-gemv](hetero-gemv/) | one GEMV split across the GPU and CPU | ~97% of the 120 GB/s bus, 11–13% faster than MLX |
| [moe-pipeline](moe-pipeline/) | pipelined SSD-to-GPU expert streaming for gpt-oss-120b | 1.45–1.51x faster than stop-and-wait, ~99.8% of raw SSD read speed |
| [moe-fused](moe-fused/) | a full MoE block in 3 kernels instead of 28 | 3.4x faster than the fastest MLX version (mostly due to how MLX gathers experts) |

There's no MLX comparison for moe-pipeline, because stock MLX can't load
gpt-oss-120b on this machine at all.

Each directory has its own README with instructions to reproduce the results.

## Hardware

```
Apple M4 (base, Mac16,1): 10 CPU cores (4P/6E), 10 GPU cores, 16 GB LPDDR5X-7500
128-bit memory bus, 120.0 GB/s theoretical peak
macOS 15.3.1 (24D70), Metal 3, simd width 32, 32 KB threadgroup memory
```

All the numbers were measured on this machine. The kernels are tuned for it,
and the bandwidth percentages only make sense for this memory configuration.

You don't need Xcode, just the Command Line Tools. Each project compiles its
`.metal` source at startup with Metal's runtime compiler
(`newLibraryWithSource:`), so to change a kernel you just edit the file and
run it again.

## Methodology

- Each kernel's output is checked against an independent implementation
  before any timing is reported.
- Bandwidth-bound kernels are reported as a percentage of measured peak
  bandwidth, not FLOP/s.
- Variants are timed round-robin in a single process. Drift between
  processes on this machine (~8%) is bigger than most of the differences
  being measured.
- Weight buffers are rotated so no call finds its data already in cache.

In moe-fused, cutting kernel launches from 28 to 3 didn't change the run time.
The block is limited by memory bandwidth, so it only gets faster by reading
fewer bytes.
