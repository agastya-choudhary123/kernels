# kernels

Metal / Apple-silicon kernel projects. Targets this machine specifically:

    Apple M4 (base, Mac16,1) — 10 CPU cores (4P/6E), 10 GPU cores, 16 GB LPDDR5X-7500
    128-bit memory bus -> 120.0 GB/s theoretical peak DRAM bandwidth
    macOS 15.3.1 (24D70), Metal 3, simd width 32, 32 KB threadgroup memory

Every project here is measured against `mlx` on the same machine, same shapes,
same protocol, and is faster than it — except `moe-pipeline`, where stock MLX
cannot load the model at all.

Ground rules for everything in here:
- No fallbacks, no simulated paths, no CPU stand-ins for GPU work.
- Every number is measured on this machine, by the harness in the project, and
  reproducible with the command printed in the project README.
- Correctness is verified against an independent implementation before any
  performance number is reported.

## Build note

No Xcode is installed, only Command Line Tools, so there is no offline `metal`
compiler and no `.metallib` to build. That is not a blocker: Metal.framework's
runtime shader compiler (`newLibraryWithSource:`) is part of the OS and compiles
MSL directly. All projects here compile their `.metal` sources at process start
through that path, which also makes kernel iteration a file edit + rerun.

## Projects

- `int4-gemv/` — bandwidth-saturating int4 GEMV for LLM decode. Metric is % of
  the 120 GB/s peak actually achieved, not FLOP/s. Reaches ~94% of peak, which is
  100% of a measured pure-read ceiling, and **+5% to +10% on
  `mlx.quantized_matmul`**.
- `vq-gemv/` — fused codebook (vector-quant) dequant-matmul. At identical
  bytes/param it matches scalar int4 throughput, runs **~6% faster than
  `mlx.quantized_matmul`**, and is 4.70% better perplexity on real WikiText-2
  with a real model.
- `hetero-gemv/` — one GEMV split across the Metal GPU and the CPU's vector units
  concurrently over unified memory, load-balanced by measured throughput.
  1.05 – 1.07x over an already-ceiling-rate GPU kernel, ~97% of the 120 GB/s bus,
  and **+11% to +13% over MLX**.
- `moe-pipeline/` — software-pipelined SSD→GPU expert streaming for gpt-oss-120b
  decode, against the real 62 GB store with the page cache bypassed. 1.45-1.51x
  over stop-and-wait and ~99.8% of the pure-IO ceiling; the win is a never-idle
  device, not IO/compute overlap. Stock MLX cannot load this model here at all.
- `moe-fused/` — router + top-k + expert gather + int4 matmul for a gpt-oss-120b
  MoE block in 3 kernels instead of 28, verified against MLX to 2.2e-07 and
  **3.4x faster than MLX's best formulation**. The 9.3x launch reduction is worth
  nothing on its own: the block is bound by bytes, not by launches.
