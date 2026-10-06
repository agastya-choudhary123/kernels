# moe-pipeline

Pipelined expert streaming for gpt-oss-120b decode. Reader threads load the
next experts from the SSD while the GPU computes the current one, instead of
waiting for each read to finish before starting the next step.

It works with real experts on a real disk. It reads the repacked
`experts.bin` from [moe-stream](https://github.com/agastya-choudhary123/moe-stream)
(62 GB, 4608 experts of 14,024,704 bytes each) directly into `MTLBuffer`
slots, and runs each expert's three int4 GEMVs plus SwiGLU on the GPU. The page
cache (`F_NOCACHE`) and read-ahead (`F_RDAHEAD`) are both turned off, and no
expert is read twice in a run, so every byte really comes off the SSD.

Stock MLX can't run this model on this machine at all. `mlx_lm.load` loads
every expert up front, and the model is 62 GB against 16 GB of RAM. The
baseline here is a simple stop-and-wait loop.

## Result

288 experts (4.04 GB), pipeline depth 3, 4 reader threads, best of 3 runs:

| driver | GB/s | tok/s | SSD busy |
|---|---|---|---|
| `seq`: read, compute, read, compute | 2.32–2.38 | 1.16–1.18 | 73–75% |
| `pipe`: depth-3 pipeline | 3.46–3.47 | 1.72 | 100% |
| `io`: reads only (the ceiling) | 3.46–3.48 | 1.72 | 100% |

Across seven runs at two sizes, the pipeline was 1.45–1.51x faster than
stop-and-wait, and reached 99.5–99.9% of the reads-only rate.

Here a "token" means 36 layers × top-4 = 144 expert reads with no cache hits.
That's a worst case, so it isn't comparable to moe-stream's 2.35 tok/s, where
caching and prefetching mean it only reads about 1.1 GiB per token. At this
pipeline's bandwidth, moe-stream would go from about 1.99 to 2.94 tok/s. That's
a calculation, not a measurement.

## Where the speedup comes from

The obvious explanation would be that GPU compute overlaps with I/O. But I/O
is far bigger than compute:

```
IO      per expert   4.03 ms   (13.38 MB at 3.47 GB/s)
compute per expert   0.17 ms   (13.38 MB at 84 GB/s, over cold slots)
```

So overlapping compute could save at most 4%, and that accounts for less than
3 points of the ~47% gain. The rest comes from keeping the SSD busy.
Stop-and-wait leaves the SSD idle about 25% of the time, in the gap between one
read finishing and the next one being issued. With a pipeline there's always a
read in flight, so SSD utilization goes from 73–75% to 100%.

That means a faster GPU kernel or fused expert kernels wouldn't help. The only
way to speed this up further is to read fewer bytes.

## Depth and worker count

| depth | GB/s | | workers | GB/s (reads only) |
|---|---|---|---|---|
| 1 | 3.22 | | 1 | 3.25 |
| 2 | 3.43 | | 2 | 3.50 |
| 3 | 3.48 | | 4 | 3.50 |
| 4 | 3.48 | | 8 | 3.46 |
| 8 | 3.49 | | 12 | 3.48 |
| 16 | 3.40 | | | |

Double buffering gets 98% of the gain, and depth 3 is the maximum. Because
compute is only about 5% of I/O time, one extra read in flight is enough. Two
reader threads are also enough: a single 13.4 MB read already hides the
device's latency.

## Bugs I fixed

- **Refill order.** The first version refilled a slot after computing on it:
  `wait(i); compute(i); submit(i + depth)`. That leaves the read queue empty
  while the GPU works. Submitting the refill before the compute,
  `wait(i); submit(i + depth); compute(i)`, raised hidden compute from about
  50% to 97%, and throughput from 97.8% to 99.9% of the ceiling. With
  `depth + 2` slots, slot `i + depth` is never one the GPU still needs.
- **Missed wakeup.** Workers signaled the condition variable without holding
  the mutex, so a signal could land after the waiter checked its condition
  but before it went to sleep. It never actually hung, but it could have.
- **Timing drivers in blocks.** Originally I ran every rep of `seq`, then
  every rep of `io`, then every rep of `pipe`. SSD and thermal drift then
  penalized whichever driver ran last. Switching to round-robin moved the
  measured speedup from 1.38x to 1.47x.

## Correctness

`verify_mlx.py` rereads the expert the kernel just used, recomputes all
three projections and the SwiGLU with `mlx.core`, and compares the results:

```
  gate   vs MLX: max|d|/|ref|inf = 2.32e-07  ok
  up     vs MLX: max|d|/|ref|inf = 3.31e-07  ok
  h      vs MLX: max|d|/|ref|inf = 6.13e-07  ok
  y      vs MLX: max|d|/|ref|inf = 1.53e-07  ok
```

The error is about 1e-7, not the 1e-4 seen in the other kernels, because this
kernel accumulates in fp32 (K is only 2880). The SwiGLU matches
`mlx_lm.models.gpt_oss.swiglu` exactly, including the clamps and the +1 bias.
Scales and biases are read directly as MSL `bfloat`, which is the format
gpt-oss ships in.

## Build and run

You'll need moe-stream's `model-120b/experts.bin`. By default the program
looks for it at `~/Desktop/moe-stream/model-120b/experts.bin`. Use `--store`
to point it somewhere else.

```sh
make
./moepipe --experts 288 --depth 3 --workers 4        # the table above
./moepipe --experts 192 --depth 2 --mode pipe        # one driver only
mkdir -p results/verify && ./moepipe --verify results/verify && python3 verify_mlx.py
```

Flags: `--store --experts --depth --workers --reps --mode {all,seq,io,pipe}
--verify`. Saved runs are in `results/`.
