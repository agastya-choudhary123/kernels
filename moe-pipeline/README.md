# moe-pipeline — stream-and-compute pipelined MoE decode

Software-pipelines the SSD→unified-memory boundary: worker threads stream the
next gpt-oss-120b experts off the SSD while the GPU computes the previous one,
instead of stop-and-wait.

Real experts, real device. This reads
[`../../moe-stream`](../../moe-stream)'s repacked `experts.bin` — 62 GB,
4608 experts of 14,024,704 bytes each — straight into `MTLBuffer` slots the GPU
reads in place, and runs the expert's three int4 GEMVs plus SwiGLU on them. The
page cache is bypassed (`F_NOCACHE`), read-ahead is off (`F_RDAHEAD`), and no
expert is fetched twice in a run, so every byte reported here actually came off
the device.

## Result

288 experts (4.04 GB), depth 3, 4 reader threads, best of 3:

| driver | GB/s | tok/s | SSD busy | |
|---|---|---|---|---|
| `seq` stop-and-wait | 2.32 – 2.38 | 1.16 – 1.18 | 73 – 75% | read, compute, read, compute |
| **`pipe` stream+compute** | **3.46 – 3.47** | **1.72** | **100%** | depth-3 pipeline |
| `io` reads only | 3.46 – 3.48 | 1.72 | 100% | the ceiling |

**1.45 – 1.51x over stop-and-wait, 1.15 → 1.74 tok/s**, across seven runs at two
sizes. The
pipeline lands at **99.5 – 99.9% of the pure-IO rate**, so the GPU work is now
completely free: streaming with compute costs the same wall time as streaming
with the compute deleted.

`tok/s` here counts a token as 36 layers x top-4 = 144 *full* expert fetches with
a 100% miss rate. That is deliberately the worst case and is **not** comparable
to moe-stream's end-to-end 2.35 tok/s, which reads ~1.1 GiB/token instead of 1.93
because its slot cache and free-prefetch absorb ~40% of the fetches. Applying the
rates above to that measured 1.1 GiB/token projects 1.99 → 2.94 tok/s; that is
arithmetic on this pipeline's bandwidth, not an end-to-end measurement.

## There is no MLX baseline for this, and that is the point

`mlx_lm.load` materialises every expert before the first token. This model is
**62 GB** of weights against **16 GB** of RAM and an 11.5 GB recommended Metal
working set, so stock MLX cannot reach the first token at all here — there is no
MLX number to be faster than. The baseline that exists is stop-and-wait, which
is what a straightforward streaming implementation does, and that is what the
table above measures against.

## The win is not where the brief said it would be

The framing was "overlap IO with GPU compute on the previous expert". Measured
on this machine, that is worth almost nothing, because the two are not remotely
the same size:

```
IO      per expert   4.03 ms   (13.38 MB at 3.47 GB/s)
compute per expert   0.17 ms   (the same 13.38 MB at 84 GB/s, measured over
                                rotating cold slots, not one hot one)
                     -> 24x more IO than compute
```

Perfectly overlapping compute with IO can therefore remove at most **4.0%** of
stop-and-wait's wall time. The pipeline does capture essentially all of it — ~100%
of GPU time is hidden — but that is **under 3 points of the ~47% total gain**.

The other 44 points come from something the brief did not mention: stop-and-wait
leaves **the SSD idle ~25% of the time**. Not idle during compute — compute is
only 4% — idle in the gap between finishing one read and having the next one
issued. A depth-2 pipeline removes it by construction, because there is always a
read outstanding. The device busy fraction going 73 – 75% → 100% is the whole story,
and it is why this is a pipelining win rather than an overlap win.

That distinction is worth stating precisely because it changes what you would
build next: no amount of making the GPU faster helps, and neither would fusing
the expert kernels. Only fewer bytes would.

## Two bugs worth fixing

**A missed-wakeup hang.** Workers published slot readiness and signalled the
condition variable without holding the mutex the waiter tests under. A worker can
then store and notify in the window between the waiter evaluating its predicate
as false and actually blocking, and the waiter never wakes. It never hung in
practice, which is exactly why it was worth fixing rather than waiting for it to.

**Measuring the drivers in separate blocks.** All reps of `seq`, then all of
`io`, then all of `pipe` — the same mistake `../int4-gemv` had to fix. SSD and
thermal state drift across a run and the drift lands on whichever driver went
last. Round-robining the three moved the reported speedup from 1.38x to 1.47x.

## One ordering bug worth 2%

The first working pipeline refilled the consumed slot **after** the GPU compute:

```
wait_for(slot i) ; compute(i) ; submit(i + depth)
```

which drains the queue for the entire duration of the GPU work — precisely the
window the pipeline exists to cover. It hid only ~50% of compute and sat 2.2%
below the reads-only ceiling. Refilling *before* the compute:

```
wait_for(slot i) ; submit(i + depth) ; compute(i)
```

takes it to 97% of compute hidden and 99.9% of the ceiling. With `depth + 2`
slots the slot being refilled is `i + depth`, which is never one the GPU still
has to consume, so the reorder is free.

## Depth 3 is enough; depth 16 is not better

| depth | GB/s | | workers | GB/s (reads only) |
|---|---|---|---|---|
| 1 | 3.22 | | 1 | 3.25 |
| 2 | 3.43 | | 2 | 3.50 |
| **3** | **3.48** | | **4** | **3.50** |
| 4 | 3.48 | | 8 | 3.46 |
| 8 | 3.49 | | 12 | 3.48 |
| 16 | 3.40 | | | |

Double buffering gets 98% of the win and depth 3 saturates. This follows directly
from the ratio above: when compute is 5% of IO you only need one read in flight
beyond the current one, and queueing sixteen only adds scheduling noise. Two
reader threads are likewise enough — at 13.4 MB a single request already
amortises device latency, so concurrency is worth 1.08x, not 2x.

## Correctness

`verify_mlx.py` re-reads the same expert blob the kernel just consumed, rebuilds
all three projections and the SwiGLU with `mlx.core`, and compares:

```
  gate   vs MLX: max|d|/|ref|inf = 2.32e-07  ok
  up     vs MLX: max|d|/|ref|inf = 3.31e-07  ok
  h      vs MLX: max|d|/|ref|inf = 6.13e-07  ok
  y      vs MLX: max|d|/|ref|inf = 1.53e-07  ok
```

Agreement is ~1e-07 rather than the ~1e-04 of the other projects here because
this kernel accumulates the int4 dot product in fp32 throughout; K is only 2880,
so there is no reason to spend the accuracy. The SwiGLU matches
`mlx_lm.models.gpt_oss.swiglu` exactly, clamps and the +1 linear bias included.

Scales and biases are read as native MSL `bfloat` — that is how gpt-oss ships,
and the streamed bytes are consumed in place with no conversion pass.

## Build and run

```sh
make
./moepipe --experts 288 --depth 3 --workers 4        # the headline table
./moepipe --experts 192 --depth 2 --mode pipe        # one driver only
mkdir -p results/verify && ./moepipe --verify results/verify && python3 verify_mlx.py
```

Flags: `--store --experts --depth --workers --reps --mode {all,seq,io,pipe} --verify`.
Needs `~/Desktop/moe-stream/model-120b/experts.bin`. Captured runs in `results/`.
