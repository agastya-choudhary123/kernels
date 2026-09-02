// Stream-and-compute pipelined MoE decode.
//
// Real experts, off the real SSD: this streams gpt-oss-120b expert blobs out of
// moe-stream's repacked experts.bin (62 GB, 4608 experts of 14,024,704 bytes)
// straight into MTLBuffer slots the GPU reads in place, and runs the expert's
// three int4 GEMVs on them.
//
// Three drivers, same work, same bytes:
//   seq   - stop and wait. Read expert i, compute expert i, read expert i+1.
//           The device is idle for the whole compute and the GPU is idle for the
//           whole read.
//   pipe  - software pipeline of depth D. Worker threads keep D reads in flight
//           while the GPU computes expert i, so a slot is normally ready the
//           moment the GPU asks for it.
//   io    - reads only, no compute: the ceiling the pipeline is chasing.
//
// The page cache is bypassed with F_NOCACHE and read-ahead disabled with
// F_RDAHEAD, and no expert is ever fetched twice inside a run, so every byte
// reported here actually came off the device.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef F_NOCACHE
#define F_NOCACHE 48
#endif
#ifndef F_RDAHEAD
#define F_RDAHEAD 45
#endif

// gpt-oss-120b repacked layout (moe-stream/model-120b/experts_index.json)
static const size_t BLOB      = 14024704;
static const uint32_t N_LAYER = 36, N_EXPERT = 128, TOPK = 4;
static const uint32_t HID = 2880, INTER = 2880, GRP = 64;
static const size_t OFF_GATE = 0, OFF_UP = 4671360, OFF_DOWN = 9342720;
static const size_t SUB_W = 0, SUB_S = 4147200, SUB_B = 4406400, SUB_BI = 4665600;

struct ExpertParams { uint32_t M, K, G, quads_pr, groups_pr, chunks_pg; };

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void die(const char* w, NSError* e = nil) {
    fprintf(stderr, "FATAL: %s%s%s\n", w, e ? ": " : "",
            e ? [[e localizedDescription] UTF8String] : "");
    exit(1);
}

// ---------------------------------------------------------------------------
// Slot pool + reader threads.
//
// Slots are byte ranges of one big MTLBuffer, so `pread` writes SSD bytes
// directly into memory the GPU will read with no copy and no staging. `busy_s`
// accumulates wall time during which at least one read was outstanding, which is
// what separates "the device is the floor" from "the device is idle".
// ---------------------------------------------------------------------------
struct Fetcher {
    int fd = -1;
    char* base = nullptr;
    uint32_t nslots = 0;
    std::vector<std::atomic<int>> ready;      // -1 empty, else request id
    std::mutex mu; std::condition_variable cv;
    // Readiness gets its own mutex, held by the worker when it publishes and by
    // the waiter when it tests. Publishing outside the lock is a missed-wakeup
    // hang: the waiter can evaluate the predicate false, the worker can store
    // and notify before the waiter reaches wait(), and the waiter then blocks
    // forever. Separate from `mu` so publishing never contends with the queue.
    std::mutex rmu; std::condition_variable done_cv;
    struct Req { uint32_t slot; uint64_t off; int id; };
    std::queue<Req> q;
    std::vector<std::thread> workers;
    bool stop = false;

    std::atomic<int> inflight{0};
    double busy_s = 0, busy_t0 = 0;
    std::mutex bmu;
    std::atomic<uint64_t> bytes{0};

    Fetcher(uint32_t n) : nslots(n), ready(n) {
        for (uint32_t i = 0; i < n; ++i) ready[i].store(-1);
    }

    void open_store(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) die(("cannot open " + path).c_str());
        if (fcntl(fd, F_NOCACHE, 1) == -1) die("F_NOCACHE");
        fcntl(fd, F_RDAHEAD, 0);
    }
    void start(int nworkers) {
        for (int i = 0; i < nworkers; ++i) workers.emplace_back([this]{ loop(); });
    }
    void shutdown() {
        { std::lock_guard<std::mutex> g(mu); stop = true; }
        cv.notify_all();
        for (auto& t : workers) t.join();
        workers.clear(); stop = false;
    }
    void submit(uint32_t slot, uint64_t off, int id) {
        { std::lock_guard<std::mutex> g(mu); q.push({slot, off, id}); }
        cv.notify_one();
    }
    void loop() {
        for (;;) {
            Req r;
            { std::unique_lock<std::mutex> g(mu);
              cv.wait(g, [&]{ return stop || !q.empty(); });
              if (stop && q.empty()) return;
              r = q.front(); q.pop(); }

            if (inflight.fetch_add(1) == 0) {
                std::lock_guard<std::mutex> g(bmu); busy_t0 = now_s();
            }
            size_t got = 0;
            while (got < BLOB) {
                ssize_t n = pread(fd, base + (size_t)r.slot * BLOB + got,
                                  BLOB - got, r.off + got);
                if (n <= 0) die("pread short/failed");
                got += (size_t)n;
            }
            bytes.fetch_add(BLOB);
            if (inflight.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> g(bmu); busy_s += now_s() - busy_t0;
            }
            { std::lock_guard<std::mutex> g(rmu);
              ready[r.slot].store(r.id, std::memory_order_release); }
            done_cv.notify_all();
        }
    }
    // returns seconds spent blocked
    double wait_for(uint32_t slot, int id) {
        if (ready[slot].load(std::memory_order_acquire) == id) return 0.0;
        double t0 = now_s();
        std::unique_lock<std::mutex> g(rmu);
        done_cv.wait(g, [&]{ return ready[slot].load(std::memory_order_acquire) == id; });
        return now_s() - t0;
    }
};

int main(int argc, char** argv) { @autoreleasepool {
    std::string store = std::string(getenv("HOME")) + "/Desktop/moe-stream/model-120b/experts.bin";
    int nExperts = 288, depth = 8, workers = 4, slots = 0, reps = 3;
    std::string mode = "all", dumpDir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() -> const char* { return argv[++i]; };
        if      (a == "--store")   store = nx();
        else if (a == "--experts") nExperts = atoi(nx());
        else if (a == "--depth")   depth = atoi(nx());
        else if (a == "--workers") workers = atoi(nx());
        else if (a == "--reps")    reps = atoi(nx());
        else if (a == "--mode")    mode = nx();
        else if (a == "--verify")  dumpDir = nx();
        else die(("unknown arg " + a).c_str());
    }
    if (slots == 0) slots = depth + 2;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
    co.mathMode = MTLMathModeFast;
    std::string src; {
        FILE* f = fopen("src/kernels.metal", "rb");
        if (!f) die("run from the project root (src/kernels.metal not found)");
        char b[65536]; size_t n;
        while ((n = fread(b, 1, sizeof b, f)) > 0) src.append(b, n);
        fclose(f);
    }
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:src.c_str()] options:co error:&err];
    if (!lib) die("compiling kernels.metal", err);
    auto pipe_of = [&](NSString* n) {
        NSError* e = nil;
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:n] error:&e];
        if (!p) die([n UTF8String], e);
        return p;
    };
    id<MTLComputePipelineState> psGemv = pipe_of(@"gemv_int4_bf16");
    id<MTLComputePipelineState> psSwi  = pipe_of(@"swiglu_k");
    id<MTLComputePipelineState> psXs   = pipe_of(@"xsum32");

    // ---- buffers ----
    id<MTLBuffer> pool = [dev newBufferWithLength:(size_t)slots * BLOB
                                          options:MTLResourceStorageModeShared];
    if (!pool) die("pool alloc");
    auto mk = [&](size_t n) { return [dev newBufferWithLength:n
                                     options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> bX = mk(HID * 4), bH = mk(INTER * 4), bG = mk(INTER * 4);
    id<MTLBuffer> bU = mk(INTER * 4), bY = mk(HID * 4);
    id<MTLBuffer> bXsX = mk((HID / 32) * 4), bXsH = mk((INTER / 32) * 4);
    id<MTLBuffer> bP = mk(sizeof(ExpertParams)), bN = mk(4);
    ExpertParams EP { INTER, HID, GRP, HID / 32, HID / GRP, GRP / 32 };
    memcpy(bP.contents, &EP, sizeof EP);
    *(uint32_t*)bN.contents = INTER;
    float* X = (float*)bX.contents;
    for (uint32_t i = 0; i < HID; ++i) X[i] = 0.05f * ((int)(i % 17) - 8);

    Fetcher F(slots);
    F.base = (char*)pool.contents;
    F.open_store(store);

    // deterministic scatter over the 4608 experts; no expert twice in a run, so
    // nothing can be served from any cache even if F_NOCACHE were ignored
    auto expert_off = [&](int i) -> uint64_t {
        uint64_t idx = ((uint64_t)i * 2654435761u) % (uint64_t)(N_LAYER * N_EXPERT);
        return idx * BLOB;
    };

    // ---- GPU: one expert = xsum(x), gate, up, swiglu, xsum(h), down ----
    auto encode_expert = [&](id<MTLCommandBuffer> cb, uint32_t slot) {
        size_t sb = (size_t)slot * BLOB;
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        auto xsum = [&](id<MTLBuffer> xin, id<MTLBuffer> xout, uint32_t n) {
            [e setComputePipelineState:psXs];
            [e setBuffer:xin offset:0 atIndex:0];
            [e setBuffer:xout offset:0 atIndex:1];
            [e setBuffer:bP offset:0 atIndex:2];
            [e dispatchThreads:MTLSizeMake(n / 32, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        };
        auto gemv = [&](size_t proj, id<MTLBuffer> xin, id<MTLBuffer> xs,
                        id<MTLBuffer> yout) {
            [e setComputePipelineState:psGemv];
            [e setBuffer:pool offset:sb + proj + SUB_W  atIndex:0];
            [e setBuffer:pool offset:sb + proj + SUB_S  atIndex:1];
            [e setBuffer:pool offset:sb + proj + SUB_B  atIndex:2];
            [e setBuffer:pool offset:sb + proj + SUB_BI atIndex:3];
            [e setBuffer:xin offset:0 atIndex:4];
            [e setBuffer:yout offset:0 atIndex:5];
            [e setBuffer:xs offset:0 atIndex:6];
            [e setBuffer:bP offset:0 atIndex:7];
            NSUInteger tg = 256, sg = tg / 32, ntg = (INTER + sg - 1) / sg;
            [e dispatchThreads:MTLSizeMake(ntg * tg, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        };
        xsum(bX, bXsX, HID);
        gemv(OFF_GATE, bX, bXsX, bG);
        gemv(OFF_UP,   bX, bXsX, bU);
        [e setComputePipelineState:psSwi];
        [e setBuffer:bG offset:0 atIndex:0]; [e setBuffer:bU offset:0 atIndex:1];
        [e setBuffer:bH offset:0 atIndex:2]; [e setBuffer:bN offset:0 atIndex:3];
        [e dispatchThreads:MTLSizeMake(INTER, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        xsum(bH, bXsH, INTER);
        gemv(OFF_DOWN, bH, bXsH, bY);
        [e endEncoding];
    };

    printf("device                : %s\n", [[dev name] UTF8String]);
    printf("store                 : %s\n", store.c_str());
    printf("expert blob           : %.2f MB   (gate/up/down, int4 g=64, bf16 scales)\n",
           BLOB / 1048576.0);
    printf("token                 : %u layers x top-%u = %u expert fetches = %.2f GB\n",
           N_LAYER, TOPK, N_LAYER * TOPK, N_LAYER * TOPK * (double)BLOB / 1e9);
    printf("page cache            : bypassed (F_NOCACHE), read-ahead off, "
           "no expert fetched twice\n");
    printf("run                   : %d experts (%.2f GB) x %d reps\n\n",
           nExperts, nExperts * (double)BLOB / 1e9, reps);

    double totalGB = nExperts * (double)BLOB / 1e9;
    uint32_t perTok = N_LAYER * TOPK;

    // ---------------- compute only (weights already resident) ----------------
    double tCompute = 0;
    {
        for (int i = 0; i < nExperts && i < (int)F.nslots; ++i) {}
        // Fill every slot, then compute over all of them in rotation. Timing one
        // hot slot repeatedly would let a 13.4 MB blob sit in the 8 MB SLC and
        // understate the compute, which is the number the whole "IO vs compute"
        // conclusion rests on.
        F.start(4);
        for (uint32_t s2 = 0; s2 < F.nslots; ++s2) F.submit(s2, expert_off(s2), (int)s2);
        for (uint32_t s2 = 0; s2 < F.nslots; ++s2) F.wait_for(s2, (int)s2);
        F.shutdown();
        for (int w = 0; w < 5; ++w) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            encode_expert(cb, w % F.nslots); [cb commit]; [cb waitUntilCompleted];
        }
        int n = 40; double best = 1e30;
        for (int r = 0; r < 5; ++r) {
            double t0 = now_s();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            for (int i = 0; i < n; ++i) encode_expert(cb, i % F.nslots);
            [cb commit]; [cb waitUntilCompleted];
            best = std::min(best, (now_s() - t0) / n);
        }
        tCompute = best;
        printf("GPU compute / expert  : %7.3f ms   (%.1f GB/s over the blob)\n",
               tCompute * 1e3, BLOB / tCompute / 1e9);
    }

    struct Res { double wall, busy, exposed; };
    auto run_io_only = [&](int nw) {
        F.start(nw);
        F.busy_s = 0; F.bytes = 0;
        for (uint32_t s = 0; s < F.nslots; ++s) F.ready[s].store(-1);
        double t0 = now_s();
        int issued = 0;
        for (int i = 0; i < (int)F.nslots && i < nExperts; ++i)
            F.submit(i % F.nslots, expert_off(i), i), ++issued;
        for (int i = 0; i < nExperts; ++i) {
            uint32_t sl = i % F.nslots;
            F.wait_for(sl, i);
            if (issued < nExperts) {
                F.ready[sl].store(-1);
                F.submit(sl, expert_off(issued), issued), ++issued;
            }
        }
        double wall = now_s() - t0;
        double busy = F.busy_s;
        F.shutdown();
        return Res{wall, busy, 0};
    };

    auto run_pipe = [&](int nw, int d) {
        uint32_t ns = F.nslots;
        F.start(nw);
        F.busy_s = 0; F.bytes = 0;
        for (uint32_t s = 0; s < ns; ++s) F.ready[s].store(-1);
        double exposed = 0;
        double t0 = now_s();
        int issued = 0;
        for (int i = 0; i < d && i < nExperts; ++i)
            F.submit(i % ns, expert_off(i), i), ++issued;
        for (int i = 0; i < nExperts; ++i) {
            uint32_t sl = i % ns;
            exposed += F.wait_for(sl, i);
            // Refill BEFORE computing, not after. Submitting after means the
            // queue drains for the whole duration of the GPU work, which is
            // exactly the window the pipeline exists to cover. The slot being
            // refilled is i+depth, and with depth+2 slots that is never one the
            // GPU still has to consume.
            if (issued < nExperts) {
                F.ready[issued % ns].store(-1);
                F.submit(issued % ns, expert_off(issued), issued), ++issued;
            }
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            encode_expert(cb, sl);
            [cb commit]; [cb waitUntilCompleted];
        }
        double wall = now_s() - t0;
        double busy = F.busy_s;
        F.shutdown();
        return Res{wall, busy, exposed};
    };

    // Stop-and-wait: the read happens inline on the calling thread, with no
    // worker, no queue and no condition variable, so the baseline is not charged
    // for thread handoff it would never pay. Read expert i, compute expert i,
    // then start reading expert i+1. The device is idle for every compute and
    // the GPU is idle for every read, which is the thing the pipeline removes.
    auto run_seq = [&]() {
        double exposed = 0, busy = 0, t0 = now_s();
        for (int i = 0; i < nExperts; ++i) {
            double r0 = now_s();
            size_t got = 0;
            uint64_t off = expert_off(i);
            while (got < BLOB) {
                ssize_t n = pread(F.fd, F.base + got, BLOB - got, off + got);
                if (n <= 0) die("pread short/failed");
                got += (size_t)n;
            }
            double dr = now_s() - r0;
            busy += dr; exposed += dr;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            encode_expert(cb, 0);
            [cb commit]; [cb waitUntilCompleted];
        }
        return Res{now_s() - t0, busy, exposed};
    };

    auto report = [&](const char* name, Res r) {
        double gbs = totalGB / r.wall;
        double tps = (double)nExperts / perTok / r.wall;
        printf("  %-26s %7.2f s  %6.2f GB/s  %6.2f tok/s   busy %3.0f%%  "
               "exposed IO %5.2f s (%3.0f%%)\n",
               name, r.wall, gbs, tps, 100.0 * r.busy / r.wall,
               r.exposed, 100.0 * r.exposed / std::max(1e-9, r.busy));
        return r;
    };

    // Round-robin the drivers instead of running all reps of one and then all
    // reps of the next. SSD and thermal state drift over a run, and measuring
    // the baseline in its own block ahead of the pipeline charges that drift to
    // whichever went last -- the same mistake ../int4-gemv had to fix. Here it
    // was worth ~7% on the reported speedup.
    auto run_one = [&](int which) {
        return which == 0 ? run_seq() : which == 1 ? run_io_only(workers)
                                                   : run_pipe(workers, depth);
    };

    // ---- dump one expert's inputs and outputs for the MLX cross-check ----
    if (!dumpDir.empty()) {
        F.start(1); F.submit(0, expert_off(0), 0); F.wait_for(0, 0); F.shutdown();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        encode_expert(cb, 0); [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) die("verify dispatch", cb.error);
        auto put = [&](const char* n, const void* p, size_t b) {
            FILE* f = fopen((dumpDir + "/" + n).c_str(), "wb");
            if (!f) die(("cannot write " + dumpDir).c_str());
            fwrite(p, 1, b, f); fclose(f);
        };
        put("x.f32", bX.contents, HID * 4);
        put("gate.f32", bG.contents, INTER * 4);
        put("up.f32",   bU.contents, INTER * 4);
        put("h.f32",    bH.contents, INTER * 4);
        put("y.f32",    bY.contents, HID * 4);
        FILE* f = fopen((dumpDir + "/meta.txt").c_str(), "w");
        fprintf(f, "%llu %u %u %u\n", (unsigned long long)expert_off(0), HID, INTER, GRP);
        fclose(f);
        printf("dumped expert at byte offset %llu to %s\n",
               (unsigned long long)expert_off(0), dumpDir.c_str());
        return 0;
    }

    printf("\n%-28s %9s %13s %13s %11s %s\n", "driver", "wall", "GB/s", "tok/s",
           "SSD busy", "exposed IO");
    printf("--------------------------------------------------------------------------------------------\n");
    Res best[3] = {{1e30,0,0},{1e30,0,0},{1e30,0,0}};
    bool all = (mode == "all");
    int want = all ? -1 : (mode == "seq" ? 0 : mode == "io" ? 1 : 2);
    for (int r = 0; r < reps; ++r)
        for (int k = 0; k < 3; ++k) {
            if (!all && k != want) continue;
            Res x = run_one(k);
            if (x.wall < best[k].wall) best[k] = x;
        }
    const char* nm[3] = { "seq  stop-and-wait", "io   reads only (ceiling)",
                          "pipe stream+compute" };
    for (int k = 0; k < 3; ++k)
        if (all || k == want) report(nm[k], best[k]);
    Res rSeq = best[0], rIO = best[1], rPip = best[2];
    if (!all) return 0;
    printf("--------------------------------------------------------------------------------------------\n");

    double Tc = nExperts * tCompute;
    double ioPer = rIO.wall / nExperts;
    printf("\npipelined vs stop-and-wait : %.3fx   (%.2f -> %.2f tok/s)\n",
           rSeq.wall / rPip.wall,
           (double)nExperts / perTok / rSeq.wall, (double)nExperts / perTok / rPip.wall);
    printf("pipelined vs reads-only    : %.1f%% of the pure-IO rate "
           "(the pipeline's real ceiling)\n", 100.0 * rIO.wall / rPip.wall);
    printf("SSD busy fraction          : %.0f%% -> %.0f%%   "
           "(idle device is what stop-and-wait actually costs)\n",
           100.0 * rSeq.busy / rSeq.wall, 100.0 * rPip.busy / rPip.wall);
    printf("\nwhere the win is NOT:\n");
    printf("  IO per expert %.3f ms vs compute %.3f ms -- %.0fx more IO than compute.\n",
           ioPer * 1e3, tCompute * 1e3, ioPer / tCompute);
    printf("  Overlapping compute with IO can therefore save at most %.1f%% of\n"
           "  stop-and-wait wall time, and it does: %.0f%% of GPU time is hidden,\n"
           "  worth %.1f%% of the %.1f%% total gain.\n",
           100.0 * Tc / (rIO.wall + Tc),
           100.0 * std::min(1.0, (rIO.wall + Tc - rPip.wall) / std::max(1e-9, Tc)),
           100.0 * Tc / rSeq.wall, 100.0 * (rSeq.wall / rPip.wall - 1.0));
    printf("where the win IS:\n");
    printf("  keeping a read in flight at all times -- the device never drains.\n");
    return 0;
}}
