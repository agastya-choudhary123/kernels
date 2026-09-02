// Benchmark harness for int4 decode GEMV on Apple silicon.
//
// Everything here runs on the real GPU: shaders are compiled at launch by
// Metal.framework's runtime MSL compiler, timings come from the command
// buffer's GPU timestamps, and every kernel's output is checked against an
// independent CPU reference before its number is reported.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <chrono>
#include <string>
#include <dirent.h>
#include <vector>

// M4 (base): LPDDR5X-7500 on a 128-bit bus -> 7500 MT/s * 16 B = 120.0 GB/s.
static const double kPeakGBs = 120.0;

struct GemvParams { uint32_t M, K, G, words_pr, groups_pr, chunks_pg; };

// Which weight copy the next dispatch reads. File scope because ObjC blocks
// capture globals by reference and __block variables cannot cross a lambda.
static uint32_t g_copy = 0, g_ncopy = 1;
// Dispatches packed into one command buffer. A single 4096x4096 GEMV runs in
// ~70 us, which is short enough that the command buffer GPU timestamps
// systematically over-report throughput; batching until the measured span is
// milliseconds removes that. Copies rotate between dispatches so the extra
// repeats do not become cache hits.
static int g_repeat = 1;

static void die(NSError* e, const char* what) {
    fprintf(stderr, "FATAL: %s: %s\n", what,
            e ? [[e localizedDescription] UTF8String] : "(no error object)");
    exit(1);
}

static std::string readFile(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p.c_str()); exit(1); }
    std::string s; char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

// ---------------------------------------------------------------------------
// half <-> float (IEEE binary16, round-to-nearest-even)
// ---------------------------------------------------------------------------
typedef uint16_t h16;
static inline float h2f(h16 h) { __fp16 v; memcpy(&v, &h, 2); return (float)v; }
static inline h16  f2h(float f) { __fp16 v = (__fp16)f; h16 h; memcpy(&h, &v, 2); return h; }

// ---------------------------------------------------------------------------
// CPU reference. Same affine dequant, fp32 accumulate, evaluated in the
// straightforward order so it is genuinely independent of the kernels.
// ---------------------------------------------------------------------------
static void cpuRefRows(const uint32_t* W, const h16* S, const h16* B, const h16* x,
                       float* y, const GemvParams& P, uint32_t m0, uint32_t m1) {
    for (uint32_t m = m0; m < m1; ++m) {
        const uint32_t* wr = W + (size_t)m * P.words_pr;
        const h16* sr = S + (size_t)m * P.groups_pr;
        const h16* br = B + (size_t)m * P.groups_pr;
        double acc = 0.0;
        for (uint32_t k = 0; k < P.K; ++k) {
            uint32_t word = wr[k >> 3];
            uint32_t q = (word >> (4 * (k & 7))) & 0xF;
            uint32_t g = k / P.G;
            acc += ((double)h2f(sr[g]) * (double)q + (double)h2f(br[g])) * (double)h2f(x[k]);
        }
        y[m] = (float)acc;
    }
}

static void cpuRef(const uint32_t* W, const h16* S, const h16* B, const h16* x,
                   float* y, const GemvParams& P) {
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    uint32_t chunk = (P.M + n - 1) / n;
    for (unsigned i = 0; i < n; ++i) {
        uint32_t a = i * chunk, b = std::min(P.M, a + chunk);
        if (a >= b) break;
        th.emplace_back(cpuRefRows, W, S, B, x, y, std::cref(P), a, b);
    }
    for (auto& t : th) t.join();
}

struct Timing { double best_s, med_s; };

static Timing timeBuffer(id<MTLCommandQueue> q, int warmup, int iters,
                         void (^encode)(id<MTLCommandBuffer>)) {
    for (int i = 0; i < warmup; ++i) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        encode(cb);
        [cb commit]; [cb waitUntilCompleted];
    }
    std::vector<double> ts;
    ts.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        encode(cb);
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "FATAL: command buffer failed: %s\n",
                    [[cb.error localizedDescription] UTF8String]);
            exit(1);
        }
        ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
    }
    std::sort(ts.begin(), ts.end());
    return { ts.front(), ts[ts.size() / 2] };
}

int main(int argc, char** argv) { @autoreleasepool {
    uint32_t M = 8192, K = 8192, G = 64;
    int iters = 50, warmup = 10, tgsize = 256, repeat = 0;
    std::string srcPath = "src/kernels.metal";
    std::string dumpDir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return argv[++i]; };
        if      (a == "--M")      M = (uint32_t)atoi(next());
        else if (a == "--K")      K = (uint32_t)atoi(next());
        else if (a == "--G")      G = (uint32_t)atoi(next());
        else if (a == "--iters")  iters = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else if (a == "--tg")     tgsize = atoi(next());
        else if (a == "--repeat") repeat = atoi(next());
        else if (a == "--src")    srcPath = next();
        else if (a == "--dump")   dumpDir = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (K % 512 != 0) { fprintf(stderr, "K must be a multiple of 512\n"); return 1; }
    if (G % 32 != 0 || K % G != 0) { fprintf(stderr, "G must be mult of 32 and divide K\n"); return 1; }

    GemvParams P { M, K, G, K / 8, K / G, G / 32 };

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "FATAL: no Metal device\n"); return 1; }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSError* err = nil;
    MTLCompileOptions* copts = [MTLCompileOptions new];
    copts.mathMode = MTLMathModeFast;
    NSString* src = [NSString stringWithUTF8String:readFile(srcPath).c_str()];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:copts error:&err];
    if (!lib) die(err, "compiling kernels.metal");

    // ---------------- buffers ----------------
    size_t wBytes  = (size_t)M * P.words_pr * 4;
    size_t sbBytes = (size_t)M * P.groups_pr * 2;
    size_t xBytes  = (size_t)K * 2;
    size_t yBytes  = (size_t)M * 4;
    size_t xsBytes = (size_t)(K / 32) * 4;

    // Rotate over N identical copies of the weights so that consecutive timed
    // iterations never re-read what the previous one left in the 8 MB system
    // level cache. Without this a 10 MB matrix reports >100% of the pure-read
    // ceiling, because most of it is still resident. This is also the physically
    // honest model of decode: a real model streams a *different* weight matrix
    // at every layer, so no call ever finds its weights warm.
    const size_t kSLCBytes = 8ull << 20;
    const size_t kRotTarget = 24 * kSLCBytes;          // 192 MB of distinct data
    size_t setBytes = wBytes + 2 * sbBytes + 2 * sbBytes;
    uint32_t NCOPY = (uint32_t)std::min<size_t>(64,
                        std::max<size_t>(1, (kRotTarget + setBytes - 1) / setBytes));
    g_ncopy = NCOPY;

    auto mkbuf = [&](size_t n) {
        id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) { fprintf(stderr, "FATAL: alloc %zu bytes failed\n", n); exit(1); }
        return b;
    };
    id<MTLBuffer> bW = mkbuf(wBytes * NCOPY), bS = mkbuf(sbBytes * NCOPY),
                  bB = mkbuf(sbBytes * NCOPY);
    id<MTLBuffer> bX = mkbuf(xBytes), bY = mkbuf(yBytes),  bXS = mkbuf(xsBytes);
    id<MTLBuffer> bP = mkbuf(sizeof(GemvParams));
    id<MTLBuffer> bSB = mkbuf(2 * sbBytes * NCOPY); // interleaved (scale,bias) half2

    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> ud;
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);

    uint32_t* W = (uint32_t*)bW.contents;
    for (size_t i = 0; i < wBytes / 4; ++i) W[i] = ud(rng);
    h16* S = (h16*)bS.contents; h16* B = (h16*)bB.contents;
    for (size_t i = 0; i < sbBytes / 2; ++i) {
        S[i] = f2h(uf(rng) * 0.02f);
        B[i] = f2h(uf(rng) * 0.15f);
    }
    h16* SB = (h16*)bSB.contents;
    for (size_t i = 0; i < sbBytes / 2; ++i) { SB[2*i] = S[i]; SB[2*i+1] = B[i]; }
    for (uint32_t c = 1; c < NCOPY; ++c) {
        memcpy((char*)bW.contents  + (size_t)c * wBytes,      W,  wBytes);
        memcpy((char*)bS.contents  + (size_t)c * sbBytes,     S,  sbBytes);
        memcpy((char*)bB.contents  + (size_t)c * sbBytes,     B,  sbBytes);
        memcpy((char*)bSB.contents + (size_t)c * 2 * sbBytes, SB, 2 * sbBytes);
    }
    h16* X = (h16*)bX.contents;
    for (uint32_t i = 0; i < K; ++i) X[i] = f2h(uf(rng));
    memcpy(bP.contents, &P, sizeof P);

    // ---------------- pipelines ----------------
    auto pipeline = [&](NSString* name, MTLFunctionConstantValues* fc) {
        NSError* e = nil;
        id<MTLFunction> fn = fc ? [lib newFunctionWithName:name constantValues:fc error:&e]
                                : [lib newFunctionWithName:name];
        if (!fn) die(e, [[@"newFunction " stringByAppendingString:name] UTF8String]);
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) die(e, [[@"pipeline " stringByAppendingString:name] UTF8String]);
        return ps;
    };
    auto fcRows = [&](uint32_t r) {
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        [fc setConstantValue:&r type:MTLDataTypeUInt atIndex:0];
        return fc;
    };

    id<MTLComputePipelineState> psXS     = pipeline(@"xsum32", nil);
    id<MTLComputePipelineState> psScalar = pipeline(@"gemv_int4_scalar", nil);
    id<MTLComputePipelineState> psSimd   = pipeline(@"gemv_int4_simd", nil);
    auto fcAt = [&](uint32_t idx, uint32_t v) {
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        [fc setConstantValue:&v type:MTLDataTypeUInt atIndex:idx];
        return fc;
    };

    printf("device                : %s\n", [[dev name] UTF8String]);
    printf("shape                 : M=%u K=%u  group=%u  (4-bit, fp16 scale+bias)\n", M, K, G);
    printf("weight bytes          : %.2f MB  (%.4f bytes/param incl. metadata)\n",
           wBytes / 1048576.0,
           (double)(wBytes + 2 * sbBytes) / ((double)M * K));
    printf("threadgroup size      : %d threads (%d simdgroups)\n", tgsize, tgsize / 32);
    printf("cache-defeat rotation : %u identical weight copies (%.0f MB distinct)\n",
           NCOPY, (double)setBytes * NCOPY / 1048576.0);
    printf("compiled via          : MTLDevice newLibraryWithSource (no Xcode toolchain)\n\n");

    // xsum32 once (its cost is included in the per-call byte budget below
    // only as the xs read; it is a per-token prepass, benchmarked separately).
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:psXS];
        [enc setBuffer:bX offset:0 atIndex:0];
        [enc setBuffer:bXS offset:0 atIndex:1];
        [enc setBuffer:bP offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(K / 32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    }

    // ---------------- empirical bandwidth ceiling ----------------
    // ---------------- clock settle ----------------
    // A cold GPU on this machine reads ~93 GB/s and a warm one ~113 GB/s on the
    // identical kernel. Measuring before the clocks plateau would attribute a
    // DVFS ramp to whichever kernel happened to run first, so hold the memory
    // system at sustained load until the achieved rate stops improving.
    {
        id<MTLBuffer> sink = mkbuf(64);
        uint32_t nquads = (uint32_t)(wBytes / 16);
        id<MTLBuffer> bN = mkbuf(4); *(uint32_t*)bN.contents = nquads;
        id<MTLComputePipelineState> ps = pipeline(@"stream_read", fcAt(1, 1));
        NSUInteger threads = ((nquads / 4) / 256) * 256;
        double prevBest = 0.0; int stable = 0, round = 0;
        printf("settling GPU clocks   : "); fflush(stdout);
        while (round++ < 120 && (round < 20 || stable < 8)) {
            Timing t = timeBuffer(queue, 0, 12, ^(id<MTLCommandBuffer> cb) {
                g_copy = (g_copy + 1) % g_ncopy;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:bW offset:(size_t)g_copy * wBytes atIndex:0];
                [enc setBuffer:sink offset:0 atIndex:1];
                [enc setBuffer:bN offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            });
            double gbs = wBytes / t.best_s / 1e9;
            if (gbs < prevBest * 1.005) ++stable; else stable = 0;
            (void)0;
            prevBest = std::max(prevBest, gbs);
        }
        printf("plateaued at %.1f GB/s after %d rounds\n", prevBest, round);
    }

    // Sweep unroll depth x thread count and keep the best: the ceiling has to be
    // the best this machine can do on a pure read, or it is not a ceiling.
    double ceilGBs = 0.0;
    std::string ceilCfg;
    uint32_t ceilUnroll = 1; NSUInteger ceilThreads = 65536;
    {
        id<MTLBuffer> sink = mkbuf(64);
        uint32_t nquads = (uint32_t)(wBytes / 16);
        id<MTLBuffer> bN = mkbuf(4); *(uint32_t*)bN.contents = nquads;
        for (uint32_t U : {1u, 2u, 4u, 8u}) {
            id<MTLComputePipelineState> ps = pipeline(@"stream_read", fcAt(1, U));
            for (uint32_t qpt : {1u, 2u, 4u, 8u, 16u, 32u}) {
                NSUInteger threads = nquads / (qpt * U);
                if (threads < 1024) continue;
                threads = (threads / 256) * 256;
                Timing t = timeBuffer(queue, 3, std::max(8, iters / 3),
                                      ^(id<MTLCommandBuffer> cb) {
                    g_copy = (g_copy + 1) % g_ncopy;
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:bW offset:(size_t)g_copy * wBytes atIndex:0];
                    [enc setBuffer:sink offset:0 atIndex:1];
                    [enc setBuffer:bN offset:0 atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                });
                double gbs = wBytes / t.best_s / 1e9;
                if (gbs > ceilGBs) {
                    ceilGBs = gbs; ceilUnroll = U; ceilThreads = threads;
                    char c[64]; snprintf(c, sizeof c, "unroll=%u, %lu threads", U,
                                         (unsigned long)threads);
                    ceilCfg = c;
                }
            }
        }
        printf("stream_read tuning    : %7.2f GB/s  (%.1f%% of %.0f GB/s peak)  "
               "pure read of the same %.1f MB  [%s]\n\n",
               ceilGBs, 100.0 * ceilGBs / kPeakGBs, kPeakGBs,
               wBytes / 1048576.0, ceilCfg.c_str());
    }

    // ---------------- reference ----------------
    printf("computing CPU reference (%u rows x %u)...\n", M, K); fflush(stdout);
    std::vector<float> ref(M);
    cpuRef(W, S, B, X, ref.data(), P);

    // total DRAM traffic one GEMV call must move
    double bytesPerCall = (double)wBytes + 2.0 * sbBytes + xBytes + xsBytes + yBytes;

    struct Variant {
        std::string name;
        void (^enc)(id<MTLCommandBuffer>);
        double bytes;
        bool   check;
        double best_s, med_s, relerr, best_wall_s;
    };
    std::vector<Variant> vars;

    auto addGemv = [&](const std::string& name, id<MTLComputePipelineState> ps,
                       NSUInteger gridThreads, NSUInteger tgThreads,
                       NSUInteger tgMem, bool fusedMeta) {
        id<MTLBuffer> meta = fusedMeta ? bSB : bS;
        void (^enc)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            size_t mb = fusedMeta ? 2 * sbBytes : sbBytes;
            [e setBuffer:bX offset:0 atIndex:3];
            [e setBuffer:bY offset:0 atIndex:4];
            [e setBuffer:bXS offset:0 atIndex:5];
            [e setBuffer:bP offset:0 atIndex:6];
            if (tgMem) [e setThreadgroupMemoryLength:tgMem atIndex:0];
            for (int rep = 0; rep < g_repeat; ++rep) {
                g_copy = (g_copy + 1) % g_ncopy;
                [e setBuffer:bW   offset:(size_t)g_copy * wBytes  atIndex:0];
                [e setBuffer:meta offset:(size_t)g_copy * mb      atIndex:1];
                [e setBuffer:bB   offset:(size_t)g_copy * sbBytes atIndex:2];
                [e dispatchThreads:MTLSizeMake(gridThreads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(tgThreads, 1, 1)];
            }
            [e endEncoding];
        };
        vars.push_back({ name, enc, bytesPerCall, true, 0, 0, 0, 0 });
    };

    addGemv("v0 scalar (1 thread/row)", psScalar, M, 256, 0, false);
    {
        NSUInteger sgPerTg = tgsize / 32;
        NSUInteger tgs = (M + sgPerTg - 1) / sgPerTg;
        addGemv("v1 simdgroup/row, uint4 loads", psSimd, tgs * tgsize, tgsize, 0, false);
    }
    for (uint32_t R : {2u, 4u, 8u}) {
        NSUInteger rowsPerTg = (tgsize / 32) * R;
        char nm[96]; snprintf(nm, sizeof nm, "v2 multirow R=%u", R);
        addGemv(nm, pipeline(@"gemv_int4_multirow", fcRows(R)),
                ((M + rowsPerTg - 1) / rowsPerTg) * tgsize, tgsize, 0, false);
    }
    if ((size_t)K * 2 <= [dev maxThreadgroupMemoryLength]) {
        for (uint32_t R : {2u, 4u}) {
            NSUInteger rowsPerTg = (tgsize / 32) * R;
            char nm[96]; snprintf(nm, sizeof nm, "v3 multirow R=%u + x in tgmem", R);
            addGemv(nm, pipeline(@"gemv_int4_multirow_tgx", fcRows(R)),
                    ((M + rowsPerTg - 1) / rowsPerTg) * tgsize, tgsize, (size_t)K * 2, false);
        }
    }
    {
        NSUInteger sgPerTg = tgsize / 32;
        NSUInteger tgs = (M + sgPerTg - 1) / sgPerTg;
        addGemv("v5 wide x loads + xs in tgmem", pipeline(@"gemv_int4_wide", nil),
                tgs * tgsize, tgsize, (size_t)(K / 32) * 4, true);
        for (uint32_t U : {2u, 4u, 8u}) {
            char nm[96]; snprintf(nm, sizeof nm, "v6 wide + unroll %u", U);
            addGemv(nm, pipeline(@"gemv_int4_wide_unroll", fcAt(2, U)),
                    tgs * tgsize, tgsize, (size_t)(K / 32) * 4, true);
        }
    }
    for (uint32_t R : {1u, 2u, 4u}) {
        NSUInteger rowsPerTg = (tgsize / 32) * R;
        char nm[96]; snprintf(nm, sizeof nm, "v4 fused scale+bias R=%u", R);
        addGemv(nm, pipeline(@"gemv_int4_fused", fcRows(R)),
                ((M + rowsPerTg - 1) / rowsPerTg) * tgsize, tgsize, 0, true);
    }

    // the pure-read ceiling rides in the same rotation so that "% of ceiling"
    // compares two numbers measured under the same thermal and clock state
    {
        id<MTLBuffer> sink = mkbuf(64);
        id<MTLBuffer> bN = mkbuf(4); *(uint32_t*)bN.contents = (uint32_t)(wBytes / 16);
        id<MTLComputePipelineState> ps = pipeline(@"stream_read", fcAt(1, ceilUnroll));
        NSUInteger threads = ceilThreads;
        void (^enc)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:sink offset:0 atIndex:1];
            [e setBuffer:bN offset:0 atIndex:2];
            for (int rep = 0; rep < g_repeat; ++rep) {
                g_copy = (g_copy + 1) % g_ncopy;
                [e setBuffer:bW offset:(size_t)g_copy * wBytes atIndex:0];
                [e dispatchThreads:MTLSizeMake(threads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding];
        };
        vars.push_back({ "[ceiling] pure streaming read", enc, (double)wBytes, false, 0, 0, 0, 0 });
    }

    // stale dumps from an earlier shape would silently be cross-checked as if
    // they belonged to this run
    if (!dumpDir.empty()) {
        if (DIR* d = opendir(dumpDir.c_str())) {
            while (struct dirent* e = readdir(d)) {
                std::string n = e->d_name;
                if (n.rfind("y_", 0) == 0) remove((dumpDir + "/" + n).c_str());
            }
            closedir(d);
        }
    }

    // pick `repeat` from a probe so every timed span is milliseconds long
    if (repeat <= 0) {
        g_repeat = 1;
        id<MTLCommandBuffer> cb = [queue commandBuffer]; vars[1].enc(cb);
        [cb commit]; [cb waitUntilCompleted];
        double t = cb.GPUEndTime - cb.GPUStartTime;
        repeat = (int)std::min(64.0, std::max(1.0, std::ceil(5e-3 / std::max(t, 1e-6))));
    }
    g_repeat = repeat;
    printf("dispatches per buffer : %d  (timed span ~%.1f ms)\n", g_repeat, 5.0);

    // ---- correctness, once per variant, before anything is timed ----
    double refinf = 0.0;
    for (uint32_t m = 0; m < M; ++m) refinf = std::max(refinf, (double)std::fabs(ref[m]));
    for (auto& v : vars) {
        if (!v.check) continue;
        memset(bY.contents, 0, yBytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer]; v.enc(cb);
        [cb commit]; [cb waitUntilCompleted];
        const float* got = (const float*)bY.contents;
        double maxabs = 0.0;
        for (uint32_t m = 0; m < M; ++m)
            maxabs = std::max(maxabs, std::fabs((double)got[m] - ref[m]));
        v.relerr = maxabs / std::max(1e-30, refinf);
        if (v.relerr >= 2e-3) {
            printf("  %-34s MISMATCH  err/|ref|inf %.2e\n", v.name.c_str(), v.relerr);
            return 1;
        }
        if (!dumpDir.empty()) {
            std::string fn = dumpDir + "/y_";
            for (char c : v.name) fn += (isalnum(c) ? c : '_');
            fn += ".f32";
            FILE* f = fopen(fn.c_str(), "wb");
            if (f) { fwrite(got, 1, yBytes, f); fclose(f); }
        }
    }
    printf("all %zu variants verified against the CPU reference (max err/|ref|inf %.1e)\n",
           vars.size() - 1, vars[0].relerr);

    // ---- interleaved timing ----
    // Every variant is timed once per round, round-robin. Comparing variants
    // measured in separate passes attributes the GPU's clock drift to whichever
    // ran last; on this machine that drift is ~8%, larger than the differences
    // being measured.
    for (auto& v : vars)
        for (int i = 0; i < warmup; ++i) {
            id<MTLCommandBuffer> cb = [queue commandBuffer]; v.enc(cb);
            [cb commit]; [cb waitUntilCompleted];
        }
    std::vector<std::vector<double>> samples(vars.size()), walls(vars.size());
    for (int r = 0; r < iters; ++r) {
        for (size_t i = 0; i < vars.size(); ++i) {
            auto w0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer]; vars[i].enc(cb);
            [cb commit]; [cb waitUntilCompleted];
            walls[i].push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - w0).count());
            if (cb.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "FATAL: command buffer failed\n"); return 1;
            }
            samples[i].push_back(cb.GPUEndTime - cb.GPUStartTime);
        }
    }
    for (size_t i = 0; i < vars.size(); ++i) {
        std::sort(samples[i].begin(), samples[i].end());
        vars[i].best_s = samples[i].front() / g_repeat;
        vars[i].med_s  = samples[i][samples[i].size() / 2] / g_repeat;
        std::sort(walls[i].begin(), walls[i].end());
        vars[i].best_wall_s = walls[i].front() / g_repeat;
    }

    ceilGBs = vars.back().bytes / vars.back().best_s / 1e9;

    printf("\n%-36s %9s %9s %8s %8s %7s %7s\n", "variant", "GB/s", "GB/s(med)",
           "gpu ms", "wall ms", "% peak", "% ceil");
    printf("--------------------------------------------------------------------------------------\n");
    for (auto& v : vars) {
        double gbs = v.bytes / v.best_s / 1e9;
        double gmd = v.bytes / v.med_s / 1e9;
        printf("  %-34s %9.2f %9.2f %8.3f %8.3f %7.1f %7.1f\n", v.name.c_str(),
               gbs, gmd, v.best_s * 1e3, v.best_wall_s * 1e3,
               100.0 * gbs / kPeakGBs, 100.0 * gbs / ceilGBs);
    }
    printf("-------------------------------------------------------------------------\n");
    const Variant* best = nullptr;
    for (auto& v : vars)
        if (v.check && (!best || v.bytes / v.best_s > best->bytes / best->best_s)) best = &v;
    double bg = best->bytes / best->best_s / 1e9;
    printf("\nbytes moved per GEMV  : %.2f MB\n", bytesPerCall / 1048576.0);
    printf("pure-read ceiling     : %.2f GB/s (%.1f%% of peak)\n",
           ceilGBs, 100.0 * ceilGBs / kPeakGBs);
    printf("best GEMV variant     : %s\n", best->name.c_str());
    printf("achieved bandwidth    : %.2f GB/s   (%.3f ms gpu, %.3f ms wall)\n",
           bg, best->best_s * 1e3, best->best_wall_s * 1e3);
    printf("  wall-clock bandwidth: %.2f GB/s  (command-buffer submission included,\n"
           "                        amortised over %d dispatches -- this is the number\n"
           "                        bench/compare_mlx.py is directly comparable to)\n",
           best->bytes / best->best_wall_s / 1e9, g_repeat);
    printf("  %% of 120 GB/s peak  : %.1f%%\n", 100.0 * bg / kPeakGBs);
    printf("  %% of stream ceiling : %.1f%%\n", 100.0 * bg / ceilGBs);

    if (!dumpDir.empty()) {
        auto dump = [&](const char* n, const void* p, size_t b) {
            std::string path = dumpDir + "/" + n;
            FILE* f = fopen(path.c_str(), "wb");
            if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
            fwrite(p, 1, b, f); fclose(f);
        };
        dump("W.u32", bW.contents, wBytes);
        dump("S.f16", bS.contents, sbBytes);
        dump("B.f16", bB.contents, sbBytes);
        dump("x.f16", bX.contents, xBytes);
        dump("xs.f32", bXS.contents, xsBytes);
        dump("y_gpu.f32", bY.contents, yBytes);
        dump("y_cpu.f32", ref.data(), yBytes);
        FILE* f = fopen((dumpDir + "/shape.txt").c_str(), "w");
        fprintf(f, "%u %u %u\n", M, K, G); fclose(f);
        printf("\ndumped tensors to %s for the MLX cross-check\n", dumpDir.c_str());
    }
    return 0;
}}
