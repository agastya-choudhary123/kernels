// Heterogeneous int4 decode GEMV: Metal GPU and the CPU's vector/matrix units
// working on ONE matmul at the same time, over unified memory with no copies.
//
// The GPU takes rows [0, split) and the CPU takes rows [split, M). Both read the
// same MTLBuffer contents through the same physical addresses -- the CPU pointer
// is literally buffer.contents -- so the split costs no transfer at all.
//
// What makes this interesting rather than obvious: on this machine the two
// engines share one 120 GB/s memory system. Splitting a *bandwidth*-bound kernel
// cannot add bandwidth. It can only help if the second engine is limited by
// something other than the bus, which for int4 the CPU is: dequantising nibbles
// is arithmetic, and the CPU's share of the rows costs it far fewer bytes than
// the GPU's ~89% of peak is already extracting.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "cpu_gemv.h"

static const double kPeakGBs = 120.0;

struct GemvParams { uint32_t M, K, G, words_pr, groups_pr, chunks_pg; };

static inline h16 f2h_(float f) { __fp16 v = (__fp16)f; h16 h; memcpy(&h, &v, 2); return h; }

static void die(NSError* e, const char* what) {
    fprintf(stderr, "FATAL: %s: %s\n", what,
            e ? [[e localizedDescription] UTF8String] : "(none)");
    exit(1);
}
static std::string readFile(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p.c_str()); exit(1); }
    std::string s; char b[65536]; size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f); return s;
}

// fp64 reference, independent of both production paths
static void refRows(const uint32_t* W, const h16* SB, const h16* x, float* y,
                    GemvParams P, uint32_t m0, uint32_t m1) {
    for (uint32_t m = m0; m < m1; ++m) {
        const uint32_t* wr = W + (size_t)m * P.words_pr;
        const h16* sb = SB + (size_t)m * P.groups_pr * 2;
        double acc = 0.0;
        for (uint32_t k = 0; k < P.K; ++k) {
            uint32_t q = (wr[k >> 3] >> (4 * (k & 7))) & 0xF;
            uint32_t g = k / P.G;
            acc += ((double)h2f_(sb[2*g]) * (double)q + (double)h2f_(sb[2*g+1]))
                 * (double)h2f_(x[k]);
        }
        y[m] = (float)acc;
    }
}

int main(int argc, char** argv) { @autoreleasepool {
    uint32_t M = 16384, K = 16384, G = 64;
    int iters = 12, warmup = 4, tgsize = 256, blk = 32, nsteps = 21, repeat = 8;
    std::string srcPath = "src/kernels.metal", cpuMode = "neon";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return argv[++i]; };
        if      (a == "--M")      M = (uint32_t)atoi(next());
        else if (a == "--K")      K = (uint32_t)atoi(next());
        else if (a == "--iters")  iters = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else if (a == "--tg")     tgsize = atoi(next());
        else if (a == "--block")  blk = atoi(next());
        else if (a == "--steps")  nsteps = atoi(next());
        else if (a == "--repeat") repeat = atoi(next());
        else if (a == "--cpu")    cpuMode = next();
        else if (a == "--src")    srcPath = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    GemvParams P { M, K, G, K / 8, K / G, G / 32 };

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* copts = [MTLCompileOptions new];
    copts.mathMode = MTLMathModeFast;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:readFile(srcPath).c_str()] options:copts error:&err];
    if (!lib) die(err, "compiling kernels.metal");

    size_t wBytes = (size_t)M * P.words_pr * 4;
    size_t sbBytes = (size_t)M * P.groups_pr * 4;      // half2 per group
    size_t xBytes = (size_t)K * 2, yBytes = (size_t)M * 4;
    size_t xsBytes = (size_t)(K / 32) * 4;
    double bytesTotal = (double)wBytes + sbBytes + xBytes + xsBytes + yBytes;

    // Same cache-defeat rotation as ../int4-gemv: without it a 32 MB weight
    // matrix partly lives in the 8 MB SLC across repeats and every engine looks
    // faster than the bus allows. It matters more here, because the CPU has its
    // own 4 MB L2 per cluster and its slice of the rows is small enough to sit in
    // it entirely.
    const size_t kSLC = 8ull << 20, kRot = 24 * kSLC;
    size_t setBytes = wBytes + sbBytes;
    uint32_t NCOPY = (uint32_t)std::min<size_t>(64,
                        std::max<size_t>(1, (kRot + setBytes - 1) / setBytes));

    auto mkbuf = [&](size_t n) {
        id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) { fprintf(stderr, "FATAL: alloc %zu failed\n", n); exit(1); }
        return b;
    };
    id<MTLBuffer> bW = mkbuf(wBytes * NCOPY), bSB = mkbuf(sbBytes * NCOPY),
                  bX = mkbuf(xBytes);
    id<MTLBuffer> bY = mkbuf(yBytes), bXS = mkbuf(xsBytes), bP = mkbuf(sizeof(GemvParams));

    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> ud;
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    uint32_t* W = (uint32_t*)bW.contents;
    for (size_t i = 0; i < wBytes / 4; ++i) W[i] = ud(rng);
    h16* SB = (h16*)bSB.contents;
    for (size_t i = 0; i < sbBytes / 4; ++i) {
        SB[2*i] = f2h_(uf(rng) * 0.02f); SB[2*i+1] = f2h_(uf(rng) * 0.15f);
    }
    h16* X = (h16*)bX.contents;
    for (uint32_t i = 0; i < K; ++i) X[i] = f2h_(uf(rng));
    for (uint32_t c = 1; c < NCOPY; ++c) {
        memcpy((char*)bW.contents  + (size_t)c * wBytes,  W,  wBytes);
        memcpy((char*)bSB.contents + (size_t)c * sbBytes, SB, sbBytes);
    }
    memcpy(bP.contents, &P, sizeof P);

    // fp32 view of x for the CPU paths, and the per-32 partial sums both sides use
    std::vector<float> xf(K);
    for (uint32_t i = 0; i < K; ++i) xf[i] = h2f_(X[i]);
    float* xs = (float*)bXS.contents;
    for (uint32_t j = 0; j < K / 32; ++j) {
        float s = 0; for (uint32_t t = 0; t < 32; ++t) s += xf[j*32 + t];
        xs[j] = s;
    }

    auto pipeline = [&](NSString* n, MTLFunctionConstantValues* fc) {
        NSError* e = nil;
        id<MTLFunction> f = fc ? [lib newFunctionWithName:n constantValues:fc error:&e]
                               : [lib newFunctionWithName:n];
        if (!f) die(e, [n UTF8String]);
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:f error:&e];
        if (!ps) die(e, [n UTF8String]);
        return ps;
    };
    // v6 from ../int4-gemv: wide x loads, xsum staged in threadgroup memory,
    // weight loads unrolled 4x. Worth ~5% over the fused-half2 kernel this used
    // to call, and the GPU half of a heterogeneous split should obviously be the
    // best GPU kernel available -- otherwise the CPU is being credited for
    // headroom the GPU simply was not using.
    MTLFunctionConstantValues* fc4 = [MTLFunctionConstantValues new];
    uint32_t four = 4; [fc4 setConstantValue:&four type:MTLDataTypeUInt atIndex:2];
    id<MTLComputePipelineState> psGPU = pipeline(@"gemv_int4_wide_unroll", fc4);

    const uint8_t* Wb = (const uint8_t*)bW.contents;
    float* Y = (float*)bY.contents;

    unsigned ncpu = std::max(1u, std::thread::hardware_concurrency());
    printf("device                : %s\n", [[dev name] UTF8String]);
    printf("shape                 : M=%u K=%u group=%u  (%.2f MB weights)\n",
           M, K, G, wBytes / 1048576.0);
    printf("unified memory        : GPU and CPU both read buffer.contents, no copies\n");
    printf("cpu path              : %s, %u threads (4P + 6E)\n", cpuMode.c_str(), ncpu);
    printf("bytes per full GEMV   : %.2f MB\n", bytesTotal / 1048576.0);
    printf("GEMVs per timed batch : %d (amortises command-buffer submission)\n", repeat);
    printf("cache-defeat rotation : %u copies (%.0f MB distinct)\n\n",
           NCOPY, (double)setBytes * NCOPY / 1048576.0);

    // ---- per-work-item fp32 scratch for the Accelerate path ----
    // One per dispatch_apply chunk, not one per core: the work is
    // over-decomposed for P/E balance, so chunk indices run past the core count.
    const size_t kMaxChunks = (size_t)ncpu * 6;
    std::vector<std::vector<float>> scratch(cpuMode == "amx" ? kMaxChunks : 0);
    for (auto& sc : scratch) sc.assign((size_t)blk * K, 0.0f);
    if (cpuMode == "amx")
        printf("amx scratch           : %zu blocks x %d rows x %u = %.0f MB "
               "(sized to stay in L2)\n\n", kMaxChunks, blk, K,
               (double)kMaxChunks * blk * K * 4 / 1048576.0);

    // GPU on rows [0, split); a private params copy carries the row count
    id<MTLBuffer> bPg = mkbuf(sizeof(GemvParams));
    __block uint32_t g_split = M;
    auto encodeGPU = [&](id<MTLCommandBuffer> cb, uint32_t split, uint32_t cp) {
        if (split == 0) return;
        GemvParams Q = P; Q.M = split;
        memcpy(bPg.contents, &Q, sizeof Q);
        NSUInteger sg = tgsize / 32, tgs = (split + sg - 1) / sg;
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psGPU];
        [e setBuffer:bW  offset:(size_t)cp * wBytes  atIndex:0];
        [e setBuffer:bSB offset:(size_t)cp * sbBytes atIndex:1];
        [e setBuffer:bSB offset:(size_t)cp * sbBytes atIndex:2];
        [e setBuffer:bX offset:0 atIndex:3];
        [e setBuffer:bY offset:0 atIndex:4];  [e setBuffer:bXS offset:0 atIndex:5];
        [e setBuffer:bPg offset:0 atIndex:6];
        [e setThreadgroupMemoryLength:(size_t)(K / 32) * 4 atIndex:0];
        [e dispatchThreads:MTLSizeMake(tgs * tgsize, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tgsize, 1, 1)];
        [e endEncoding];
    };

    dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    auto runCPU = [&](uint32_t m0, uint32_t m1, uint32_t cp) {
        if (m1 <= m0) return;
        uint32_t rows = m1 - m0;
        // Several work items per thread, not one. This machine has 4 performance
        // and 6 efficiency cores, and equal-size chunks make the E-cores the
        // critical path: the P-cores finish and idle while the slowest E-core
        // still holds a full share. Over-decomposing lets GCD hand the fast cores
        // more chunks. The chunks stay contiguous row ranges so each one is still
        // a long sequential run through the weight stream.
        size_t nchunk = std::min<size_t>(kMaxChunks, std::max<size_t>(1, rows / 8));
        dispatch_apply(nchunk, dq, ^(size_t c) {
            uint32_t a = m0 + (uint32_t)((uint64_t)rows * c / nchunk);
            uint32_t b = m0 + (uint32_t)((uint64_t)rows * (c + 1) / nchunk);
            const uint8_t* Wc = Wb + (size_t)cp * wBytes;
            const h16*     Sc = SB + (size_t)cp * (sbBytes / 2);
            if (cpuMode == "amx")
                cpu_gemv_amx(Wc, Sc, xf.data(), Y, scratch[c].data(), (uint32_t)blk,
                             K, G, a, b);
            else if (cpuMode == "neon16")
                cpu_gemv_neon16(Wc, Sc, (const __fp16*)X, xs, Y, K, G, a, b);
            else
                cpu_gemv_neon(Wc, Sc, xf.data(), xs, Y, K, G, a, b);
        });
    };

    // `rep` full GEMVs at a given split, timed wall-clock end to end.
    //
    // Batching matters here more than anywhere else in these projects: one
    // command-buffer submission costs ~0.25-0.8 ms wall, which is larger than the
    // entire GEMV being measured, and it would be charged to the GPU side of every
    // split. The R GPU dispatches go into ONE command buffer and the CPU makes R
    // passes over its rows, so the overlap is preserved and the submission is
    // amortised exactly the way it would be inside a real decode loop.
    auto runSplit = [&](uint32_t split, int rep) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int r = 0; r < rep; ++r) encodeGPU(cb, split, (uint32_t)(r % NCOPY));
        [cb commit];
        for (int r = 0; r < rep; ++r) runCPU(split, M, (uint32_t)(r % NCOPY));
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted && split > 0)
            die(cb.error, "gpu half");
    };

    // ---- correctness: every split must reproduce the fp64 reference ----
    printf("computing fp64 reference (%u x %u)...\n", M, K); fflush(stdout);
    std::vector<float> ref(M);
    {
        unsigned n = ncpu; std::vector<std::thread> th;
        uint32_t chunk = (M + n - 1) / n;
        for (unsigned i = 0; i < n; ++i) {
            uint32_t a = i * chunk, b = std::min(M, a + chunk);
            if (a >= b) break;
            th.emplace_back(refRows, W, SB, X, ref.data(), P, a, b);
        }
        for (auto& t : th) t.join();
    }
    double refinf = 0;
    for (uint32_t m = 0; m < M; ++m) refinf = std::max(refinf, (double)std::fabs(ref[m]));
    for (double f : {0.0, 0.5, 0.85, 1.0}) {
        uint32_t sp = (uint32_t)(f * M);
        memset(Y, 0, yBytes);
        runSplit(sp, 1);
        double e = 0;
        for (uint32_t m = 0; m < M; ++m) e = std::max(e, std::fabs((double)Y[m] - ref[m]));
        double rel = e / refinf;
        printf("  split %5.2f (gpu rows %6u, cpu rows %6u)  err/|ref|inf %.2e  %s\n",
               f, sp, M - sp, rel, rel < 2e-3 ? "ok" : "MISMATCH");
        if (rel >= 2e-3) return 1;
    }
    printf("\n");

    auto timeSplit = [&](uint32_t split) {
        for (int i = 0; i < warmup; ++i) runSplit(split, repeat);
        double best = 1e30;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            runSplit(split, repeat);
            best = std::min(best, std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count() / repeat);
        }
        return best;
    };

    // settle clocks on both engines before any of it counts
    for (int i = 0; i < 12; ++i) { runSplit(M, repeat); runSplit(0, repeat); }

    printf("%8s %8s %8s %11s %11s %9s\n",
           "gpu frac", "gpu rows", "cpu rows", "ms", "GB/s", "% peak");
    printf("---------------------------------------------------------------------\n");
    double bestT = 1e30, bestF = 0, gpuOnly = 0, cpuOnly = 0;
    std::vector<std::pair<double,double>> curve;
    for (int s = 0; s < nsteps; ++s) {
        double f = (double)s / (nsteps - 1);
        uint32_t split = (uint32_t)std::llround(f * M);
        double t = timeSplit(split);
        double gbs = bytesTotal / t / 1e9;
        curve.push_back({f, t});
        if (split == 0) cpuOnly = t;
        if (split == M) gpuOnly = t;
        if (t < bestT) { bestT = t; bestF = f; }
        printf("%8.3f %8u %8u %11.3f %11.2f %9.1f\n",
               f, split, M - split, t * 1e3, gbs, 100.0 * gbs / kPeakGBs);
    }
    printf("---------------------------------------------------------------------\n");

    // Load balance predicted from measured throughput: if the GPU clears rows at
    // r_g = M/t_gpu and the CPU at r_c = M/t_cpu, the split that finishes both at
    // the same instant is M * r_g / (r_g + r_c). That prediction assumes the two
    // rates survive being run at once, which on a shared memory bus they do not,
    // so the gap between it and the measured optimum is the cost of contention.
    double rg = M / gpuOnly, rc = M / cpuOnly;
    double predF = rg / (rg + rc);
    printf("\nGPU only              : %8.3f ms  %7.2f GB/s\n", gpuOnly * 1e3,
           bytesTotal / gpuOnly / 1e9);
    printf("CPU only (%-4s)       : %8.3f ms  %7.2f GB/s\n", cpuMode.c_str(),
           cpuOnly * 1e3, bytesTotal / cpuOnly / 1e9);
    printf("best combined         : %8.3f ms  %7.2f GB/s   at gpu fraction %.3f\n",
           bestT * 1e3, bytesTotal / bestT / 1e9, bestF);
    printf("speedup vs GPU only   : %.4fx\n", gpuOnly / bestT);
    printf("split predicted from measured throughput: %.3f   (measured optimum %.3f)\n",
           predF, bestF);
    printf("ideal if perfectly additive: %.4fx  (1/(1/t_gpu + 1/t_cpu) = %.3f ms)\n",
           gpuOnly / (1.0 / (1.0 / gpuOnly + 1.0 / cpuOnly)),
           1e3 / (1.0 / gpuOnly + 1.0 / cpuOnly));
    return 0;
}}
