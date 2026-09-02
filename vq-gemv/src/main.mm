// Benchmark harness for fused codebook (VQ) dequant-matmul vs scalar int4.
//
// Measurement protocol is the one established in ../int4-gemv, and it is not
// optional: without the clock settle, the cache-defeat rotation, the dispatch
// batching and the interleaved timing, this machine reports differences between
// formats that are really differences in GPU thermal state.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const double kPeakGBs = 120.0;   // M4 base: LPDDR5X-7500, 128-bit bus

struct VQParams   { uint32_t M, K, chunks_pr, cb_size; };
struct Int4Params { uint32_t M, K, G, words_pr, groups_pr, chunks_pg; };

static uint32_t g_copy = 0, g_ncopy = 1;
static int g_repeat = 1;

typedef uint16_t h16;
static inline float h2f(h16 h) { __fp16 v; memcpy(&v, &h, 2); return (float)v; }
static inline h16  f2h(float f) { __fp16 v = (__fp16)f; h16 h; memcpy(&h, &v, 2); return h; }

static void die(NSError* e, const char* what) {
    fprintf(stderr, "FATAL: %s: %s\n", what,
            e ? [[e localizedDescription] UTF8String] : "(none)");
    exit(1);
}
static std::string readFile(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p.c_str()); exit(1); }
    std::string s; char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f); return s;
}

// ---------------------------------------------------------------------------
// CPU references, fp64 accumulate, one thread per row block.
// ---------------------------------------------------------------------------
static void vqRefRows(const uint8_t* IDX, const h16* C, const h16* S, const h16* x,
                      float* y, VQParams P, uint32_t m0, uint32_t m1) {
    for (uint32_t m = m0; m < m1; ++m) {
        const uint8_t* ir = IDX + (size_t)m * (P.K / 2);
        const h16* sr = S + (size_t)m * P.chunks_pr;
        double acc = 0.0;
        for (uint32_t i = 0; i < P.K / 2; ++i) {
            uint32_t c = ir[i];
            double s = h2f(sr[i / 16]);
            acc += s * ((double)h2f(C[2*c]) * (double)h2f(x[2*i])
                      + (double)h2f(C[2*c+1]) * (double)h2f(x[2*i+1]));
        }
        y[m] = (float)acc;
    }
}
static void int4RefRows(const uint32_t* W, const h16* SB, const h16* x,
                        float* y, Int4Params P, uint32_t m0, uint32_t m1) {
    for (uint32_t m = m0; m < m1; ++m) {
        const uint32_t* wr = W + (size_t)m * P.words_pr;
        const h16* sb = SB + (size_t)m * P.groups_pr * 2;
        double acc = 0.0;
        for (uint32_t k = 0; k < P.K; ++k) {
            uint32_t q = (wr[k >> 3] >> (4 * (k & 7))) & 0xF;
            uint32_t g = k / P.G;
            acc += ((double)h2f(sb[2*g]) * (double)q + (double)h2f(sb[2*g+1]))
                 * (double)h2f(x[k]);
        }
        y[m] = (float)acc;
    }
}
template <class F> static void parRows(uint32_t M, F f) {
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th; uint32_t chunk = (M + n - 1) / n;
    for (unsigned i = 0; i < n; ++i) {
        uint32_t a = i * chunk, b = std::min(M, a + chunk);
        if (a >= b) break;
        th.emplace_back(f, a, b);
    }
    for (auto& t : th) t.join();
}

int main(int argc, char** argv) { @autoreleasepool {
    uint32_t M = 16384, K = 16384, G = 64, CB = 256;
    int iters = 15, warmup = 5, tgsize = 256, repeat = 0;
    std::string srcPath = "src/kernels.metal", dumpDir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return argv[++i]; };
        if      (a == "--M")      M = (uint32_t)atoi(next());
        else if (a == "--K")      K = (uint32_t)atoi(next());
        else if (a == "--iters")  iters = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else if (a == "--tg")     tgsize = atoi(next());
        else if (a == "--repeat") repeat = atoi(next());
        else if (a == "--src")    srcPath = next();
        else if (a == "--dump")   dumpDir = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (K % 512) { fprintf(stderr, "K must be a multiple of 512\n"); return 1; }

    VQParams   PV { M, K, K / 32, CB };
    Int4Params PI { M, K, G, K / 8, K / G, G / 32 };

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "FATAL: no Metal device\n"); return 1; }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSError* err = nil;
    MTLCompileOptions* copts = [MTLCompileOptions new];
    copts.mathMode = MTLMathModeFast;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:readFile(srcPath).c_str()] options:copts error:&err];
    if (!lib) die(err, "compiling kernels.metal");

    // ---- sizes. the two formats are the same number of bytes by construction --
    size_t idxBytes = (size_t)M * (K / 2);          // VQ: uint8 per weight pair
    size_t vsBytes  = (size_t)M * (K / 32) * 2;     // VQ: fp16 scale per 32
    size_t cbBytes  = (size_t)CB * 4;               // VQ: 256 x half2
    size_t wBytes   = (size_t)M * (K / 8) * 4;      // int4: packed nibbles
    size_t sbBytes  = (size_t)M * (K / G) * 2 * 2;  // int4: half2 (scale,bias)
    size_t xBytes   = (size_t)K * 2, yBytes = (size_t)M * 4;
    size_t xsBytes  = (size_t)(K / 32) * 4;

    double vqStream  = (double)(idxBytes + vsBytes) + xBytes + yBytes + cbBytes;
    double i4Stream  = (double)(wBytes + sbBytes) + xBytes + xsBytes + yBytes;

    const size_t kSLC = 8ull << 20, kRot = 24 * kSLC;
    size_t setBytes = std::max(idxBytes + vsBytes, wBytes + sbBytes);
    uint32_t NCOPY = (uint32_t)std::min<size_t>(64,
                        std::max<size_t>(1, (kRot + setBytes - 1) / setBytes));
    g_ncopy = NCOPY;

    auto mkbuf = [&](size_t n) {
        id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) { fprintf(stderr, "FATAL: alloc %zu bytes failed\n", n); exit(1); }
        return b;
    };
    id<MTLBuffer> bIDX = mkbuf(idxBytes * NCOPY), bVS = mkbuf(vsBytes * NCOPY);
    id<MTLBuffer> bC   = mkbuf(cbBytes * 32);   // room for the replicated copy
    id<MTLBuffer> bW   = mkbuf(wBytes * NCOPY), bSB = mkbuf(sbBytes * NCOPY);
    id<MTLBuffer> bX   = mkbuf(xBytes), bY = mkbuf(yBytes), bXS = mkbuf(xsBytes);
    id<MTLBuffer> bPV  = mkbuf(sizeof(VQParams)), bPI = mkbuf(sizeof(Int4Params));

    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> ud;
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);

    uint8_t*  IDX = (uint8_t*)bIDX.contents;
    for (size_t i = 0; i < idxBytes; ++i) IDX[i] = (uint8_t)(ud(rng) & 0xFF);
    h16* C = (h16*)bC.contents;
    for (uint32_t c = 0; c < CB; ++c) { C[2*c] = f2h(uf(rng)); C[2*c+1] = f2h(uf(rng)); }
    h16* VS = (h16*)bVS.contents;
    for (size_t i = 0; i < vsBytes / 2; ++i) VS[i] = f2h(uf(rng) * 0.05f);
    uint32_t* W = (uint32_t*)bW.contents;
    for (size_t i = 0; i < wBytes / 4; ++i) W[i] = ud(rng);
    h16* SB = (h16*)bSB.contents;
    for (size_t i = 0; i < sbBytes / 4; ++i) {
        SB[2*i] = f2h(uf(rng) * 0.02f); SB[2*i+1] = f2h(uf(rng) * 0.15f);
    }
    h16* X = (h16*)bX.contents;
    for (uint32_t i = 0; i < K; ++i) X[i] = f2h(uf(rng));
    for (uint32_t c = 1; c < NCOPY; ++c) {
        memcpy((char*)bIDX.contents + (size_t)c * idxBytes, IDX, idxBytes);
        memcpy((char*)bVS.contents  + (size_t)c * vsBytes,  VS,  vsBytes);
        memcpy((char*)bW.contents   + (size_t)c * wBytes,   W,   wBytes);
        memcpy((char*)bSB.contents  + (size_t)c * sbBytes,  SB,  sbBytes);
    }
    memcpy(bPV.contents, &PV, sizeof PV);
    memcpy(bPI.contents, &PI, sizeof PI);

    auto pipeline = [&](NSString* name, MTLFunctionConstantValues* fc) {
        NSError* e = nil;
        id<MTLFunction> fn = fc ? [lib newFunctionWithName:name constantValues:fc error:&e]
                                : [lib newFunctionWithName:name];
        if (!fn) die(e, [name UTF8String]);
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
        if (!ps) die(e, [name UTF8String]);
        return ps;
    };
    auto fcAt = [&](uint32_t idx, uint32_t v) {
        MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
        [fc setConstantValue:&v type:MTLDataTypeUInt atIndex:idx];
        return fc;
    };

    printf("device                : %s\n", [[dev name] UTF8String]);
    printf("shape                 : M=%u K=%u\n", M, K);
    printf("VQ   d=2 K=%u         : %.4f bytes/param  (%.2f MB idx + %.2f MB scale)\n",
           CB, (double)(idxBytes + vsBytes) / ((double)M * K),
           idxBytes / 1048576.0, vsBytes / 1048576.0);
    printf("int4 affine g=%u      : %.4f bytes/param  (%.2f MB w  + %.2f MB scale/bias)\n",
           G, (double)(wBytes + sbBytes) / ((double)M * K),
           wBytes / 1048576.0, sbBytes / 1048576.0);
    printf("threadgroup size      : %d threads (%d simdgroups)\n", tgsize, tgsize / 32);
    printf("max threadgroup mem   : %lu bytes (replicated codebook needs %zu)\n",
           (unsigned long)[dev maxThreadgroupMemoryLength], (size_t)CB * 32 * 4);
    printf("cache-defeat rotation : %u copies (%.0f MB distinct)\n\n",
           NCOPY, (double)setBytes * NCOPY / 1048576.0);

    // ---- xsum32 prepass for the int4 opponent ----
    {
        id<MTLComputePipelineState> ps = pipeline(@"xsum32", nil);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps];
        [e setBuffer:bX offset:0 atIndex:0]; [e setBuffer:bXS offset:0 atIndex:1];
        [e setBuffer:bPI offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(K / 32, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }

    // ---- settle GPU clocks ----
    id<MTLBuffer> sink = mkbuf(64);
    id<MTLBuffer> bN = mkbuf(4); *(uint32_t*)bN.contents = (uint32_t)(idxBytes / 16);
    {
        id<MTLComputePipelineState> ps = pipeline(@"stream_read", fcAt(1, 1));
        NSUInteger threads = ((idxBytes / 16 / 4) / 256) * 256;
        double prevBest = 0; int stable = 0, round = 0;
        printf("settling GPU clocks   : "); fflush(stdout);
        while (round++ < 120 && (round < 20 || stable < 8)) {
            std::vector<double> ts;
            for (int i = 0; i < 12; ++i) {
                g_copy = (g_copy + 1) % g_ncopy;
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:bIDX offset:(size_t)g_copy * idxBytes atIndex:0];
                [e setBuffer:sink offset:0 atIndex:1]; [e setBuffer:bN offset:0 atIndex:2];
                [e dispatchThreads:MTLSizeMake(threads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
            }
            double gbs = idxBytes / *std::min_element(ts.begin(), ts.end()) / 1e9;
            if (gbs < prevBest * 1.005) ++stable; else stable = 0;
            prevBest = std::max(prevBest, gbs);
        }
        printf("plateaued at %.1f GB/s after %d rounds\n\n", prevBest, round);
    }

    // ---- CPU references ----
    printf("computing CPU references (%u rows x %u, both formats)...\n", M, K); fflush(stdout);
    std::vector<float> refVQ(M), refI4(M);
    parRows(M, [&](uint32_t a, uint32_t b) { vqRefRows(IDX, C, VS, X, refVQ.data(), PV, a, b); });
    parRows(M, [&](uint32_t a, uint32_t b) { int4RefRows(W, SB, X, refI4.data(), PI, a, b); });

    struct Variant {
        std::string name; void (^enc)(id<MTLCommandBuffer>);
        double bytes; int refsel;          // 0 = VQ ref, 1 = int4 ref, -1 = none
        double best_s, med_s, relerr, best_wall_s;
    };
    std::vector<Variant> vars;

    auto addVQ = [&](const std::string& name, id<MTLComputePipelineState> ps,
                     NSUInteger grid, NSUInteger tg, NSUInteger tgMem) {
        void (^enc)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:bC offset:0 atIndex:1];
            [e setBuffer:bX offset:0 atIndex:3];
            [e setBuffer:bY offset:0 atIndex:4];
            [e setBuffer:bPV offset:0 atIndex:5];
            if (tgMem) [e setThreadgroupMemoryLength:tgMem atIndex:0];
            for (int r = 0; r < g_repeat; ++r) {
                g_copy = (g_copy + 1) % g_ncopy;
                [e setBuffer:bIDX offset:(size_t)g_copy * idxBytes atIndex:0];
                [e setBuffer:bVS  offset:(size_t)g_copy * vsBytes  atIndex:2];
                [e dispatchThreads:MTLSizeMake(grid, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            }
            [e endEncoding];
        };
        vars.push_back({ name, enc, vqStream, 0, 0, 0, 0, 0 });
    };

    NSUInteger sgPerTg = tgsize / 32;
    addVQ("VQ v0 scalar (1 thread/row)", pipeline(@"gemv_vq_scalar", nil), M, 256, 0);
    addVQ("VQ v1 shared codebook (1 KB)", pipeline(@"gemv_vq_shared", nil),
          ((M + sgPerTg - 1) / sgPerTg) * tgsize, tgsize, cbBytes);
    addVQ("VQ v3 codebook in device mem", pipeline(@"gemv_vq_device", nil),
          ((M + sgPerTg - 1) / sgPerTg) * tgsize, tgsize, 0);
    // Persistent threadgroups: enough to fill the GPU, then grid-stride over
    // rows, so the codebook staging cost is paid once per threadgroup rather
    // than once per 8 rows. REP is swept so that the effect of replication is
    // separated from the effect of running persistently: REP=1 here is the same
    // codebook as v1, only with persistent threadgroups.
    for (uint32_t R : {1u, 4u, 8u, 32u}) {
        size_t tgm = (size_t)CB * R * 4;
        if (tgm > [dev maxThreadgroupMemoryLength]) continue;
        id<MTLComputePipelineState> ps = pipeline(@"gemv_vq_replicated", fcAt(0, R));
        NSUInteger ntg = std::min<NSUInteger>((M + sgPerTg - 1) / sgPerTg, 240);
        char nm[96];
        snprintf(nm, sizeof nm, "VQ v2 persistent, %2u codebook copies", R);
        addVQ(nm, ps, ntg * tgsize, tgsize, tgm);
    }
    for (uint32_t U : {2u, 4u}) {
        NSUInteger tgs = (M + sgPerTg - 1) / sgPerTg;
        char nm[96]; snprintf(nm, sizeof nm, "VQ v5 wide + unroll %u", U);
        addVQ(nm, pipeline(@"gemv_vq_wide", fcAt(2, U)), tgs * tgsize, tgsize, cbBytes);
    }
    {
        id<MTLComputePipelineState> ps = pipeline(@"gemv_int4_wide_unroll", fcAt(2, 4));
        NSUInteger tgs = (M + sgPerTg - 1) / sgPerTg;
        void (^enc)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:bX offset:0 atIndex:3]; [e setBuffer:bY offset:0 atIndex:4];
            [e setBuffer:bXS offset:0 atIndex:5]; [e setBuffer:bPI offset:0 atIndex:6];
            [e setThreadgroupMemoryLength:(size_t)(K/32)*4 atIndex:0];
            for (int r = 0; r < g_repeat; ++r) {
                g_copy = (g_copy + 1) % g_ncopy;
                [e setBuffer:bW  offset:(size_t)g_copy * wBytes  atIndex:0];
                [e setBuffer:bSB offset:(size_t)g_copy * sbBytes atIndex:1];
                [e dispatchThreads:MTLSizeMake(tgs * tgsize, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(tgsize, 1, 1)];
            }
            [e endEncoding];
        };
        vars.push_back({ "int4 wide+unroll (same bytes)", enc, i4Stream, 1, 0, 0, 0, 0 });
    }
    {
        id<MTLComputePipelineState> ps = pipeline(@"gemv_int4_fused", nil);
        void (^enc)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:bX offset:0 atIndex:3]; [e setBuffer:bY offset:0 atIndex:4];
            [e setBuffer:bXS offset:0 atIndex:5]; [e setBuffer:bPI offset:0 atIndex:6];
            for (int r = 0; r < g_repeat; ++r) {
                g_copy = (g_copy + 1) % g_ncopy;
                [e setBuffer:bW  offset:(size_t)g_copy * wBytes  atIndex:0];
                [e setBuffer:bSB offset:(size_t)g_copy * sbBytes atIndex:1];
                [e dispatchThreads:MTLSizeMake(((M + sgPerTg - 1) / sgPerTg) * tgsize, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(tgsize, 1, 1)];
            }
            [e endEncoding];
        };
        vars.push_back({ "int4 affine (same bytes/param)", enc, i4Stream, 1, 0, 0, 0, 0 });
    }
    {
        id<MTLComputePipelineState> ps = pipeline(@"stream_read", fcAt(1, 1));
        NSUInteger threads = ((idxBytes / 16 / 4) / 256) * 256;
        void (^enc)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:sink offset:0 atIndex:1]; [e setBuffer:bN offset:0 atIndex:2];
            for (int r = 0; r < g_repeat; ++r) {
                g_copy = (g_copy + 1) % g_ncopy;
                [e setBuffer:bIDX offset:(size_t)g_copy * idxBytes atIndex:0];
                [e dispatchThreads:MTLSizeMake(threads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e endEncoding];
        };
        vars.push_back({ "[ceiling] pure streaming read", enc, (double)idxBytes, -1, 0, 0, 0, 0 });
    }

    if (repeat <= 0) {
        g_repeat = 1;
        double t = 1e9;
        for (int i = 0; i < 3; ++i) {          // first dispatch is cold
            id<MTLCommandBuffer> cb = [queue commandBuffer]; vars[1].enc(cb);
            [cb commit]; [cb waitUntilCompleted];
            t = std::min(t, cb.GPUEndTime - cb.GPUStartTime);
        }
        repeat = (int)std::min(64.0, std::max(1.0, std::ceil(5e-3 / std::max(t, 1e-6))));
    }
    g_repeat = repeat;
    printf("dispatches per buffer : %d\n", g_repeat);

    if (!dumpDir.empty())
        if (DIR* d = opendir(dumpDir.c_str())) {
            while (struct dirent* e = readdir(d)) {
                std::string n = e->d_name;
                if (n.rfind("y_", 0) == 0) remove((dumpDir + "/" + n).c_str());
            }
            closedir(d);
        }

    for (auto& v : vars) {
        if (v.refsel < 0) continue;
        const std::vector<float>& ref = v.refsel ? refI4 : refVQ;
        double refinf = 0;
        for (uint32_t m = 0; m < M; ++m) refinf = std::max(refinf, (double)std::fabs(ref[m]));
        memset(bY.contents, 0, yBytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer]; v.enc(cb);
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            die(cb.error, ("running " + v.name).c_str());
        const float* got = (const float*)bY.contents;
        double maxabs = 0;
        for (uint32_t m = 0; m < M; ++m)
            maxabs = std::max(maxabs, std::fabs((double)got[m] - ref[m]));
        v.relerr = maxabs / std::max(1e-30, refinf);
        if (v.relerr >= 2e-3) {
            printf("  %-34s MISMATCH err/|ref|inf %.2e\n", v.name.c_str(), v.relerr);
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
    printf("all variants verified against their CPU reference\n");

    for (auto& v : vars)
        for (int i = 0; i < warmup; ++i) {
            id<MTLCommandBuffer> cb = [queue commandBuffer]; v.enc(cb);
            [cb commit]; [cb waitUntilCompleted];
        }
    std::vector<std::vector<double>> samples(vars.size()), walls(vars.size());
    for (int r = 0; r < iters; ++r)
        for (size_t i = 0; i < vars.size(); ++i) {
            auto w0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer]; vars[i].enc(cb);
            [cb commit]; [cb waitUntilCompleted];
            walls[i].push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - w0).count());
            samples[i].push_back(cb.GPUEndTime - cb.GPUStartTime);
        }
    for (size_t i = 0; i < vars.size(); ++i) {
        std::sort(samples[i].begin(), samples[i].end());
        std::sort(walls[i].begin(), walls[i].end());
        vars[i].best_s = samples[i].front() / g_repeat;
        vars[i].med_s  = samples[i][samples[i].size() / 2] / g_repeat;
        vars[i].best_wall_s = walls[i].front() / g_repeat;
    }
    double ceilGBs = vars.back().bytes / vars.back().best_s / 1e9;

    printf("\n%-34s %9s %9s %8s %7s %7s %9s\n", "variant", "GB/s", "GB/s(med)",
           "gpu ms", "% peak", "% ceil", "err");
    printf("---------------------------------------------------------------------------------------\n");
    for (auto& v : vars) {
        double gbs = v.bytes / v.best_s / 1e9;
        printf("  %-32s %9.2f %9.2f %8.3f %7.1f %7.1f %9.1e\n", v.name.c_str(),
               gbs, v.bytes / v.med_s / 1e9, v.best_s * 1e3,
               100.0 * gbs / kPeakGBs, 100.0 * gbs / ceilGBs,
               v.refsel < 0 ? 0.0 : v.relerr);
    }
    printf("---------------------------------------------------------------------------------------\n");

    const Variant* bestVQ = nullptr; const Variant* i4 = nullptr;
    for (auto& v : vars) {
        if (v.refsel == 0 && (!bestVQ || v.best_s < bestVQ->best_s)) bestVQ = &v;
        if (v.refsel == 1 && (!i4 || v.best_s < i4->best_s)) i4 = &v;
    }
    printf("\nbest VQ kernel        : %s\n", bestVQ->name.c_str());
    printf("  %.2f GB/s  (%.1f%% of peak, %.1f%% of ceiling)  %.3f ms\n",
           bestVQ->bytes / bestVQ->best_s / 1e9,
           100.0 * bestVQ->bytes / bestVQ->best_s / 1e9 / kPeakGBs,
           100.0 * bestVQ->bytes / bestVQ->best_s / 1e9 / ceilGBs,
           bestVQ->best_s * 1e3);
    printf("int4 at equal bytes   : %.2f GB/s  (%.1f%% of peak)  %.3f ms\n",
           i4->bytes / i4->best_s / 1e9,
           100.0 * i4->bytes / i4->best_s / 1e9 / kPeakGBs, i4->best_s * 1e3);
    printf("VQ / int4 throughput  : %.3fx   (both %.4f bytes/param)\n",
           i4->best_s / bestVQ->best_s, (double)(idxBytes + vsBytes) / ((double)M * K));
    return 0;
}}
