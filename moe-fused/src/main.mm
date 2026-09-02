// Fully-fused MoE decode block vs the same block unfused.
//
// Both paths run in this harness, against the same 128 real gpt-oss-120b
// experts held resident, and the harness counts its own dispatches, so the
// launch-count comparison is a measurement rather than a claim. Outputs are
// checked against an fp64 CPU reference before either is timed.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static const size_t BLOB = 14024704;
static const size_t OFF_GATE = 0, OFF_UP = 4671360, OFF_DOWN = 9342720;
static const size_t SUB_W = 0, SUB_S = 4147200, SUB_B = 4406400, SUB_BI = 4665600;

struct MoEParams {
    uint32_t hidden, inter, n_experts, top_k;
    uint32_t quads_hid, quads_int, groups_hid, groups_int, chunks_pg;
    uint64_t blob, off_gate, off_up, off_down, sub_w, sub_s, sub_b, sub_bi;
};
struct GemvJob { uint32_t slot_i, M, K, quads, groups; uint64_t proj; uint32_t has_bias; };

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void die(const char* w, NSError* e = nil) {
    fprintf(stderr, "FATAL: %s%s%s\n", w, e ? ": " : "",
            e ? [[e localizedDescription] UTF8String] : "");
    exit(1);
}
static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

// fp64 CPU reference for one expert projection
static void refProj(const uint8_t* blob, size_t proj, const double* x,
                    double* y, uint32_t M, uint32_t K, uint32_t m0, uint32_t m1) {
    const uint32_t* W = (const uint32_t*)(blob + proj + SUB_W);
    const uint16_t* S = (const uint16_t*)(blob + proj + SUB_S);
    const uint16_t* B = (const uint16_t*)(blob + proj + SUB_B);
    const uint16_t* BI = (const uint16_t*)(blob + proj + SUB_BI);
    uint32_t wpr = K / 8, gpr = K / 64;
    for (uint32_t m = m0; m < m1; ++m) {
        double a = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t q = (W[(size_t)m*wpr + (k>>3)] >> (4*(k&7))) & 0xF;
            a += ((double)bf2f(S[(size_t)m*gpr + k/64]) * q
                + (double)bf2f(B[(size_t)m*gpr + k/64])) * x[k];
        }
        y[m] = a + (double)bf2f(BI[m]);
    }
}
template <class F> static void par(uint32_t M, F f) {
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th; uint32_t c = (M + n - 1) / n;
    for (unsigned i = 0; i < n; ++i) {
        uint32_t a = i*c, b = std::min(M, a+c);
        if (a >= b) break;
        th.emplace_back(f, a, b);
    }
    for (auto& t : th) t.join();
}

int main(int argc, char** argv) { @autoreleasepool {
    std::string store = std::string(getenv("HOME")) + "/Desktop/moe-stream/model-120b/experts.bin";
    uint32_t HID = 2880, INTER = 2880, NEXP = 128, TOPK = 4;
    int iters = 30, warmup = 8, tgsize = 256, layers = 36, repeat = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() -> const char* { return argv[++i]; };
        if      (a == "--store")   store = nx();
        else if (a == "--experts") NEXP = (uint32_t)atoi(nx());
        else if (a == "--iters")   iters = atoi(nx());
        else if (a == "--warmup")  warmup = atoi(nx());
        else if (a == "--repeat")  repeat = atoi(nx());
        else if (a == "--layers")  layers = atoi(nx());
        else die(("unknown arg " + a).c_str());
    }

    MoEParams P { HID, INTER, NEXP, TOPK, HID/32, INTER/32, HID/64, INTER/64, 2,
                  BLOB, OFF_GATE, OFF_UP, OFF_DOWN, SUB_W, SUB_S, SUB_B, SUB_BI };

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* co = [MTLCompileOptions new]; co.mathMode = MTLMathModeFast;
    std::string src; {
        FILE* f = fopen("src/kernels.metal", "rb");
        if (!f) die("run from the project root");
        char b[65536]; size_t n;
        while ((n = fread(b,1,sizeof b,f)) > 0) src.append(b,n);
        fclose(f);
    }
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:src.c_str()] options:co error:&err];
    if (!lib) die("compiling kernels.metal", err);
    uint32_t qunr = (uint32_t)(getenv("QUNR") ? atoi(getenv("QUNR")) : 1);
    MTLFunctionConstantValues* fcq = [MTLFunctionConstantValues new];
    [fcq setConstantValue:&qunr type:MTLDataTypeUInt atIndex:0];
    auto PS = [&](NSString* n) {
        NSError* e = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:n constantValues:fcq error:&e];
        if (!fn) fn = [lib newFunctionWithName:n];
        if (!fn) die([n UTF8String], e);
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
        if (!p) die([n UTF8String], e);
        return p;
    };
    id<MTLComputePipelineState> psRouterTopk = PS(@"router_topk");
    id<MTLComputePipelineState> psGUS = PS(@"fused_gate_up_swiglu");
    id<MTLComputePipelineState> psDC  = PS(@"fused_down_combine");
    id<MTLComputePipelineState> psGUSa = PS(@"fused_gate_up_swiglu_at");
    id<MTLComputePipelineState> psDCa  = PS(@"fused_down_combine_at");
    id<MTLComputePipelineState> psURouter = PS(@"u_router"), psUTopk = PS(@"u_topk");
    id<MTLComputePipelineState> psUSm = PS(@"u_softmax_k"), psUXs = PS(@"u_xsum");
    id<MTLComputePipelineState> psUGemv = PS(@"u_gemv"), psUSwi = PS(@"u_swiglu");
    id<MTLComputePipelineState> psUAxpy = PS(@"u_axpy");

    // ---- resident expert pool ----
    size_t poolBytes = (size_t)NEXP * BLOB;
    id<MTLBuffer> pool = [dev newBufferWithLength:poolBytes options:MTLResourceStorageModeShared];
    if (!pool) die("expert pool alloc");
    {
        int fd = open(store.c_str(), O_RDONLY);
        if (fd < 0) die(("cannot open " + store).c_str());
        printf("loading %u experts (%.2f GB) ...", NEXP, poolBytes / 1e9); fflush(stdout);
        double t0 = now_s();
        for (uint32_t e = 0; e < NEXP; ++e) {
            size_t got = 0;
            while (got < BLOB) {
                ssize_t n = pread(fd, (char*)pool.contents + (size_t)e*BLOB + got,
                                  BLOB - got, (off_t)((size_t)e * BLOB) + got);
                if (n <= 0) die("pread");
                got += (size_t)n;
            }
        }
        close(fd);
        printf(" %.1f s\n", now_s() - t0);
    }

    auto mk = [&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> bR = mk((size_t)NEXP*HID*4), bX = mk(HID*4), bY = mk(HID*4);
    id<MTLBuffer> bH = mk((size_t)TOPK*INTER*2), bIdx = mk(TOPK*4), bWgt = mk(TOPK*4);
    id<MTLBuffer> bXh = mk((size_t)HID*2);   // fp16 copy of x, written by pass 1
    id<MTLBuffer> bLg = mk((size_t)NEXP*4), bSel = mk(TOPK*4), bP = mk(sizeof(MoEParams));
    id<MTLBuffer> bXsX = mk((HID/32)*4), bXsH = mk((size_t)TOPK*(INTER/32)*4);
    id<MTLBuffer> bG = mk(INTER*4), bU = mk(INTER*4);
    memcpy(bP.contents, &P, sizeof P);

    // Router weights in a sane range. The first version of this line did
    // `% 2000 - 1000` in unsigned arithmetic, which wraps to ~4e9 and produced
    // logits around 9e13 -- large enough that fp32 could not resolve the 3rd and
    // 4th ranked experts and the top-k order differed from the fp64 reference.
    // The kernels were correct; the test data was not.
    float* R = (float*)bR.contents;
    for (size_t i = 0; i < (size_t)NEXP*HID; ++i) {
        uint32_t h = (uint32_t)(i * 1103515245u + 12345u) >> 16;
        R[i] = 0.02f * ((int)(h % 2001) - 1000) / 1000.0f;
    }
    float* X = (float*)bX.contents;
    for (uint32_t i = 0; i < HID; ++i) X[i] = 0.05f * ((int)(i % 17) - 8);

    int nDispatch = 0;   // counted, not asserted
    NSUInteger SG = tgsize / 32;

    auto encodeFused = [&](id<MTLCommandBuffer> cb, bool count, bool atomicVar) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:psRouterTopk];
        [e setBuffer:bR offset:0 atIndex:0]; [e setBuffer:bX offset:0 atIndex:1];
        [e setBuffer:bIdx offset:0 atIndex:2]; [e setBuffer:bWgt offset:0 atIndex:3];
        [e setBuffer:bP offset:0 atIndex:4]; [e setBuffer:bXsH offset:0 atIndex:5];
        [e setBuffer:bXh offset:0 atIndex:6];
        [e setThreadgroupMemoryLength:NEXP*4 atIndex:0];
        [e dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        if (count) ++nDispatch;

        NSUInteger rows = (NSUInteger)TOPK * INTER, ntg = (rows + SG - 1) / SG;
        [e setComputePipelineState:(atomicVar ? psGUSa : psGUS)];
        [e setBuffer:pool offset:0 atIndex:0]; [e setBuffer:bIdx offset:0 atIndex:1];
        [e setBuffer:bXh offset:0 atIndex:2];  [e setBuffer:bH offset:0 atIndex:3];
        [e setBuffer:bP offset:0 atIndex:4];   [e setBuffer:bXsH offset:0 atIndex:5];
        [e setThreadgroupMemoryLength:(HID/32)*4 atIndex:0];
        [e dispatchThreads:MTLSizeMake(ntg*tgsize,1,1) threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)];
        if (count) ++nDispatch;

        NSUInteger ntg2 = (HID + SG - 1) / SG;
        [e setComputePipelineState:(atomicVar ? psDCa : psDC)];
        [e setBuffer:pool offset:0 atIndex:0]; [e setBuffer:bIdx offset:0 atIndex:1];
        [e setBuffer:bWgt offset:0 atIndex:2]; [e setBuffer:bH offset:0 atIndex:3];
        [e setBuffer:bY offset:0 atIndex:4];   [e setBuffer:bP offset:0 atIndex:5];
        [e setBuffer:bXsH offset:0 atIndex:6];
        [e setThreadgroupMemoryLength:TOPK*(INTER/32)*4 atIndex:0];
        [e dispatchThreads:MTLSizeMake(ntg2*tgsize,1,1) threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)];
        if (count) ++nDispatch;
        [e endEncoding];
    };

    auto encodeUnfused = [&](id<MTLCommandBuffer> cb, bool count) {
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        auto bump = [&]{ if (count) ++nDispatch; };
        // the fp16 activation copy is produced by the router pass in the fused
        // path; the unfused path gets it from the same router kernel
        [e setComputePipelineState:psURouter];
        [e setBuffer:bR offset:0 atIndex:0]; [e setBuffer:bX offset:0 atIndex:1];
        [e setBuffer:bLg offset:0 atIndex:2]; [e setBuffer:bP offset:0 atIndex:3];
        [e setBuffer:bXh offset:0 atIndex:4]; [e setBuffer:bXsH offset:0 atIndex:5];
        { NSUInteger n = (NEXP + SG - 1) / SG;
          [e dispatchThreads:MTLSizeMake(n*tgsize,1,1)
                threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)]; } bump();

        [e setComputePipelineState:psUTopk];
        [e setBuffer:bLg offset:0 atIndex:0]; [e setBuffer:bIdx offset:0 atIndex:1];
        [e setBuffer:bSel offset:0 atIndex:2]; [e setBuffer:bP offset:0 atIndex:3];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; bump();

        [e setComputePipelineState:psUSm];
        [e setBuffer:bSel offset:0 atIndex:0]; [e setBuffer:bWgt offset:0 atIndex:1];
        [e setBuffer:bP offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; bump();

        uint32_t nqh = HID/32, nqi = INTER/32;
        [e setComputePipelineState:psUXs];
        [e setBuffer:bXh offset:0 atIndex:0]; [e setBuffer:bXsX offset:0 atIndex:1];
        [e setBytes:&nqh length:4 atIndex:2];
        [e dispatchThreads:MTLSizeMake(nqh,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; bump();

        auto gemv = [&](uint32_t slot, uint64_t proj, uint32_t M, uint32_t K,
                        id<MTLBuffer> xin, id<MTLBuffer> xsin, id<MTLBuffer> yout) {
            GemvJob J { slot, M, K, K/32, K/64, proj, 1 };
            [e setComputePipelineState:psUGemv];
            [e setBuffer:pool offset:0 atIndex:0]; [e setBuffer:bIdx offset:0 atIndex:1];
            [e setBuffer:xin offset:0 atIndex:2];  [e setBuffer:xsin offset:0 atIndex:3];
            [e setBuffer:yout offset:0 atIndex:4]; [e setBuffer:bP offset:0 atIndex:5];
            [e setBytes:&J length:sizeof J atIndex:6];
            NSUInteger n = (M + SG - 1) / SG;
            [e dispatchThreads:MTLSizeMake(n*tgsize,1,1) threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)];
            bump();
        };
        for (uint32_t s = 0; s < TOPK; ++s) {
            gemv(s, OFF_GATE, INTER, HID, bXh, bXsX, bG);
            gemv(s, OFF_UP,   INTER, HID, bXh, bXsX, bU);
            uint32_t n = INTER;
            [e setComputePipelineState:psUSwi];
            [e setBuffer:bG offset:0 atIndex:0]; [e setBuffer:bU offset:0 atIndex:1];
            [e setBuffer:bH offset:(size_t)s*INTER*2 atIndex:2];
            [e setBytes:&n length:4 atIndex:3];
            [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; bump();
        }
        for (uint32_t s = 0; s < TOPK; ++s) {
            [e setComputePipelineState:psUXs];
            [e setBuffer:bH offset:(size_t)s*INTER*2 atIndex:0];
            [e setBuffer:bXsH offset:(size_t)s*nqi*4 atIndex:1];
            [e setBytes:&nqi length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(nqi,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; bump();

            GemvJob J { s, HID, INTER, INTER/32, INTER/64, OFF_DOWN, 1 };
            [e setComputePipelineState:psUGemv];
            [e setBuffer:pool offset:0 atIndex:0]; [e setBuffer:bIdx offset:0 atIndex:1];
            [e setBuffer:bH offset:(size_t)s*INTER*2 atIndex:2];
            [e setBuffer:bXsH offset:(size_t)s*nqi*4 atIndex:3];
            [e setBuffer:bG offset:0 atIndex:4]; [e setBuffer:bP offset:0 atIndex:5];
            [e setBytes:&J length:sizeof J atIndex:6];
            NSUInteger n = (HID + SG - 1) / SG;
            [e dispatchThreads:MTLSizeMake(n*tgsize,1,1) threadsPerThreadgroup:MTLSizeMake(tgsize,1,1)]; bump();

            uint32_t nh = HID;
            [e setComputePipelineState:psUAxpy];
            [e setBuffer:bG offset:0 atIndex:0]; [e setBuffer:bWgt offset:0 atIndex:1];
            [e setBuffer:bY offset:0 atIndex:2];
            [e setBytes:&nh length:4 atIndex:3]; [e setBytes:&s length:4 atIndex:4];
            [e dispatchThreads:MTLSizeMake(nh,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; bump();
        }
        [e endEncoding];
    };

    printf("device                : %s\n", [[dev name] UTF8String]);
    printf("block                 : hidden=%u inter=%u  %u experts, top-%u  "
           "(gpt-oss-120b, int4 g=64)\n", HID, INTER, NEXP, TOPK);
    printf("resident experts      : %.2f GB\n", poolBytes / 1e9);
    printf("weights read / block  : %.2f MB  (top-%u x gate+up+down, decimal MB)\n",
           (TOPK * 3.0 * 4147200 + TOPK * 6.0 * 259200) / 1e6, TOPK);

    // ---- count dispatches ----
    nDispatch = 0; { id<MTLCommandBuffer> cb = [queue commandBuffer]; encodeFused(cb, true, false);
                     [cb commit]; [cb waitUntilCompleted]; }
    int nFused = nDispatch;
    nDispatch = 0; { id<MTLCommandBuffer> cb = [queue commandBuffer]; encodeUnfused(cb, true);
                     [cb commit]; [cb waitUntilCompleted]; }
    int nUnfused = nDispatch;

    // ---- correctness vs fp64 CPU reference ----
    const uint8_t* poolp = (const uint8_t*)pool.contents;
    std::vector<double> xd(HID); for (uint32_t i=0;i<HID;++i) xd[i]=X[i];
    std::vector<double> logits(NEXP);
    for (uint32_t e2 = 0; e2 < NEXP; ++e2) {
        double s = 0; const float* r = R + (size_t)e2*HID;
        for (uint32_t k = 0; k < HID; ++k) s += (double)r[k]*xd[k];
        logits[e2] = s;
    }
    std::vector<uint32_t> ridx(TOPK); std::vector<double> rsel(TOPK), lg2 = logits;
    for (uint32_t j = 0; j < TOPK; ++j) {
        uint32_t bi=0; double bv=-1e300;
        for (uint32_t e2=0;e2<NEXP;++e2) if (lg2[e2]>bv){bv=lg2[e2];bi=e2;}
        ridx[j]=bi; rsel[j]=bv; lg2[bi]=-1e300;
    }
    double mx = *std::max_element(rsel.begin(), rsel.end()), sm = 0;
    std::vector<double> rw(TOPK);
    for (uint32_t j=0;j<TOPK;++j){ rw[j]=exp(rsel[j]-mx); sm+=rw[j]; }
    for (uint32_t j=0;j<TOPK;++j) rw[j]/=sm;

    std::vector<double> yref(HID, 0.0);
    for (uint32_t j = 0; j < TOPK; ++j) {
        const uint8_t* bb = poolp + (size_t)ridx[j]*BLOB;
        std::vector<double> g(INTER), u(INTER), h(INTER), yy(HID);
        par(INTER, [&](uint32_t a,uint32_t b){ refProj(bb, OFF_GATE, xd.data(), g.data(), INTER, HID, a, b); });
        par(INTER, [&](uint32_t a,uint32_t b){ refProj(bb, OFF_UP,   xd.data(), u.data(), INTER, HID, a, b); });
        for (uint32_t i=0;i<INTER;++i){
            double gg=std::min(g[i],7.0), ll=std::max(-7.0,std::min(u[i],7.0));
            h[i]=(gg/(1.0+exp(-1.702*gg)))*(ll+1.0);
        }
        par(HID, [&](uint32_t a,uint32_t b){ refProj(bb, OFF_DOWN, h.data(), yy.data(), HID, INTER, a, b); });
        for (uint32_t i=0;i<HID;++i) yref[i] += rw[j]*yy[i];
    }
    double yinf = 0; for (double v : yref) yinf = std::max(yinf, std::fabs(v));

    auto check = [&](const char* name, void (^enc)(id<MTLCommandBuffer>)) {
        memset(bY.contents, 0, HID*4);
        id<MTLCommandBuffer> cb = [queue commandBuffer]; enc(cb);
        [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) die(name, cb.error);
        const float* y = (const float*)bY.contents;
        const uint32_t* gi = (const uint32_t*)bIdx.contents;
        bool idxok = true;
        for (uint32_t j=0;j<TOPK;++j) if (gi[j]!=ridx[j]) idxok=false;
        if (!idxok) {
            const float* gw = (const float*)bWgt.contents;
            printf("      gpu idx/wgt:");
            for (uint32_t j=0;j<TOPK;++j) printf(" %u:%.6f", gi[j], gw[j]);
            printf("\n      ref idx/wgt:");
            for (uint32_t j=0;j<TOPK;++j) printf(" %u:%.6f", ridx[j], rw[j]);
            printf("\n      ref logits of those:");
            for (uint32_t j=0;j<TOPK;++j) printf(" %.9f", rsel[j]);
            printf("\n");
        }
        double e2 = 0;
        for (uint32_t i=0;i<HID;++i) e2 = std::max(e2, std::fabs((double)y[i]-yref[i]));
        double rel = e2 / yinf;
        if (false) {
            const float* xsh = (const float*)bXsH.contents;
            const __fp16* hh = (const __fp16*)bH.contents;
            printf("      xsh[0..3] = %g %g %g %g\n", xsh[0], xsh[1], xsh[2], xsh[3]);
            printf("      h[0..5] = %g %g %g %g %g %g\n", (float)hh[0],(float)hh[1],
                   (float)hh[2],(float)hh[3],(float)hh[4],(float)hh[5]);
            { double t2=0; for (int i=0;i<32;++i) t2 += (double)hh[i];
              printf("      sum h[0..31]=%g  sum h[0..3]=%g\n", t2,
                     (double)hh[0]+hh[1]+hh[2]+hh[3]); }
            for (int e2 = 0; e2 < 2; ++e2) {
                double t = 0; for (int i = 0; i < 32; ++i) t += (double)hh[(size_t)e2*INTER + i];
                printf("      true sum(h[e=%d,chunk0]) = %g   xsh = %g\n",
                       e2, t, xsh[e2 * (INTER/32)]);
            }
        }
        printf("  %-22s top-k %s   max|d|/|y|inf = %.2e  %s\n", name,
               idxok ? "match" : "MISMATCH", rel, (idxok && rel < 2e-3) ? "ok" : "FAIL");
        return idxok && rel < 2e-3;
    };
    printf("\ncorrectness vs fp64 CPU reference (router, top-k, both matmuls, SwiGLU)\n");
    bool okf = check("fused (3 kernels)", ^(id<MTLCommandBuffer> cb){ encodeFused(cb,false,false); });
    bool oka = check("fused, atomic sum(h)", ^(id<MTLCommandBuffer> cb){ encodeFused(cb,false,true); });
    bool oku = check("unfused (27 kernels)", ^(id<MTLCommandBuffer> cb){ encodeUnfused(cb,false); });
    if (!okf || !oku || !oka) return 1;

    auto timeit = [&](void (^enc)(id<MTLCommandBuffer>)) {
        for (int i=0;i<warmup;++i){ id<MTLCommandBuffer> cb=[queue commandBuffer];
            for (int r=0;r<repeat;++r) enc(cb); [cb commit]; [cb waitUntilCompleted]; }
        double best = 1e30;
        for (int i=0;i<iters;++i){
            double t0 = now_s();
            id<MTLCommandBuffer> cb=[queue commandBuffer];
            for (int r=0;r<repeat;++r) enc(cb);
            [cb commit]; [cb waitUntilCompleted];
            best = std::min(best, (now_s()-t0)/repeat);
        }
        return best;
    };
    { FILE* f = fopen("results/y_metal.f32", "wb");
      if (f) { fwrite(bY.contents, 1, HID*4, f); fclose(f); }
      FILE* g2 = fopen("results/topk.txt", "w");
      if (g2) { const uint32_t* gi=(const uint32_t*)bIdx.contents;
                const float* gw=(const float*)bWgt.contents;
                for (uint32_t j=0;j<TOPK;++j) fprintf(g2,"%u %.9f\n",gi[j],gw[j]);
                fclose(g2); } }

    double tF = timeit(^(id<MTLCommandBuffer> cb){ encodeFused(cb,false,false); });
    double tA = timeit(^(id<MTLCommandBuffer> cb){ encodeFused(cb,false,true); });
    double tU = timeit(^(id<MTLCommandBuffer> cb){ encodeUnfused(cb,false); });

    double mb = TOPK * 3.0 * 4147200 + TOPK * 6.0 * 259200;
    printf("\n%-24s %10s %12s %12s %12s\n", "path", "launches", "ms/block",
           "GB/s", "tok/s");
    printf("------------------------------------------------------------------------\n");
    printf("  %-22s %10d %12.4f %12.2f %12.1f\n", "unfused", nUnfused, tU*1e3,
           mb/tU/1e9, 1.0/(layers*tU));
    printf("  %-22s %10d %12.4f %12.2f %12.1f\n", "fused", nFused, tF*1e3,
           mb/tF/1e9, 1.0/(layers*tF));
    printf("  %-22s %10d %12.4f %12.2f %12.1f\n", "fused, atomic sum(h)", nFused,
           tA*1e3, mb/tA/1e9, 1.0/(layers*tA));
    printf("------------------------------------------------------------------------\n");
    printf("\nlaunches per block    : %d -> %d  (%.1fx fewer)\n", nUnfused, nFused,
           (double)nUnfused / nFused);
    printf("launches per token    : %d -> %d  (%d layers)\n",
           nUnfused*layers, nFused*layers, layers);
    double tBest = std::min(tF, tA);
    printf("block time            : %.4f -> %.4f ms  (%.3fx)\n", tU*1e3, tBest*1e3, tU/tBest);
    printf("decode rate           : %.1f -> %.1f tok/s  (experts resident, MoE only)\n",
           1.0/(layers*tU), 1.0/(layers*tBest));
    return 0;
}}
