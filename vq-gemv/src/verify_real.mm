// Run the VQ kernel on real quantised model weights and write its output.
//
// The perplexity harness measures the quantiser by dequantising to fp16 and
// using the stock forward pass. That is only a valid stand-in for the kernel if
// the kernel actually computes the same thing on the same tensors, which is what
// this tool exists to establish: it loads the PQ artefacts that
// bench/verify_real.py produced from a real Qwen weight matrix, runs
// gemv_vq_shared on them, and writes y for that script to compare against MLX.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct VQParams { uint32_t M, K, chunks_pr, cb_size; };

static std::vector<char> slurp(const std::string& p, size_t want) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", p.c_str()); exit(1); }
    std::vector<char> v(want);
    size_t n = fread(v.data(), 1, want, f);
    fclose(f);
    if (n != want) { fprintf(stderr, "FATAL: %s has %zu bytes, expected %zu\n",
                             p.c_str(), n, want); exit(1); }
    return v;
}

int main(int argc, char** argv) { @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: verify_real <dir>\n"); return 1; }
    std::string d = argv[1];

    uint32_t M = 0, K = 0;
    { FILE* f = fopen((d + "/meta.txt").c_str(), "r");
      if (!f || fscanf(f, "%u %u", &M, &K) != 2) {
          fprintf(stderr, "FATAL: bad %s/meta.txt\n", d.c_str()); return 1; }
      fclose(f); }
    VQParams P { M, K, K / 32, 256 };

    auto IDX = slurp(d + "/idx.u8",  (size_t)M * (K / 2));
    auto C   = slurp(d + "/cb.f16",  256 * 2 * 2);
    auto S   = slurp(d + "/scale.f16", (size_t)M * (K / 32) * 2);
    auto X   = slurp(d + "/x.f16",   (size_t)K * 2);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* e = nil;
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.mathMode = MTLMathModeFast;
    std::string src;
    { FILE* f = fopen("src/kernels.metal", "rb");
      if (!f) { fprintf(stderr, "FATAL: run from the project root\n"); return 1; }
      char buf[65536]; size_t n;
      while ((n = fread(buf, 1, sizeof buf, f)) > 0) src.append(buf, n);
      fclose(f); }
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:src.c_str()] options:o error:&e];
    if (!lib) { fprintf(stderr, "FATAL: %s\n", [[e localizedDescription] UTF8String]); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"gemv_vq_shared"] error:&e];
    if (!ps) { fprintf(stderr, "FATAL: %s\n", [[e localizedDescription] UTF8String]); return 1; }

    auto buf = [&](const void* p, size_t n) {
        return [dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> bI = buf(IDX.data(), IDX.size()), bC = buf(C.data(), C.size());
    id<MTLBuffer> bS = buf(S.data(), S.size()),     bX = buf(X.data(), X.size());
    id<MTLBuffer> bY = [dev newBufferWithLength:(size_t)M * 4
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bP = buf(&P, sizeof P);

    NSUInteger tg = 256, sg = tg / 32, ntg = (M + sg - 1) / sg;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:bI offset:0 atIndex:0]; [enc setBuffer:bC offset:0 atIndex:1];
    [enc setBuffer:bS offset:0 atIndex:2]; [enc setBuffer:bX offset:0 atIndex:3];
    [enc setBuffer:bY offset:0 atIndex:4]; [enc setBuffer:bP offset:0 atIndex:5];
    [enc setThreadgroupMemoryLength:256 * 4 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(ntg * tg, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "FATAL: %s\n", [[cb.error localizedDescription] UTF8String]); return 1; }

    FILE* f = fopen((d + "/y_metal.f32").c_str(), "wb");
    fwrite(bY.contents, 1, (size_t)M * 4, f); fclose(f);
    printf("ran gemv_vq_shared on real weights: M=%u K=%u -> %s/y_metal.f32\n", M, K, d.c_str());
    return 0;
}}
