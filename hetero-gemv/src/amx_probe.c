// Does cblas_sgemv actually reach the CPU's matrix coprocessor?
//
// There is no userspace way to ask, so compare against cblas_sgemm on the same
// operand: if Accelerate routes GEMM to the matrix units and GEMV to ordinary
// SIMD, GEMM will show an order of magnitude more FLOP/s. Measured on this M4:
//
//     sgemv  0.180 ms    46.6 GFLOP/s
//     sgemm  0.612 ms   877.2 GFLOP/s   (n=64)
//
// 19x. The matrix hardware is plainly there and plainly not being used for the
// matrix-vector case -- which is the expected answer, since GEMV touches each
// weight once and a matrix engine has no reuse to exploit.
//
//     cc -O2 src/amx_probe.c -framework Accelerate -o amx_probe && ./amx_probe
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(){
  int M=2048,K=2048;
  float*A=aligned_alloc(64,(size_t)M*K*4),*B=aligned_alloc(64,(size_t)K*64*4),*C=aligned_alloc(64,(size_t)M*64*4);
  for(size_t i=0;i<(size_t)M*K;i++)A[i]=0.001f*(i%13);
  for(size_t i=0;i<(size_t)K*64;i++)B[i]=0.5f;
  double b1=1e9,b2=1e9;
  for(int r=0;r<30;r++){double t=now();cblas_sgemv(CblasRowMajor,CblasNoTrans,M,K,1,A,K,B,1,0,C,1);t=now()-t;if(t<b1)b1=t;}
  for(int r=0;r<30;r++){double t=now();cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,64,K,1,A,K,B,64,0,C,64);t=now()-t;if(t<b2)b2=t;}
  printf("sgemv  %.3f ms  %6.1f GFLOP/s\n",b1*1e3,2.0*M*K/b1/1e9);
  printf("sgemm  %.3f ms  %6.1f GFLOP/s  (n=64)\n",b2*1e3,2.0*M*K*64/b2/1e9);
  return 0;}
