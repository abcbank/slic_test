#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>


#ifndef BYTE
#define BYTE unsigned char
#endif
__device__ int getRand(curandState *s, int A, int B);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void initMeanWithCuda(BYTE *mean, int*counter, BYTE *img, const int mean_size, const int cur_means, dim3 Info, dim3 block,dim3 thread);
void Cycle1(BYTE *src, BYTE *dst, BYTE *means,int*counter, BYTE *label, dim3 size, int mean_num, dim3 block, dim3 thread);
__global__ void k_means(BYTE *src, BYTE *dst, BYTE *means,int*counter, BYTE *label, dim3 size, int mean_num);
__global__ void addKernel(int *c, const int *a, const int *b);

__global__ void initMeanValue(BYTE *mean, int*counter, BYTE *img, int mean_size, const int cur_means, dim3 Info);