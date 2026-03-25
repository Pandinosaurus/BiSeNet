
#ifndef _KERNELS_HPP_
#define _KERNELS_HPP_

#include <iostream>
#include <functional>
#include <algorithm>
#include <cfloat>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "NvInfer.h"



#define BLOCKSIZE 256



template<typename scalar_t>
__forceinline__ __device__
void broadcast_block_x(scalar_t& val, int src_id) {
    __shared__ scalar_t shm;
    if (threadIdx.x == src_id) shm = val;
    __syncthreads();
    val = shm;
}

template<typename scalar_t>
__forceinline__ __device__
scalar_t shfl_down_sync_func(scalar_t val, uint32_t delta) {
    return __shfl_down_sync(0xffffffff, val, delta);
}

template<>
__forceinline__ __device__
int8_t shfl_down_sync_func(int8_t val, uint32_t delta) {
    int32_t ival = static_cast<int32_t>(val);
    ival = __shfl_down_sync(0xffffffff, ival, delta);
    return static_cast<int8_t>(ival);
}

template<>
__forceinline__ __device__
__half shfl_down_sync_func(__half val, uint32_t delta) {
    float fval = __half2float(val);
    fval = __shfl_down_sync(0xffffffff, fval, delta);
    return __float2half(fval);
}

template<>
__forceinline__ __device__
__nv_bfloat16 shfl_down_sync_func(__nv_bfloat16 val, uint32_t delta) {
    float fval = __bfloat162float(val);
    fval = __shfl_down_sync(0xffffffff, fval, delta);
    return __float2bfloat16(fval);
}

template<typename scalar_t>
__forceinline__ __device__
void max_pair_shfl_func(scalar_t& val, int32_t& ind, const uint32_t delta) {
    scalar_t other_v = shfl_down_sync_func(val, delta);
    int32_t  other_i = shfl_down_sync_func(ind, delta);

    // On tie, prefer the smaller channel index (first-occurrence semantics).
    if (other_v > val || (other_v == val && other_i < ind)) {
        val = other_v;
        ind = other_i;
    }
}

template<typename scalar_t>
__forceinline__ __device__
void reduce_max(scalar_t& val, int32_t& ind, bool broadcast) {
    __syncthreads();
    max_pair_shfl_func(val, ind, 16);
    max_pair_shfl_func(val, ind, 8);
    max_pair_shfl_func(val, ind, 4);
    max_pair_shfl_func(val, ind, 2);
    max_pair_shfl_func(val, ind, 1);

    __shared__ scalar_t shm_v[32];
    __shared__ int32_t  shm_i[32];

    if (threadIdx.x % 32 == 0) {
        shm_v[threadIdx.x >> 5] = val;
        shm_i[threadIdx.x >> 5] = ind;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        val = shm_v[0];
        ind = shm_i[0];
        int32_t n_warps = (blockDim.x >> 5);
        if (threadIdx.x < n_warps) {
            val = shm_v[threadIdx.x];
            ind = shm_i[threadIdx.x];
        }
        max_pair_shfl_func(val, ind, 16);
        max_pair_shfl_func(val, ind, 8);
        max_pair_shfl_func(val, ind, 4);
        max_pair_shfl_func(val, ind, 2);
        max_pair_shfl_func(val, ind, 1);
    }

    if (broadcast) {
        broadcast_block_x(val, 0);
        broadcast_block_x(ind, 0);
    }
}



template<typename scalar_t>
__global__ void arg_max_depth(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int32_t *oten) {

    scalar_t max_val;
    int32_t max_ind;

    int samplesize = n_size * m_size;

    for (int i=blockIdx.x; i < samplesize; i += gridDim.x) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        /// NOTE: This is not reliable when dimsize < blockDim.x
        int idx = n_idx * dimsize * m_size + threadIdx.x * m_size + m_idx;
        int j = threadIdx.x + blockDim.x;
        max_val = __ldg(&inten[idx]);
        max_ind = threadIdx.x;
        for (; j < dimsize; j += blockDim.x) {
            idx += blockDim.x * m_size;
            scalar_t val = __ldg(&inten[idx]);
            if (val > max_val) {
                max_val = val;
                max_ind = j;
            }
        }
        reduce_max(max_val, max_ind, false);

        if (threadIdx.x == 0) {
            idx = n_idx * m_size + m_idx;
            oten[idx] = max_ind;
        }
    }
}


template<typename scalar_t>
__global__ void arg_max_spatial(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int32_t *oten) {

    int sample_offset = gridDim.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int samplesize = n_size * m_size;

    for (int i{tid}; i < samplesize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        // obtain max
        int idx = n_idx * dimsize * m_size + m_idx;
        scalar_t max_val = __ldg(&inten[idx]);
        int res = 0;
        for (int j{1}; j < dimsize; ++j) {
            idx += m_size;
            scalar_t val = __ldg(&inten[idx]);
            if (val > max_val) {
                max_val = val;
                res = j;
            }
        }
        idx = n_idx * m_size + m_idx;
        oten[idx] = res;
    }
}


// NHWC kernel: each thread handles one pixel, all C classes are contiguous in memory
template<typename scalar_t>
__global__ void arg_max_hwc(
        const int n_spatial, const int c_size,
        const scalar_t* inten, int32_t* oten) {

    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < n_spatial; i += stride) {
        const scalar_t* pixel = inten + i * c_size;
        scalar_t max_val = __ldg(pixel);
        int32_t max_idx = 0;
        for (int c = 1; c < c_size; c++) {
            scalar_t val = __ldg(pixel + c);
            if (val > max_val) { max_val = val; max_idx = c; }
        }
        oten[i] = max_idx;
    }
}

// INT8 HWC4 kernel: pixel stride = physical_c (multiple of 4, padded by TensorRT).
// Alignment guaranteed: inten is cudaMalloc-aligned and physical_c % 4 == 0,
// so pixel address = inten + i * physical_c is always 4-byte aligned.
__global__ void arg_max_hwc4_int8(
        const int n_spatial, const int c_size, const int physical_c,
        const int8_t* inten, int32_t* oten) {

    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int full_vecs = c_size / 4;

    for (int i = tid; i < n_spatial; i += stride) {
        const int8_t* pixel = inten + i * physical_c;  // padded stride
        int8_t max_val = -128;
        int32_t max_idx = 0;

        const char4* vec_ptr = reinterpret_cast<const char4*>(pixel);
        for (int v = 0; v < full_vecs; v++) {
            char4 packed = __ldg(vec_ptr + v);
            int base = v * 4;
            if (packed.x > max_val) { max_val = packed.x; max_idx = base;     }
            if (packed.y > max_val) { max_val = packed.y; max_idx = base + 1; }
            if (packed.z > max_val) { max_val = packed.z; max_idx = base + 2; }
            if (packed.w > max_val) { max_val = packed.w; max_idx = base + 3; }
        }
        // remaining 0-3 valid channels (padding bytes at physical_c are never read)
        for (int c = full_vecs * 4; c < c_size; c++) {
            int8_t val = __ldg(pixel + c);
            if (val > max_val) { max_val = val; max_idx = c; }
        }
        oten[i] = max_idx;
    }
}

// FP16 HWC8 kernel: pixel stride = physical_c (multiple of 8, padded by TensorRT).
// float4 load = 16 bytes = 8 fp16 values per load.
// Alignment: physical_c % 8 == 0 → pixel stride = physical_c * 2B (multiple of 16) → 16-byte aligned.
__global__ void arg_max_hwc8_fp16(
        const int n_spatial, const int c_size, const int physical_c,
        const __half* inten, int32_t* oten) {

    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int full_vecs = c_size / 8;

    for (int i = tid; i < n_spatial; i += stride) {
        const __half* pixel = inten + i * physical_c;
        __half max_val = __float2half(-65504.0f);
        int32_t max_idx = 0;

        const float4* vec_ptr = reinterpret_cast<const float4*>(pixel);
        for (int v = 0; v < full_vecs; v++) {
            float4 raw = __ldg(vec_ptr + v);
            const __half* h = reinterpret_cast<const __half*>(&raw);
            int base = v * 8;
            if (h[0] > max_val) { max_val = h[0]; max_idx = base;     }
            if (h[1] > max_val) { max_val = h[1]; max_idx = base + 1; }
            if (h[2] > max_val) { max_val = h[2]; max_idx = base + 2; }
            if (h[3] > max_val) { max_val = h[3]; max_idx = base + 3; }
            if (h[4] > max_val) { max_val = h[4]; max_idx = base + 4; }
            if (h[5] > max_val) { max_val = h[5]; max_idx = base + 5; }
            if (h[6] > max_val) { max_val = h[6]; max_idx = base + 6; }
            if (h[7] > max_val) { max_val = h[7]; max_idx = base + 7; }
        }
        // remaining 0-7 channels
        for (int c = full_vecs * 8; c < c_size; c++) {
            __half val = __ldg(pixel + c);
            if (val > max_val) { max_val = val; max_idx = c; }
        }
        oten[i] = max_idx;
    }
}

// Dispatch for FP16 HWC8 format.
// physical_c = ceil(c_size / 8) * 8, matching TensorRT kHWC8 layout.
inline void argMaxHWC8FP16Func(const __half* inten,
                                  int32_t* oten, const int n_spatial,
                                  const int c_size, cudaStream_t* stream) {
    if (inten == nullptr || oten == nullptr) std::terminate();
    const int physical_c = (c_size + 7) & ~7;
    int blockx, gridx;
    cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
            arg_max_hwc8_fp16, 0, n_spatial);
    gridx = std::min(4096, gridx << 2);
    if (stream == nullptr)
        arg_max_hwc8_fp16<<<gridx, blockx>>>(n_spatial, c_size, physical_c, inten, oten);
    else
        arg_max_hwc8_fp16<<<gridx, blockx, 0, *stream>>>(n_spatial, c_size, physical_c, inten, oten);
}

// FP32 HWC4 kernel: pixel stride = physical_c (multiple of 4, padded by TensorRT).
// float4 load = 16 bytes = 4 fp32 values per load.
// Alignment: physical_c % 4 == 0 → pixel stride = physical_c * 4B (multiple of 16) → 16-byte aligned.
__global__ void arg_max_hwc4_fp32(
        const int n_spatial, const int c_size, const int physical_c,
        const float* inten, int32_t* oten) {

    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int full_vecs = c_size / 4;

    for (int i = tid; i < n_spatial; i += stride) {
        const float* pixel = inten + i * physical_c;
        float max_val = -FLT_MAX;
        int32_t max_idx = 0;

        const float4* vec_ptr = reinterpret_cast<const float4*>(pixel);
        for (int v = 0; v < full_vecs; v++) {
            float4 p = __ldg(vec_ptr + v);
            int base = v * 4;
            if (p.x > max_val) { max_val = p.x; max_idx = base;     }
            if (p.y > max_val) { max_val = p.y; max_idx = base + 1; }
            if (p.z > max_val) { max_val = p.z; max_idx = base + 2; }
            if (p.w > max_val) { max_val = p.w; max_idx = base + 3; }
        }
        // remaining 0-3 channels
        for (int c = full_vecs * 4; c < c_size; c++) {
            float val = __ldg(pixel + c);
            if (val > max_val) { max_val = val; max_idx = c; }
        }
        oten[i] = max_idx;
    }
}

// Dispatch for FP32 HWC4 format.
// physical_c = ceil(c_size / 4) * 4, matching TensorRT kHWC4 layout.
inline void argMaxHWC4FP32Func(const float* inten,
                                  int32_t* oten, const int n_spatial,
                                  const int c_size, cudaStream_t* stream) {
    if (inten == nullptr || oten == nullptr) std::terminate();
    const int physical_c = (c_size + 3) & ~3;
    int blockx, gridx;
    cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
            arg_max_hwc4_fp32, 0, n_spatial);
    gridx = std::min(4096, gridx << 2);
    if (stream == nullptr)
        arg_max_hwc4_fp32<<<gridx, blockx>>>(n_spatial, c_size, physical_c, inten, oten);
    else
        arg_max_hwc4_fp32<<<gridx, blockx, 0, *stream>>>(n_spatial, c_size, physical_c, inten, oten);
}

// Dispatch for INT8 HWC4 format.
// physical_c = ceil(c_size / 4) * 4, matching TensorRT kHWC4 layout.
inline void argMaxHWC4Int8Func(const int8_t* inten,
                                 int32_t* oten, const int n_spatial,
                                 const int c_size, cudaStream_t* stream) {
    if (inten == nullptr || oten == nullptr) std::terminate();
    const int physical_c = (c_size + 3) & ~3;
    int blockx, gridx;
    cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
            arg_max_hwc4_int8, 0, n_spatial);
    gridx = std::min(4096, gridx << 2);
    if (stream == nullptr)
        arg_max_hwc4_int8<<<gridx, blockx>>>(n_spatial, c_size, physical_c, inten, oten);
    else
        arg_max_hwc4_int8<<<gridx, blockx, 0, *stream>>>(n_spatial, c_size, physical_c, inten, oten);
}

template<typename scalar_t>
void argMaxHWCFunc(const scalar_t* inten,
                    int32_t* oten, const int n_spatial,
                    const int c_size, cudaStream_t* stream) {

    if (inten == nullptr || oten == nullptr) std::terminate();

    int blockx, gridx;
    cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
            arg_max_hwc<scalar_t>, 0, n_spatial);
    gridx = std::min(4096, gridx << 2);

    if (stream == nullptr)
        arg_max_hwc<scalar_t><<<gridx, blockx>>>(n_spatial, c_size, inten, oten);
    else
        arg_max_hwc<scalar_t><<<gridx, blockx, 0, *stream>>>(n_spatial, c_size, inten, oten);
}

template<typename scalar_t>
void argMaxNCHWFunc(const scalar_t *inten,
                int32_t *oten, const int n_size,
                const int dimsize, const int m_size,
                cudaStream_t* stream) {

    if (inten == nullptr or oten == nullptr) std::terminate();

    int samplesize = n_size * m_size;
    dim3 grid, block;

    if (dimsize <= 1024) {
        int blockx, gridx;
        cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
                arg_max_spatial<scalar_t>, 0, samplesize);
        gridx = std::min(4096, gridx << 2);
        block.x = blockx; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_spatial<scalar_t><<<grid, block, 0>>>(
                    n_size, dimsize, m_size, inten, oten);
        } else {
            arg_max_spatial<scalar_t><<<grid, block, 0, *stream>>>(
                    n_size, dimsize, m_size, inten, oten);
        }

    } else {
        int blockx, gridx;
        int block_lmt = std::min(BLOCKSIZE, dimsize);
        blockx = 32;
        while (blockx <= block_lmt) blockx = (blockx << 1);
        blockx = (blockx >> 1); // must make sure dimsize > blockx
        gridx = std::min(16384, samplesize);
        block.x = blockx; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_depth<scalar_t><<<grid, block, 0>>>(
                    n_size, dimsize, m_size, inten, oten);
        } else {
            arg_max_depth<scalar_t><<<grid, block, 0, *stream>>>(
                    n_size, dimsize, m_size, inten, oten);
        }
    }
}

#endif
