#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
//m3_op2.out
// Apply __restrict__ to the pointer parameters
__global__ void matrix_unrolling_kernel(const float *__restrict__ input, float *__restrict__ output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    int b = blockIdx.z;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    const int W_unroll = H_out * W_out;
    const int H_unroll = Channel * K * K;

    if (h_out < H_out && w_out < W_out && b < Batch) {
        int w_unroll = h_out * W_out + w_out;

        for (int c = 0; c < Channel; c++) {
            int w_base = c * K * K;
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    int h_unroll = w_base + p * K + q;
                    float val = in_4d(b, c, h_out + p, w_out + q);
                    size_t indx = ((size_t)h_unroll) * (Batch * W_unroll) + b * W_unroll + w_unroll;
                    output[indx] = val;
                }
            }
        }
    }
    #undef in_4d
}

__global__ void matrixMultiplyShared(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t)row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t)tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

__global__ void matrix_permute_kernel(const float *__restrict__ input, float *__restrict__ output,
                                      int Map_out, int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask, float **device_output_ptr,
                                                    float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **) device_input_ptr, input_size);
    cudaMalloc((void **) device_output_ptr, output_size);
    cudaMalloc((void **) device_mask_ptr, mask_size);

    // Copy data from host to device
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float * __restrict__ device_output, const float * __restrict__ device_input,
                                             const float * __restrict__ device_mask, const int Batch,
                                             const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Height_out * Width_out;

    float *__restrict__ unrolled_matrix;
    float *__restrict__ matmul_output;

    size_t unrolled_size = (size_t)H_unroll * Batch * W_unroll * sizeof(float);
    size_t matmul_output_size = (size_t)Map_out * Batch * W_unroll * sizeof(float);

    cudaMalloc((void **) &unrolled_matrix, unrolled_size);
    cudaMalloc((void **) &matmul_output, matmul_output_size);

    // Kernel dimensions for unrolling
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((Width_out + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Height_out + TILE_WIDTH - 1) / TILE_WIDTH,
                 Batch);

    matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_input, unrolled_matrix,
                                                   Batch, Channel, Height, Width, K);

    // Matrix multiplication dimensions
    int numARows = Map_out;
    int numAColumns = H_unroll;
    int numBRows = H_unroll;
    int numBColumns = Batch * W_unroll;
    int numCRows = Map_out;
    int numCColumns = Batch * W_unroll;

    dim3 gridDimMatMul((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
                       (numCRows + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyShared<<<gridDimMatMul, blockDim>>>(device_mask, unrolled_matrix,
                                                      matmul_output, numARows, numAColumns,
                                                      numBRows, numBColumns,
                                                      numCRows, numCColumns);

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);

    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(matmul_output,
                                                                   device_output,
                                                                   Map_out, Batch,
                                                                   out_image_size);

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}

