//Joint Register and Shared Memory Tiling to speed up matrix multiplication 
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cstdio>
#include <cstdlib>

#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

#define U TILE_SZ_B
#define T TILE_SZ_A
#define S TILE_SZ_RATIO

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Implement the input matrix unrolling kernel.

    Function parameter definitions:
    X - input
    X_unroll - output
    B - batch_size (number of images in X)
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */
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
                    //float val = X[b * (C * H * W) + c * (H * W) + (h_out + p) * W + (w_out + q)];
                    float val=in_4d(b, c, h_out + p, w_out + q);
                    size_t indx = ((size_t)h_unroll) * (Batch * W_unroll) + b * W_unroll + w_unroll;
                    output[indx] = val;
                }
            }
        }
    }
    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns)
{
    /********************************************************************
     * C = A * B
     * A is (numARows x numAColumns), B is (numBRows x numBColumns)
     * C is (numCRows x numCColumns)
     *
     * We assume row-major layout for both A and B.
     * Thus:
     *   A(row,col) = A[row * numAColumns + col]
     *   B(row,col) = B[row * numBColumns + col]
     *   C(row,col) = C[row * numCColumns + col]
     ********************************************************************/

    // Thread indexing
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each block covers a tile of C of dimension (T x U)
    // Each thread computes one element in that (T x U) tile.
    // Here, blockDim.y = T and blockDim.x = 1 (from our launch configuration).
    // The output C tile top-left corner in the output matrix:
    int row = by * T + ty;
    // The column is computed as bx*U + (0 to U-1) inside the loop when we accumulate results.

    // Shared memory for a tile of B
    __shared__ float B_shared[S][U];

    // Register accumulator for C
    float C_reg[U];
    for (int i = 0; i < U; i++) {
        C_reg[i] = 0.0f;
    }

    // We will iterate over k dimension in tiles of size S
    // k dimension = numAColumns = numBRows
    int tiles = (numAColumns + S - 1) / S;

    for (int t = 0; t < tiles; t++) {
        // Load B tile into shared memory.
        // Each thread loads one element of B if in range.
        // tile_row = ty / U and tile_column = ty % U doesn't apply here directly since we have T threads in y.
        // Instead, we can use ty for indexing B's tile:
        int b_row = t * S + (ty / U);    // Each group of U threads in y loads a row of B’s tile
        int b_col = bx * U + (ty % U);
        if (b_row < numBRows && b_col < numBColumns) {
            B_shared[(ty / U)][(ty % U)] = B[b_row * (size_t)numBColumns + b_col];
        } else {
            B_shared[(ty / U)][(ty % U)] = 0.0f;
        }
        __syncthreads();

        // Now compute partial products
        // For each column of B tile in shared mem, we load an element from A
        for (int i = 0; i < S; i++) {
            float A_reg = 0.0f;
            int a_col = t * S + i; // moving along k dimension
            if (row < numARows && a_col < numAColumns) {
                A_reg = A[row * (size_t)numAColumns + a_col];
            }

            for (int j = 0; j < U; j++) {
                C_reg[j] += A_reg * B_shared[i][j];
            }
        }
        __syncthreads();
    }

    // Write the results back to C
    // Each thread writes out a single row element in the tile, for U columns.
    for (int i = 0; i < U; i++) {
        int col = bx * U + i;
        if (row < numCRows && col < numCColumns) {
            C[row * numCColumns + col] = C_reg[i];
        }
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            size_t o_idx=b * Map_out * image_size + m * image_size + x;
            size_t i_idx=m * Batch * image_size + b * image_size + x;
            output[o_idx] =
                    input[i_idx];
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

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask, const int Batch,
                                             const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Height_out * Width_out;

    float *unrolled_matrix;
    float *matmul_output;

    size_t unrolled_size = (size_t)H_unroll * Batch * W_unroll * sizeof(float);
    size_t matmul_output_size = (size_t)Map_out * Batch * W_unroll * sizeof(float);

    cudaMalloc((void **) &unrolled_matrix, unrolled_size);
    cudaMalloc((void **) &matmul_output, matmul_output_size);

    // Unrolling kernel launch configuration
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

    // New block and grid dims for joint tiling
    dim3 blockDimMatMul(1, T, 1);
    dim3 gridDimMatMul((numCColumns + U - 1) / U,
                       (numCRows + T - 1) / T);

    matrixMultiplyTiled<<<gridDimMatMul, blockDimMatMul>>>(device_mask, unrolled_matrix,
                                                           matmul_output,
                                                           numARows, numAColumns,
                                                           numBRows, numBColumns,
                                                           numCRows, numCColumns);

    // Permute the result
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

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
