#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

// Kernel to transpose a matrix from row-major to column-major
__global__ void transpose_to_colmajor(const float* __restrict__ in, float* __restrict__ out,
                                      int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        // (r, c) in row-major -> (c, r) in column-major
        // For column-major indexing out, element (c, r) means out[c * rows + r]
        out[c * rows + r] = in[r * cols + c];
    }
}

// Kernel to transpose a matrix from column-major to row-major
__global__ void transpose_to_rowmajor(const float* __restrict__ in, float* __restrict__ out,
                                      int rows, int cols) {
    // Now 'in' is in column-major form with dimension rows x cols
    // We want out in row-major form.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        // in is column-major: element at (r,c) is in[r + c*rows]
        // out is row-major: element at (r,c) is out[r*cols + c]
        out[r * cols + c] = in[r + c * rows];
    }
}

__global__ void matrix_unrolling_kernel(const float *input, float *output,
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

// We no longer need this custom matrixMultiplyShared kernel for the main multiplication.
// We keep it here for reference. We will use cuBLAS instead.
// __global__ void matrixMultiplyShared(...){...}

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

    // Kernel dimensions for unrolling
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((Width_out + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Height_out + TILE_WIDTH - 1) / TILE_WIDTH,
                 Batch);

    matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_input, unrolled_matrix,
                                                   Batch, Channel, Height, Width, K);

    // Now we have:
    // A = device_mask, dimension: Map_out x (Channel*K*K) in row-major
    // B = unrolled_matrix, dimension: (Channel*K*K) x (Batch*W_unroll) in row-major
    // C = matmul_output, dimension: Map_out x (Batch*W_unroll) in row-major

    // cuBLAS expects column-major. We'll transpose A and B into column-major arrays.
    float *A_col, *B_col, *C_col;
    size_t A_col_size=Map_out * H_unroll * sizeof(float);
    size_t B_col_size=H_unroll * Batch * W_unroll * sizeof(float);
    size_t C_col_size=Map_out * Batch * W_unroll * sizeof(float);
    cudaMalloc((void**)&A_col, A_col_size);
    cudaMalloc((void**)&B_col, B_col_size);
    cudaMalloc((void**)&C_col, C_col_size);

    // Transpose A (Map_out x H_unroll) from row-major to col-major
    {
        size_t total = Map_out * H_unroll;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        transpose_to_colmajor<<<blocks, threads>>>(device_mask, A_col, Map_out, H_unroll);
    }

    // Transpose B (H_unroll x Batch*W_unroll) from row-major to col-major
    {
        size_t total = H_unroll * Batch * W_unroll;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        transpose_to_colmajor<<<blocks, threads>>>(unrolled_matrix, B_col, H_unroll, Batch * (size_t)W_unroll);
    }

    // Use cuBLAS for matrix multiplication: C_col = A_col * B_col
    // Here, A_col is (Map_out x H_unroll) col-major, B_col is (H_unroll x (Batch*W_unroll)) col-major
    // C_col will be (Map_out x (Batch*W_unroll)) col-major
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    // Dimensions for cublasSgemm:
    // C = A * B
    // A: MxK = Map_out x H_unroll
    // B: KxN = H_unroll x (Batch*W_unroll)
    // C: MxN = Map_out x (Batch*W_unroll)
    int M = Map_out;
    int K_ = H_unroll;
    int N = Batch * W_unroll;
    // Leading dimensions (since col-major):
    // A_col: lda = M
    // B_col: ldb = K_
    // C_col: ldc = M
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K_,
                &alpha,
                A_col, M,
                B_col, K_,
                &beta,
                C_col, M);

    cublasDestroy(handle);

    // Now C_col is in column-major. We need it in row-major to apply the permutation kernel.
    // C_col is Map_out x (Batch*W_unroll) column-major.
    // We'll transpose it back to row-major into matmul_output.
    {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        transpose_to_rowmajor<<<blocks, threads>>>(C_col, matmul_output, M, N);
    }

    // Permute the result of matrix multiplication
    size_t out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);

    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(matmul_output,
                                                                   device_output,
                                                                   Map_out, Batch,
                                                                   out_image_size);

    // Free temporary arrays
    cudaFree(C_col);
    cudaFree(B_col);
    cudaFree(A_col);

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t out_size= Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, out_size, cudaMemcpyDeviceToHost);

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

