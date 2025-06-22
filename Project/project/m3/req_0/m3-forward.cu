#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Implement the input matrix unrolling kernel.

    Function parameter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in input)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
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
                    float val = in_4d(b, c, h_out + p, w_out + q);
                    size_t indx = ((size_t)h_unroll) * (Batch * W_unroll) + b * W_unroll + w_unroll;
                    output[indx] = val;
                }
            }
        }
    }
    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
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
    // Allocate device memory for mask and copy host mask to device
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void **) device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Allocate pinned host memory for input and output
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    float *pinned_host_input;
    float *pinned_host_output;
    cudaMallocHost((void**)&pinned_host_input, input_size);
    cudaMallocHost((void**)&pinned_host_output, output_size);

    // Copy host input to pinned memory
    memcpy(pinned_host_input, host_input, input_size);

    // Create CUDA streams
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Process data in chunks
    int chunk_size = (Batch + num_streams - 1) / num_streams; // Distribute batches evenly
    for (int chunk = 0; chunk < num_streams; chunk++) {
        int batch_start = chunk * chunk_size;
        int current_chunk_size = min(chunk_size, Batch - batch_start);

        if (current_chunk_size <= 0)
            break;

        // Allocate device memory for input and output for this chunk
        size_t chunk_input_size = current_chunk_size * Channel * Height * Width * sizeof(float);
        size_t chunk_output_size = current_chunk_size * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

        float *device_input;
        float *device_output;
        cudaMalloc((void **) &device_input, chunk_input_size);
        cudaMalloc((void **) &device_output, chunk_output_size);

        // Asynchronously copy input data to device
        cudaMemcpyAsync(device_input,
                        pinned_host_input + batch_start * Channel * Height * Width,
                        chunk_input_size, cudaMemcpyHostToDevice,
                        streams[chunk]);

        // Launch kernels in the stream
        // Compute grid and block dimensions
        int Height_out = Height - K + 1;
        int Width_out = Width - K + 1;
        int H_unroll = Channel * K * K;
        int W_unroll = current_chunk_size * Height_out * Width_out;

        // Allocate device memory for unrolled matrix and matmul output
        float *unrolled_matrix;
        float *matmul_output;
        size_t unrolled_size = H_unroll * W_unroll * sizeof(float);
        size_t matmul_output_size = Map_out * W_unroll * sizeof(float);

        cudaMalloc((void **)&unrolled_matrix, unrolled_size);
        cudaMalloc((void **)&matmul_output, matmul_output_size);

        // Launch unrolling kernel
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((Width_out + TILE_WIDTH - 1) / TILE_WIDTH,
                     (Height_out + TILE_WIDTH - 1) / TILE_WIDTH,
                     current_chunk_size);

        matrix_unrolling_kernel<<<gridDim, blockDim, 0, streams[chunk]>>>(device_input, unrolled_matrix,
                                                                           current_chunk_size, Channel, Height, Width, K);

        // Launch matrix multiplication kernel
        int numARows = Map_out;
        int numAColumns = H_unroll;
        int numBRows = H_unroll;
        int numBColumns = W_unroll;
        int numCRows = Map_out;
        int numCColumns = W_unroll;

        dim3 gridDimMatMul((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
                           (numCRows + TILE_WIDTH - 1) / TILE_WIDTH);

        matrixMultiplyShared<<<gridDimMatMul, blockDim, 0, streams[chunk]>>>(*device_mask_ptr, unrolled_matrix,
                                                                             matmul_output, numARows, numAColumns,
                                                                             numBRows, numBColumns,
                                                                             numCRows, numCColumns);

        // Launch permutation kernel
        int out_image_size = Height_out * Width_out;
        dim3 permute_grid_dim((out_image_size + BLOCK_SIZE - 1) / BLOCK_SIZE, current_chunk_size);
        matrix_permute_kernel<<<permute_grid_dim, BLOCK_SIZE, 0, streams[chunk]>>>(matmul_output,
                                                                                   device_output,
                                                                                   Map_out, current_chunk_size,
                                                                                   out_image_size);
        // Synchronize the stream to ensure kernel completion
        cudaStreamSynchronize(streams[chunk]);                                                                           

        // Asynchronously copy output data back to host
        cudaMemcpyAsync(pinned_host_output + batch_start * Map_out * Height_out * Width_out,
                        device_output, chunk_output_size, cudaMemcpyDeviceToHost, streams[chunk]);

        // Free device memory for input, output, unrolled matrix, matmul output
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(unrolled_matrix);
        cudaFree(matmul_output);
    }

    // Synchronize streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Copy output data from pinned memory to host_output
    memcpy((void*)host_output, pinned_host_output, output_size);

    // Free pinned host memory
    cudaFreeHost(pinned_host_input);
    cudaFreeHost(pinned_host_output);

    // Free device memory for mask
    cudaFree(*device_mask_ptr);

    // Destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask, const int Batch,
                                             const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K) {
    // This function is intentionally left empty as all the work is done in conv_forward_gpu_prolog
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    // This function is intentionally left empty as all the work is done in conv_forward_gpu_prolog
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

