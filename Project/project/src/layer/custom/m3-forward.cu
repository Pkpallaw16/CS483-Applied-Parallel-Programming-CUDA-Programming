#include <cmath>
#include <iostream>
#include <cuda.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void fused_conv_kernel(const float * __restrict__ device_input,
                                  const float * __restrict__ device_mask,
                                  float * __restrict__ device_output,
                                  const int Batch, const int Map_out, const int Channel,
                                  const int Height, const int Width, const int K) {

    // Compute output dimensions
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    // Dimensions for the matrix multiplication
    // A: (Map_out x H_unroll), where H_unroll = C*K*K
    // B: (H_unroll x (Batch * H_out * W_out))
    // C: (Map_out x (Batch * H_out * W_out))
    const int H_unroll = Channel * K * K;
    const int W_unroll = H_out * W_out;

    // Shared memory tiles for A (filters) and B (unrolled input)
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Calculate global thread row and column for C = A*B
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;  // indexing along Map_out dimension (A rows, C rows)
    int col = bx * TILE_WIDTH + tx;  // indexing along (Batch*H_out*W_out) dimension (B columns, C columns)

    float val = 0.0f;

    // Number of tiles needed
    int numTiles = (H_unroll + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Load tileA from device_mask (A matrix)
        // A has dimensions: Map_out x (C*K*K)
        // row indexes the output feature map (m), column indexes "k" dimension in A
        int A_col = t * TILE_WIDTH + tx; // indexing into H_unroll dimension
        if (row < Map_out && A_col < H_unroll) {
            tileA[ty][tx] = device_mask[row * H_unroll + A_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tileB on-the-fly from device_input:
        // B = unrolled input, dimensions: (C*K*K) x (Batch*H_out*W_out)
        // B indexing:
        //   Row in B: corresponds to a particular (c, p, q) triplet.
        //     c = row_in_B / (K*K)
        //     temp = row_in_B % (K*K)
        //     p = temp / K
        //     q = temp % K
        //
        //   Col in B: corresponds to (b, h_out, w_out):
        //     b = col_in_B / W_unroll
        //     temp2 = col_in_B % W_unroll
        //     h_out = temp2 / W_out
        //     w_out = temp2 % W_out
        //
        // Then: input index = (b, c, h_out+p, w_out+q)
        int B_row = t * TILE_WIDTH + ty; // indexing into C*K*K (H_unroll)
        int B_col = col;                 // indexing into Batch*H_out*W_out

        float B_val = 0.0f;
        if (B_row < H_unroll && B_col < (Batch * W_unroll)) {
            // Decompose B_row into c, p, q
            int c = B_row / (K*K);
            int rem = B_row % (K*K);
            int p = rem / K;
            int q = rem % K;

            // Decompose B_col into b, h_out, w_out
            int b = B_col / W_unroll;
            int tmp = B_col % W_unroll;
            int h_out_idx = tmp / W_out;
            int w_out_idx = tmp % W_out;

            // Compute the corresponding input index
            // Make sure indices are in range, they should be as per definition.
            int in_h = h_out_idx + p;
            int in_w = w_out_idx + q;
            // device_input index:
            // input layout: (B, C, H, W)
            // index = b * (C*H*W) + c*(H*W) + in_h*W + in_w
            B_val = device_input[b * (Channel * Height * Width) +
                                  c * (Height * Width) +
                                  in_h * Width + in_w];
        }

        tileB[ty][tx] = B_val;

        __syncthreads();

        // Compute partial dot product for C
        if (row < Map_out && col < (Batch * W_unroll)) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }

        __syncthreads();
    }

    // Now val contains the result for C[row, col] = sum over k of A[row,k]*B[k,col].
    // We need to write this result directly into the permuted output:
    // Original desired layout: Output: (B, M, H_out, W_out)
    // We have row = m (Map_out index)
    // For col, decompose again:
    if (row < Map_out && col < (Batch * W_unroll)) {
        int b = col / W_unroll;
        int tmp = col % W_unroll;
        int h_out_idx = tmp / W_out;
        int w_out_idx = tmp % W_out;

        // device_output index:
        // output layout: (B, M, H_out, W_out)
        // index = b*(M*H_out*W_out) + row*(H_out*W_out) + h_out_idx*(W_out) + w_out_idx
        device_output[b * (Map_out * H_out * W_out) +
                      row * (H_out * W_out) +
                      h_out_idx * W_out + w_out_idx] = val;
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask, float **device_output_ptr,
                                                    float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    size_t input_size = (size_t) Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = (size_t) Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = (size_t) Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void **) device_input_ptr, input_size);
    cudaMalloc((void **) device_output_ptr, output_size);
    cudaMalloc((void **) device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask, const int Batch,
                                             const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K) {
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    // Dimensions for fused kernel (like matmul):
    // C: Map_out x (Batch * H_out * W_out)
    // We have M = Map_out
    // N = Batch * H_out * W_out
    int M = Map_out;
    int N = Batch * H_out * W_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    fused_conv_kernel<<<gridDim, blockDim>>>(device_input, device_mask, device_output,
                                             Batch, Map_out, Channel,
                                             Height, Width, K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    cudaMemcpy(host_output, device_output,
               Batch * Map_out * H_out * W_out * sizeof(float),
               cudaMemcpyDeviceToHost);

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

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: "
                  << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, "
                  << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: "
                  << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}

