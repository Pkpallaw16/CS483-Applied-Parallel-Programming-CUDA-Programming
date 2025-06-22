// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void add(float *input, float *output, int len) {
    unsigned int thread_id = threadIdx.x;
    unsigned int block_start_idx = 2 * blockIdx.x * BLOCK_SIZE;

    // Skip first block, as it does not require modification
    if (blockIdx.x == 0) return;

    // Index of the previous block in the input array
    float increment_value = input[blockIdx.x - 1];

    // Compute indices for the two output segments this block handles
    unsigned int first_idx = block_start_idx + thread_id;
    unsigned int second_idx = block_start_idx + BLOCK_SIZE + thread_id;

    // Add the increment value to each segment, if within bounds
    if (first_idx < len) {
        output[first_idx] += increment_value;
    }
    if (second_idx < len) {
        output[second_idx] += increment_value;
    }
}

__global__ void scan(float *input, float *output, int len, float *block_sum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float T[2*BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start_idx = 2 * blockIdx.x * BLOCK_SIZE;

  T[t] = (start_idx + t < len) ? input[start_idx + t] : 0;
  T[BLOCK_SIZE + t] = (start_idx + BLOCK_SIZE + t < len) ? input[start_idx + BLOCK_SIZE + t] : 0;

  // reduction step 
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int idx = (t + 1) * stride * 2 - 1;
    if (idx < 2 * BLOCK_SIZE && (idx - stride) >= 0)
      T[idx] += T[idx-stride];
    stride *= 2;
  }

  // post scan step
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int idx = (t + 1) * stride * 2 - 1;
    if ((idx + stride) < 2 * BLOCK_SIZE)
      T[idx + stride] += T[idx];
    stride /= 2;
  }

  __syncthreads();
  if (start_idx + t < len)
    output[start_idx + t] = T[t];
  if (start_idx + BLOCK_SIZE + t < len)
    output[start_idx + BLOCK_SIZE + t] = T[BLOCK_SIZE + t];
 
  __syncthreads();
  if (t == BLOCK_SIZE - 1){
    if (block_sum != NULL){
      block_sum[blockIdx.x] = T[2 * BLOCK_SIZE - 1];
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *block_sum;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  unsigned int grid_size = (numElements-1) / (2 * BLOCK_SIZE) + 1;
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&block_sum, grid_size * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 gridDim(grid_size, 1, 1);
  dim3 DimGridAdd(1, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements, block_sum);
  scan<<<DimGridAdd, blockDim>>>(block_sum, block_sum, grid_size, NULL);
  add<<<gridDim, blockDim>>>(block_sum, deviceOutput, numElements);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
