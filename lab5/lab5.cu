// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];
#include <wb.h>

#define BLOCK_SIZE 512  //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the correct index
  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockDim.x * blockIdx.x;
  if (start + t < len)
  {
    partialSum[t] = input[start + t];
  }
  else
  {
    partialSum[t] = 0;
  }
  if (start + blockDim.x + t < len)
  {
    partialSum[blockDim.x + t] = input[start + blockDim.x + t];
  }
  else
  {
    partialSum[blockDim.x + t] = 0;
  }
  for (unsigned int stride = blockDim.x; stride >= 1; stride >>= 1) {
    __syncthreads();
    if (t < stride)
    {
      partialSum[t] += partialSum[t + stride];
    }
  }
  if (t == 0)
  {
    output[blockIdx.x] = partialSum[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers
  float *dev_Input;
  float *dev_Output;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&dev_Input, numInputElements * sizeof(float));
  cudaMalloc((void **)&dev_Output, numOutputElements * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(dev_Input, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numInputElements - 1) / BLOCK_SIZE + 1, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<DimGrid, DimBlock>>>(dev_Input, dev_Output, numInputElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, dev_Output, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /***********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input.
   * For simplicity, we do not require that for this lab!
   ***********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
 // printf("%f\n", hostOutput[0]);

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(dev_Input);
  cudaFree(dev_Output);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
