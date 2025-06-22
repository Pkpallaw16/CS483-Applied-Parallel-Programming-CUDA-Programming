#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

// Global memory version
__global__ void conv3d(float *input, float *output, const int z_size,
                             const int y_size, const int x_size) {
  // Compute the global 3D index of the current thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  // Define radius for the 3x3x3 kernel
  const int radius = MASK_WIDTH / 2;
  
  // Initialize the result to accumulate convolution values
  float result = 0.0f;

  // Iterate over the 3x3x3 kernel region centered around (x, y, z)
  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      for (int k = -radius; k <= radius; k++) {
        // Calculate the corresponding coordinates in the input
        int xx = x + k;
        int yy = y + j;
        int zz = z + i;

        // Check if the coordinates are within the bounds of the input array
        if (xx >= 0 && xx < x_size &&
            yy >= 0 && yy < y_size &&
            zz >= 0 && zz < z_size) {
          // Perform the convolution using the kernel and the input
          result += input[xx + yy * x_size + zz * x_size * y_size] * 
                    deviceKernel[i + radius][j + radius][k + radius];
        }
      }
    }
  }

  // Store the result in the output array, if the current thread coordinates are within bounds
  if (x < x_size && y < y_size && z < z_size) {
    output[x + y * x_size + z * x_size * y_size] = result;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3)*sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3)*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float));
  cudaDeviceSynchronize();
  wbTime_stop(Copy, "Copying data to the GPU");

  //@@ Initialize grid and block dimensions here
  dim3 gridDim((x_size + TILE_WIDTH - 1) / TILE_WIDTH,
               (y_size + TILE_WIDTH - 1) / TILE_WIDTH,
               (z_size + TILE_WIDTH - 1) / TILE_WIDTH);
  dim3 blockDim{TILE_WIDTH, TILE_WIDTH, TILE_WIDTH};
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3)*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

