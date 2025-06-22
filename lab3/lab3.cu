#include <wb.h>
#define BLOCK_SIZE 32
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subTileB[BLOCK_SIZE][BLOCK_SIZE];
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  float Pvalue=0;

  for (int m = 0; m < (numAColumns-1)/BLOCK_SIZE+1; m++) {
    if (BLOCK_SIZE*m + tx < numAColumns) {
      subTileA[ty][tx] = A[numAColumns*row + BLOCK_SIZE*m + tx];
    }
    else {
      subTileA[ty][tx] = 0.0f;
    }

    if (BLOCK_SIZE*m + ty < numBRows) {
      subTileB[ty][tx] = B[(BLOCK_SIZE*m+ty)*numBColumns + col];
    }
    else {
      subTileB[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++)
      if (row < numARows && col < numBColumns)
        Pvalue += subTileA[ty][k] * subTileB[k][tx];

    __syncthreads();

  }

  if (row < numARows && col < numBColumns)
    C[row*numBColumns+col] = Pvalue;

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  float *devA;
  float *devB;
  float *devC;

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  //@@ Allocate GPU memory here
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&devA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&devB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&devC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");


  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input to the GPU.");
  cudaMemcpy(devA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_start(Compute, "Performing CUDA computation");


  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((float)numCColumns/BLOCK_SIZE), ceil((float)numCRows/BLOCK_SIZE),1);
  dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  

  //@@ Launch the GPU Kernel here
  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiplyShared<<<DimGrid,DimBlock>>>(devA, devB, devC, numARows, numAColumns, numBRows,
      numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  //@@ Copy the GPU memory back to the CPU here
  wbTime_stop(Copy, "Copying output to the CPU");
  cudaMemcpy(hostC, devC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output to the CPU");

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Free GPU Memory");
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  wbTime_start(GPU, "Free GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);
  //@@ Free the hostC matrix

  return 0;
}
