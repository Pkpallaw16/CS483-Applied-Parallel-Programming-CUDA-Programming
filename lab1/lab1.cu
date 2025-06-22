// LAB 1
#include <wb.h>
#define BLOCK_SIZE 32
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
	int i =blockIdx.x * blockDim.x + threadIdx.x;
	if (i<len)
       	{
	  out[i]=in1[i] + in2[i];
	}


}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *dev_Input1;
  float *dev_Input2;
  float *dev_Output;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  cudaMalloc((void **)&dev_Input1, inputLength * sizeof(float));
  cudaMalloc((void **)&dev_Input2, inputLength * sizeof(float));
  cudaMalloc((void **)&dev_Output, inputLength * sizeof(float));
  //@@ Copy memory to the GPU here
  cudaMemcpy(dev_Input1,hostInput1,inputLength * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Input2,hostInput2,inputLength * sizeof(float),cudaMemcpyHostToDevice);

	  

  //@@ Initialize the grid and block dimensions here
 // dim3 DimGrid{(inputLength+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1};
  //dim3 DimGrid((inputLength-1)/32 + 1);

// dim3 DimGrid(ceil(float(inputLength)/float(32)));

 dim3 DimGrid((inputLength+31)/32);



  //dim3 DimGrid{ceil((float)inputLength/BLOCK_SIZE), 1, 1};
  dim3 DimBlock{BLOCK_SIZE,1,1};

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(dev_Input1,dev_Input2,dev_Output,inputLength);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput,dev_Output,inputLength * sizeof(float),cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(dev_Input1);
  cudaFree(dev_Input2);
  cudaFree(dev_Output);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
