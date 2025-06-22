// Histogram Equalization

#include <wb.h>

#define HIS_LENGTH 256
#define BLOCK_SIZE   256

__global__ void convertFloatToUChar(float *input, unsigned char *output, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < length) {
        output[index] = (unsigned char)(255 * input[index]);
    }
}

__global__ void convertRGBtoGray(unsigned char *rgbImage, unsigned char *grayImage, int pixelCount) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < pixelCount) {
        unsigned char red = rgbImage[3 * index];
        unsigned char green = rgbImage[3 * index + 1];
        unsigned char blue = rgbImage[3 * index + 2];
        grayImage[index] = (unsigned char)(0.21f * red + 0.71f * green + 0.07f * blue);
    }
}

__global__ void computeHistogram(unsigned char *input, unsigned int *globalHist, int length) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned int localHist[HIS_LENGTH];

    // Initialize shared memory histogram
    if (threadIdx.x < HIS_LENGTH) {
        localHist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Accumulate histogram in shared memory
    if (index < length) {
        atomicAdd(&(localHist[input[index]]), 1);
    }
    __syncthreads();

    // Accumulate local histogram into global histogram
    if (threadIdx.x < HIS_LENGTH) {
        atomicAdd(&(globalHist[threadIdx.x]), localHist[threadIdx.x]);
    }
}

__global__ void calculateCDF(unsigned int *input, float *output, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sharedCDF[HIS_LENGTH];

    // Load input into shared memory
    if (i < HIS_LENGTH) {
        sharedCDF[i] = input[i];
    }
    __syncthreads();

    // Perform up-sweep (reduce) step
    for (int stride = 1; stride < HIS_LENGTH; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < HIS_LENGTH && index - stride >= 0) {
            sharedCDF[index] += sharedCDF[index - stride];
        }
        __syncthreads();
    }

    // Perform down-sweep step
    for (int stride = HIS_LENGTH / 4; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < HIS_LENGTH && index >= 0) {
            sharedCDF[index + stride] += sharedCDF[index];
        }
        __syncthreads();
    }

    // Write the results to the output array
    if (i < HIS_LENGTH) {
        output[i] = sharedCDF[i] / (float)len;
    }
}

__global__ void histogramEqualization(unsigned char *image, float *cdf, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length) {
        float pixelValue = 255.0 * (cdf[image[idx]] - cdf[0]) / (1.0 - cdf[0]);
        float clampedValue = fminf(fmaxf(pixelValue, 0.0f), 255.0f);
        image[idx] = (unsigned char)clampedValue;
    }
}

__global__ void convertUnsignedCharToFloat(unsigned char *input, float *output, int length) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < length) {
        output[index] = (float)input[index] / 255.0f;
    }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceIn;
  unsigned char *deviceUChar;
  unsigned char *deviceGray;
  unsigned int *deviceHist;
  float *deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceIn, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceUChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGray, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHist, HIS_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HIS_LENGTH * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");


  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceIn, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch the kernel
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid2((imageWidth * imageHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  dim3 dimGrid((imageWidth * imageHeight * imageChannels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  convertFloatToUChar<<<dimGrid, dimBlock>>>(deviceIn, deviceUChar, imageWidth * imageHeight * imageChannels);
  convertRGBtoGray<<<dimGrid2, dimBlock>>>(deviceUChar, deviceGray, imageWidth * imageHeight);
  computeHistogram<<<dimGrid2, dimBlock>>>(deviceGray, deviceHist, imageWidth * imageHeight);
  calculateCDF<<<1, HIS_LENGTH>>>(deviceHist, deviceCDF, imageWidth * imageHeight);
  histogramEqualization<<<dimGrid, dimBlock>>>(deviceUChar, deviceCDF, imageWidth * imageHeight * imageChannels);
  convertUnsignedCharToFloat<<<dimGrid, dimBlock>>>(deviceUChar, deviceIn, imageWidth * imageHeight * imageChannels);

  wbTime_start(Copy, "Copying output memory to the CPU.");
  cudaMemcpy(hostOutputImageData, deviceIn, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU.");

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceIn);
  cudaFree(deviceUChar);
  cudaFree(deviceGray);
  cudaFree(deviceHist);
  cudaFree(deviceCDF);

  return 0;
}
