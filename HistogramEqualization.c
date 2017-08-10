// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define SECTION_SIZE 512
#define HISTO_SIZE 16

//@@ insert code here
/**
First add kernel that will (1) convert image pixel value from float to unsigned char and
(2) will convert image pixel to greyscale representation
**/
__global__ void pixelConversion(float * input, unsigned char * uchar, unsigned char * greyImg, int width, int height)
{
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  int idx = row*width + col;
  if(row < height && col < width){
    uchar[idx*3] = (unsigned char)(255*input[idx*3]);
    uchar[3*idx+1] = (unsigned char)(255*input[idx*3+1]);
    uchar[3*idx+2] = (unsigned char)(255*input[idx*3+2]);

    greyImg[idx] = (unsigned char)(0.21*uchar[idx*3] + 0.71*uchar[idx*3+1] + 0.07*uchar[idx*3+2]);
  }
}

__global__ void Histogram(unsigned char * greyImg, int * histogram, int size){
  __shared__ int histogram_private[HISTOGRAM_LENGTH];
  int bx = blockIdx.x; int tx = threadIdx.x;
  int idx = tx + bx*blockDim.x;
  int stride = blockDim.x * gridDim.x;

  if(tx < HISTOGRAM_LENGTH) histogram_private[tx] = 0;
  __syncthreads();

  while(idx < size){
    atomicAdd(&(histogram_private[greyImg[idx]]), 1);
    idx += stride;
  }
  __syncthreads();

  if(tx < HISTOGRAM_LENGTH) atomicAdd(&(histogram[tx]), histogram_private[tx]);
}

__global__ void histoScan(int * histogram, float * cdf, int size){
  __shared__ float P[SECTION_SIZE];
  int tx = threadIdx.x;

  if(tx >= HISTOGRAM_LENGTH){
    P[tx] = 0; P[tx+(SECTION_SIZE/2)] = 0;
  }
  else P[tx] = float(float(histogram[tx])/size);
  __syncthreads();

  for(int stride = 1; stride <= SECTION_SIZE/2; stride *= 2){
    int idx = (tx + 1) * 2 * stride - 1;
    if(idx < SECTION_SIZE){
      P[idx] += P[idx - stride];
    }
    __syncthreads();
  }

  for(int stride = SECTION_SIZE/4; stride > 0; stride /= 2){
    __syncthreads();
    int idx = (tx + 1) * 2 * stride - 1;
    if(idx + stride < SECTION_SIZE){
      P[idx + stride] += P[idx];
    }
  }

  __syncthreads();
  if(tx < HISTOGRAM_LENGTH) cdf[tx] = P[tx];
}

__global__ void Equalization(unsigned char * uchar, float * cdf, float * output, int width, int height){
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  int idx = row*width + col;
  float minVal = cdf[0];
  if(row < height && col < width){
    uchar[3*idx] = min(max(255*(cdf[uchar[3*idx]]-minVal)/(1-minVal), 0.0), 255.0);
    output[3*idx] = float(uchar[3*idx]/255.0);

    uchar[3*idx + 1] = min(max(255*(cdf[uchar[3*idx + 1]]-minVal)/(1-minVal), 0.0), 255.0);
    output[3*idx + 1] = float(uchar[3*idx + 1]/255.0);

    uchar[3*idx + 2] = min(max(255*(cdf[uchar[3*idx + 2]]-minVal)/(1-minVal), 0.0), 255.0);
    output[3*idx + 2] = float(uchar[3*idx + 2]/255.0);
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
  float* deviceInput;
  float* deviceOutput;
  unsigned char * deviceUCharImg;
  unsigned char * deviceGreyImg;
  int * deviceHistogram;
  float * deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(imageHeight * imageWidth * imageChannels * sizeof(float));

  cudaMalloc(&deviceHistogram, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc(&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc(&deviceInput, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc(&deviceOutput, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc(&deviceUCharImg, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  cudaMalloc(&deviceGreyImg, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));

  cudaMemcpy(deviceInput, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(int));
  cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));

  dim3 BlockDim(HISTO_SIZE, HISTO_SIZE, 1);
  dim3 GridDim((imageWidth - 1)/HISTO_SIZE + 1, (imageHeight - 1)/HISTO_SIZE + 1, 1);
  dim3 GridDimHist((imageHeight * imageWidth - 1)/HISTOGRAM_LENGTH + 1, 1, 1);
  dim3 BlockDimHist(HISTOGRAM_LENGTH, 1, 1);
  dim3 BlockDimScan(HISTOGRAM_LENGTH, 1, 1);
  dim3 GridDimScan(1, 1, 1);

  pixelConversion<<<GridDim, BlockDim>>> (deviceInput, deviceUCharImg, deviceGreyImg, imageWidth, imageHeight);
  Histogram<<<GridDimHist, BlockDimHist>>>(deviceGreyImg, deviceHistogram, imageWidth * imageHeight);
  histoScan<<<GridDimScan, BlockDimScan>>>(deviceHistogram, deviceCDF, imageWidth * imageHeight);
  Equalization<<<GridDim, BlockDim>>>(deviceUCharImg, deviceCDF, deviceOutput, imageWidth, imageHeight);

  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostInputImageData); free(hostOutputImageData);
  cudaFree(deviceGreyImg); cudaFree(deviceHistogram);
  cudaFree(deviceCDF); cudaFree(deviceInput);
  cudaFree(deviceOutput); cudaFree(deviceUCharImg);
  return 0;
}
