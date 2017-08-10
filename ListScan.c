// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this
#define NUM_THREADS 256
#define SECTION_SIZE 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *S) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here

  //Three kernel hierarchical approach
  //Kernel 1 -> Three Phase Kernel
  //Kernel 2 -> Parallel Scan on S array
  //Kernel 3 -> Add S array to output array elements not in last index of section block
  //Load in shared memory (Copy & paste from 5.1)
  __shared__ float XY[SECTION_SIZE];
  int tx = threadIdx.x; int bx = blockIdx.x; int bd = blockDim.x;
  int index = 2*bx*bd + tx;

  if(index + BLOCK_SIZE >= len){
    if(index >= len){
      XY[tx] = 0;
      XY[tx + BLOCK_SIZE] = 0;
    }
    else{
      XY[tx] = input[index];
      XY[tx + BLOCK_SIZE] = 0;
    }
  }
  else{
    XY[tx] = input[index];
    XY[tx + BLOCK_SIZE] = input[index + BLOCK_SIZE];
  }

  //Implement Brent-Kung Kernel from Chapter 6
  for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
    __syncthreads();

    int idx = (tx + 1) * 2 * stride - 1;
    if(idx < SECTION_SIZE){
      XY[idx] += XY[idx - stride];
    }
  }

  for(int stride = SECTION_SIZE/4; stride > 0; stride /= 2){
    __syncthreads();
    int idx = (tx + 1) * 2 * stride - 1;
    if(idx + stride < SECTION_SIZE){
      XY[idx + stride] += XY[idx];
    }
  }

  __syncthreads();
  if(index < len){
    output[index] = XY[tx];
    if(index + BLOCK_SIZE < len){
      output[index + BLOCK_SIZE] = XY[tx + BLOCK_SIZE];
      if(tx == BLOCK_SIZE - 1){
        //This can be done by changing the code at the end of the scan kernel
        //so that the last thread of each block writes its result into an
        //S array using its blockIdx.x as index
        S[bx] = XY[tx + BLOCK_SIZE];
      }
    }
  }
}

//Implement Brent-Kung Kernel from Chapter 6 onto S array
__global__ void phase2(float *input, float *output, int len){
  __shared__ float XY[SECTION_SIZE];
  int tx = threadIdx.x; int bx = blockIdx.x; int bd = blockDim.x;

  //No need for boundary conditions when loading into shared memory
  XY[tx] = input[tx];
  XY[tx + BLOCK_SIZE] = input[tx + BLOCK_SIZE];

  //Implement Brent-Kung Kernel from Chapter 6
  for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
    __syncthreads();

    int idx = (tx + 1) * 2 * stride - 1;
    if(idx < SECTION_SIZE){
      XY[idx] += XY[idx - stride];
    }
  }

  for(int stride = SECTION_SIZE/4; stride > 0; stride /= 2){
    __syncthreads();
    int idx = (tx + 1) * 2 * stride - 1;
    if(idx + stride < SECTION_SIZE){
      XY[idx + stride] += XY[idx];
    }
  }

  __syncthreads();
  //Output modified XY, reverse of code when loading input to shared memory
  output[tx] = XY[tx];
  output[tx + BLOCK_SIZE] = XY[tx + BLOCK_SIZE];
}

//The third kernel takes the Sarray and Yarray as inputs and writes its output back into Y.
__global__ void distribute(float *Y, float *S, int len){
  int tx = threadIdx.x; int bx = blockIdx.x; int bd = blockDim.x;
  int index = 2*bx*bd + tx;

  if(bx > 0 && index < len){
    Y[index] += S[bx - 1];
    if(index + BLOCK_SIZE < len){
      Y[index + BLOCK_SIZE] += S[bx - 1];
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *S;
  float *Y;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  cudaMalloc((void **)&S, SECTION_SIZE * sizeof(float));
  cudaMalloc((void **)&Y, SECTION_SIZE * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  cudaMemset(S, 0, SECTION_SIZE * sizeof(float));
  cudaMemset(Y, 0, SECTION_SIZE * sizeof(float));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numElements)/2 * BLOCK_SIZE, 1, 1);
  dim3 DimBlock(NUM_THREADS, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, S);
  cudaDeviceSynchronize();
  phase2<<<1, DimBlock>>>(S, Y, numElements);
  cudaDeviceSynchronize();
  distribute<<<DimGrid, DimBlock>>>(deviceOutput, Y, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(S); cudaFree(Y);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
