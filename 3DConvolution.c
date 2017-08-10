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
#define KERNEL_WIDTH 3
#define KERNEL_RADIUS 1
#define TILE_SIZE 8

//@@ Define constant memory for device kernel here
__constant__ float M[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

  __shared__ float ds_I[TILE_SIZE + KERNEL_WIDTH - 1][TILE_SIZE + KERNEL_WIDTH - 1][TILE_SIZE + KERNEL_WIDTH - 1];

  float res = 0.0;
  int z_out = bz * TILE_SIZE + tz - KERNEL_RADIUS;
  int y_out = by * TILE_SIZE + ty - KERNEL_RADIUS;
  int x_out = bx * TILE_SIZE + tx - KERNEL_RADIUS;

  if((x_out >= 0) && (x_out < x_size) && (y_out >= 0) && (y_out < y_size) && (z_out >= 0) && (z_out < z_size)){
    ds_I[tz][ty][tx] = input[(z_out * y_size * x_size) + (y_out * x_size) + x_out];
  }
  else{ ds_I[tz][ty][tx] = 0.0; }

  __syncthreads();

  x_out += 1; y_out += 1; z_out += 1;

  if(tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE){
    for(int i = 0; i < KERNEL_WIDTH; i++){
      for(int j = 0; j < KERNEL_WIDTH; j++){
        for(int k = 0; k < KERNEL_WIDTH; k++){
            if((x_out >= 0) && (x_out < x_size) && (y_out >= 0) && (y_out < y_size) && (z_out >= 0) && (z_out < z_size)){
              res += M[i][j][k] * ds_I[tz + i][ty + j][tx + k];
            }
        }
      }
    }

    if(z_out < z_size && y_out < y_size && x_out < x_size){
      output[(z_out * y_size * x_size) + (y_out * x_size) + x_out] = res;
    }
  }
  __syncthreads();
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
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc(&deviceInput, (z_size*y_size*x_size)*sizeof(float));
  cudaMalloc(&deviceOutput, (z_size*y_size*x_size)*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (z_size*y_size*x_size)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M, hostKernel, KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid((x_size - 1)/TILE_SIZE + 1, (y_size - 1)/TILE_SIZE + 1, (z_size - 1)/TILE_SIZE + 1);
  dim3 DimBlock(TILE_SIZE + KERNEL_WIDTH - 1, TILE_SIZE + KERNEL_WIDTH - 1, TILE_SIZE + KERNEL_WIDTH - 1);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, x_size * y_size * z_size * sizeof(float), cudaMemcpyDeviceToHost);
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
