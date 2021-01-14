#include <cmath>
#include <iostream>
#include "cuda_fp16.h"
#include "gpu-new-forward.h"


#define T_WIDTH 32
#define BT_WIDTH 8
#define BLOCKDIM 512
#define MINI_BATCH_SIZE 2000


__global__ void unroll(float* x, float* output, int C, int K, int H, int W, int b) {
    // variables used later
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int x_griddim = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = x_griddim % H_out;
    int w_out = x_griddim / H_out;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z / MINI_BATCH_SIZE;
    int bb = blockIdx.z % MINI_BATCH_SIZE;

    if (h_out < H_out && w_out < W_out && c < C && kk < K * K) {
        int w_base = c * K * K * H_out * W_out;
        int p = kk % K;
        int q = kk / K;
        output[bb*H_out*W_out*K*K*C+ w_base + (p * K + q) * H_out * W_out + h_out * W_out + w_out] = x4d(MINI_BATCH_SIZE*b + bb, c, (h_out + p), (w_out + q));
    }
    
    #undef x4d
}

// A:
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns, int b) {
      //@@ Insert code to implement matrix multiplication here
      //@@ You have to use shared memory for this MP
    const int TILE_WIDTH = 16;
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int bb = blockIdx.z;

    float Pvalue = 0;
    float Width = numAColumns;

  
    for (int q = 0; q < ceil(Width/TILE_WIDTH); ++q) {
        if (Row < numARows && (q * TILE_WIDTH+tx) < numAColumns) {
            subTileA[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
        } else {
            subTileA[ty][tx] = 0;
        }
        
        if (Col < numBColumns && (q * TILE_WIDTH+ty) < numBRows) {
            subTileB[ty][tx] = B[bb * numBColumns * numBRows + (q*TILE_WIDTH+ty) * numBColumns + Col];
        } else {
            subTileB[ty][tx] = 0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += subTileA[ty][k] * subTileB[k][tx]; //add pragma
        }
        __syncthreads();
    }
        
    if (Row < numCRows && Col < numCColumns) {
        C[(MINI_BATCH_SIZE*b+bb)*numCColumns*numCRows + Row * numCColumns + Col] = Pvalue;
    }
}

__host__ void conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    float* device_y;
    float* device_x;
    float* device_k;
    float *device_unrolled_x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    
    size_t unrolled_x_size = MINI_BATCH_SIZE * W_out * H_out * C * K * K * sizeof(float);

    cudaMalloc((void**)(&device_unrolled_x), unrolled_x_size);
    cudaMalloc((void**)(&device_y), B * M * W_out * H_out * sizeof(float));
    cudaMalloc((void**)(&device_x), B * C * H * W * sizeof(float));
    cudaMalloc((void**)(&device_k), M * C * K * K *sizeof(float));
    
    cudaMemcpy(device_x, host_x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_k_cols = K * K * C;
    int num_x_cols = W_out * H_out;
    dim3 gridDim(ceil((H_out * W_out*1.0)/16), ceil((K * K * 1.0)/16), MINI_BATCH_SIZE * C);
    dim3 blockDim(16, 16, 1);
    
    dim3 dimMult(ceil((1.0*num_x_cols)/16), ceil((1.0*M)/16), MINI_BATCH_SIZE);
    dim3 dimMultB(16, 16, 1);
        
    for (int i = 0; i < ceil(1.0*B/MINI_BATCH_SIZE); i += 1) {
        unroll<<<gridDim, blockDim>>>(device_x, device_unrolled_x, C, K, H, W, i);
        cudaDeviceSynchronize();
        matrixMultiplyShared<<<dimMult, dimMultB>>>(device_k, device_unrolled_x, device_y, M, num_k_cols, num_k_cols, num_x_cols, M, num_x_cols, i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_y, device_y, B * M * (H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
    cudaFree(device_unrolled_x);

    //cudaFree(device_k);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
