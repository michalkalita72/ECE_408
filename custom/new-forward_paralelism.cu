#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define T_WIDTH 16

// __constant__ float k[16*4*7*7];

__global__ void conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K, float* k) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function parameter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.z;
    int b = blockIdx.y*T_WIDTH + threadIdx.y;
    int coor = blockIdx.x*T_WIDTH + threadIdx.x;
    int w = coor % W_out;
    int h = coor / W_out;
    if ((h < H_out) && (w < W_out) && b < B) {
        //for (int b = 0; b < B; b++) {
            float temp = 0;
            for (int c = 0; c < C; c++) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        temp += x4d(b, c, h + p, w + q) * k4d(m ,c, p, q); //k[(m) * (C * K * K) + (c) * (K * K) + (p) * (K) + q]; k4d(m ,c, p, q);
                    }
                }
            }
            y4d(b, m, h, w) = temp;
        //}
    }

#undef y4d
#undef x4d
#undef k4d
}

__host__ void conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float* device_y;
    float* device_x;
    float* device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**)(&device_y), B * M * (H-K+1)*(W-K+1)*sizeof(float));
    cudaMalloc((void**)(&device_x), B * C * H * W * sizeof(float));
    cudaMalloc((void**)(&device_k), M * C * K * K *sizeof(float));

    cudaMemcpy(device_x, host_x,  B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, M * C * K * K *sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(k, host_k, M * C * K * K *sizeof(float));

    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(ceil((1.0*(W-K+1)*(H-K+1))/T_WIDTH), ceil((1.0*B)/T_WIDTH), M);
    dim3 dimBlock(T_WIDTH, T_WIDTH, 1);

    conv_forward_kernel<<<dimGrid,dimBlock>>>(device_y, device_x, B, M, C, H, W, K, device_k);


    // Copy the output back to host
    cudaMemcpy(host_y, device_y, B * M * (H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);
    


    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

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

