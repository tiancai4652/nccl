/*
 * 测试fake_cuda实现的简单程序
 */
#include <stdio.h>
#include <stdlib.h>
#include "fake_cuda/include/cuda_runtime.h"

int main() {
    printf("=== 测试NCCL CPU化fake_cuda实现 ===\n");
    
    // 测试设备数量获取
    int deviceCount = 0;
    cudaError_t result = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount: %s, 设备数量: %d\n", 
           cudaGetErrorString(result), deviceCount);
    
    if (deviceCount > 0) {
        // 测试设备设置
        result = cudaSetDevice(0);
        printf("cudaSetDevice(0): %s\n", cudaGetErrorString(result));
        
        // 测试获取当前设备
        int currentDevice = -1;
        result = cudaGetDevice(&currentDevice);
        printf("cudaGetDevice: %s, 当前设备: %d\n", 
               cudaGetErrorString(result), currentDevice);
        
        // 测试设备属性获取
        cudaDeviceProp prop;
        result = cudaGetDeviceProperties(&prop, 0);
        printf("cudaGetDeviceProperties: %s\n", cudaGetErrorString(result));
        printf("设备名称: %s\n", prop.name);
        printf("总显存: %zu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("计算能力: %d.%d\n", prop.major, prop.minor);
        
        // 测试内存分配
        void* devPtr = NULL;
        result = cudaMalloc(&devPtr, 1024);
        printf("cudaMalloc(1024): %s, 指针: %p\n", 
               cudaGetErrorString(result), devPtr);
        
        if (devPtr) {
            result = cudaFree(devPtr);
            printf("cudaFree: %s\n", cudaGetErrorString(result));
        }
        
        // 测试流创建
        cudaStream_t stream;
        result = cudaStreamCreate(&stream);
        printf("cudaStreamCreate: %s, 流: %p\n", 
               cudaGetErrorString(result), stream);
        
        if (stream) {
            result = cudaStreamDestroy(stream);
            printf("cudaStreamDestroy: %s\n", cudaGetErrorString(result));
        }
    }
    
    printf("\n=== fake_cuda测试完成 ===\n");
    return 0;
}
