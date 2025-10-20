/*************************************************************************
 * Copyright (c) 2025, NCCL-CPU-ADAPTATION Project. All rights reserved.
 *
 * Fake CUDA Implementation for CPU-only NCCL
 ************************************************************************/

#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// 虚拟GPU设备数量（通过环境变量控制）
static int getGpuDeviceCount() {
    const char* env = getenv("GPU_DEV_NUM");
    return env ? atoi(env) : 4;  // 默认4个虚拟GPU
}

// 错误字符串映射
static const char* cudaGetErrorName_impl(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
        case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
        case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
        case cudaErrorArrayIsMapped: return "cudaErrorArrayIsMapped";
        case cudaErrorAlreadyMapped: return "cudaErrorAlreadyMapped";
        case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
        case cudaErrorAlreadyAcquired: return "cudaErrorAlreadyAcquired";
        case cudaErrorNotMapped: return "cudaErrorNotMapped";
        default: return "cudaErrorUnknown";
    }
}

static const char* cudaGetErrorString_impl(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInitializationError: return "initialization error";
        case cudaErrorLaunchFailure: return "unspecified launch failure";
        case cudaErrorLaunchTimeout: return "launch timeout";
        case cudaErrorLaunchOutOfResources: return "launch out of resources";
        case cudaErrorInvalidDeviceFunction: return "invalid device function";
        case cudaErrorInvalidConfiguration: return "invalid configuration argument";
        case cudaErrorInvalidDevice: return "invalid device ordinal";
        case cudaErrorInvalidMemcpyDirection: return "invalid memcpy direction";
        default: return "unknown error";
    }
}

// 基本CUDA Runtime API实现
extern "C" {

cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    if (!count) return cudaErrorInvalidValue;
    *count = getGpuDeviceCount();
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
    if (device < 0 || device >= getGpuDeviceCount()) {
        return cudaErrorInvalidDevice;
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDevice(int *device) {
    if (!device) return cudaErrorInvalidValue;
    *device = 0;  // 始终返回设备0
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    if (!dst || !src) return cudaErrorInvalidValue;
    
    // 根据kind决定是否真的拷贝数据
    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
            memcpy(dst, src, count);
            break;
        default:
            return cudaErrorInvalidMemcpyDirection;
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    (void)stream;  // 忽略stream，同步执行
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    *devPtr = malloc(size);
    return *devPtr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    free(devPtr);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    *ptr = malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count) {
    if (!devPtr) return cudaErrorInvalidValue;
    memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    (void)stream;  // 忽略stream
    return cudaMemset(devPtr, value, count);
}

cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) {
    if (!pStream) return cudaErrorInvalidValue;
    *pStream = (cudaStream_t)0x1;  // 返回虚拟stream句柄
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetLastError(void) {
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaPeekAtLastError(void) {
    return cudaSuccess;
}

const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
    return cudaGetErrorString_impl(error);
}

const char* CUDARTAPI cudaGetErrorName(cudaError_t error) {
    return cudaGetErrorName_impl(error);
}

cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop) return cudaErrorInvalidValue;
    if (device < 0 || device >= getGpuDeviceCount()) {
        return cudaErrorInvalidDevice;
    }
    
    // 填充虚拟设备属性
    memset(prop, 0, sizeof(*prop));
    snprintf(prop->name, sizeof(prop->name), "Virtual GPU %d", device);
    prop->totalGlobalMem = 16ULL * 1024 * 1024 * 1024;  // 16GB
    prop->sharedMemPerBlock = 48 * 1024;  // 48KB
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->memPitch = 2147483647;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = 1500000;  // 1.5GHz
    prop->totalConstMem = 65536;
    prop->major = 8;  // 计算能力8.0
    prop->minor = 0;
    prop->textureAlignment = 512;
    prop->deviceOverlap = 1;
    prop->multiProcessorCount = 80;
    prop->kernelExecTimeoutEnabled = 0;
    prop->integrated = 0;
    prop->canMapHostMemory = 1;
    prop->computeMode = cudaComputeModeDefault;
    prop->concurrentKernels = 1;
    prop->ECCEnabled = 0;
    prop->pciBusID = device;
    prop->pciDeviceID = 0;
    prop->tccDriver = 0;
    
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) return cudaErrorInvalidValue;
    *driverVersion = 12000;  // CUDA 12.0
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) return cudaErrorInvalidValue;
    *runtimeVersion = 12000;  // CUDA 12.0
    return cudaSuccess;
}

// 其他必要的CUDA API
cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    if (!event) return cudaErrorInvalidValue;
    *event = (cudaEvent_t)0x2;  // 虚拟event句柄
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    (void)event;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    (void)event;
    (void)stream;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    (void)event;
    return cudaSuccess;
}

} // extern "C"
