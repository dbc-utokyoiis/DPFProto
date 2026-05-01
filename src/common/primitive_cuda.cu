#pragma once

#include <cstdio>

#include "helper_cuda.h"
#include <cuda.h>

inline void checkCudaResult(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        const char *errorName;
        const char *errorString;
        cuGetErrorName(result, &errorName);
        cuGetErrorString(result, &errorString);
        fprintf(stderr, "CUDA Driver API Error: %s: %s\n", errorName, errorString);
        exit(EXIT_FAILURE);
    }
}

inline void mb_cuda_init()
{
    checkCudaResult(cuInit(0));
}

inline CUdevice mb_cuda_get_device(int i)
{
    CUdevice device;
    checkCudaResult(cuDeviceGet(&device, i));

    return device;
}

inline CUcontext mb_cuda_new_context(CUdevice device)
{
    CUcontext ctx;
    checkCudaResult(cuDevicePrimaryCtxRetain(&ctx, device));
    return ctx;
}

inline void mb_cuda_set_context(CUcontext ctx)
{
    checkCudaResult(cuCtxSetCurrent(ctx));
}

//inline void mb_cuda_destroy_context(CUcontext ctx)
//{
//    checkCudaResult(cuCtxDestroy(ctx));
//}

// inline void mb_cuda_release_device(CUdevice dev)
// {
//     checkCudaResult(cuDevicePrimaryCtxRelease(dev));
// }

inline void *mb_cuda_alloc(size_t size)
{
    void *dev_ptr = nullptr;
    checkCudaErrors(cudaMalloc(&dev_ptr, size));
    checkCudaErrors(cudaMemset(dev_ptr, 0x00, size));
    return dev_ptr;
}

inline void mb_cuda_free(void *dev_ptr)
{
    checkCudaErrors(cudaFree(dev_ptr));
}

inline CUdeviceptr mb_cuda_alloc_v2(size_t size)
{
    CUdeviceptr dev_ptr;
    checkCudaResult(cuMemAlloc_v2(&dev_ptr, size));
    // checkCudaErrors(cudaMemset(dev_ptr, 0x00, size));
    return dev_ptr;
}

inline void mb_cuda_free_v2(CUdeviceptr dev_ptr)
{
    checkCudaResult(cuMemFree_v2(dev_ptr));
}

inline void *mb_cuda_host_alloc(size_t size)
{
    void *host_ptr = nullptr;
    checkCudaErrors(cudaHostAlloc(&host_ptr, size, 0));
    memset(host_ptr, 0x00, size);
    return host_ptr;
}

inline void mb_cuda_host_free(void *host_ptr)
{
    checkCudaErrors(cudaFreeHost(host_ptr));
}

inline void mb_cuda_event_create(CUevent* ev)
{
    checkCudaResult(cuEventCreate(ev, CU_EVENT_DISABLE_TIMING));
}

inline void mb_cuda_event_destroy(CUevent ev)
{
    checkCudaResult(cuEventDestroy(ev));
}

inline void mb_cuda_event_wait_event(CUstream hStream, CUevent ev)
{
    checkCudaResult(cuStreamWaitEvent(hStream, ev, CU_EVENT_WAIT_DEFAULT));
}

inline void mb_cuda_memcpy_host_to_device(void *dst, const void *src, size_t count)
{
    checkCudaErrors(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

inline void mb_cuda_memcpy_device_to_host(void *dst, const void *src, size_t count)
{
    checkCudaErrors(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

inline cudaStream_t mb_cuda_stream_create()
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return stream;
}

inline cudaStream_t mb_cuda_stream_create_with_nonbloking_flag()
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
}

inline void mb_cuda_stream_destroy(cudaStream_t stream)
{
    checkCudaErrors(cudaStreamDestroy(stream));
}

inline void mb_cuda_stream_synchronize(cudaStream_t stream)
{
    checkCudaErrors(cudaStreamSynchronize(stream));
}

inline void mb_cuda_memcpy_host_to_device_async(void *dst, const void *src, size_t count, cudaStream_t stream)
{
    checkCudaErrors(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
}

inline void mb_cuda_memcpy_device_to_host_async(void *dst, const void *src, size_t count, cudaStream_t stream)
{
    checkCudaErrors(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
}
