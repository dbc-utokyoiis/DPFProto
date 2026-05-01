#pragma once

#include <iostream>
#include <unistd.h>

#include "shim_nvml.cu"

struct GpuUsagePerDevice
{
    unsigned int gpu;       // SM (compute) utilization %
    unsigned int memory;    // framebuffer memory utilization %
};

struct GpuUsage
{
    std::vector<GpuUsagePerDevice> devices;
};

GpuUsage get_gpu_usage()
{
    GpuUsage gpu_usage{};
    unsigned int pid = static_cast<unsigned int>(getpid());
    unsigned int n = nvml::device_get_count();
    for (unsigned int i = 0; i < n; i++)
    {
        auto handle = nvml::device_get_handle_by_index(i);
        auto sample = nvml::device_get_process_utilization(handle, pid);
        gpu_usage.devices.push_back(GpuUsagePerDevice{
            .gpu = sample.smUtil,
            .memory = sample.memUtil,
        });
    }
    return gpu_usage;
}
