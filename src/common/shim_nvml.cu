#pragma once

#include <unistd.h>

#include <cstdio>

#include <nvml.h>

namespace nvml
{
    namespace
    {
        void check(nvmlReturn_t result)
        {
            if (result != NVML_SUCCESS)
            {
                fprintf(stderr, "NVML: %s\n", nvmlErrorString(result));
                exit(EXIT_FAILURE);
            }
        }
    }

    void init()
    {
        check(nvmlInit());
    }

    void shutdown()
    {
        check(nvmlShutdown());
    }

    unsigned int device_get_count()
    {
        unsigned int count = 0;
        check(nvmlDeviceGetCount(&count));
        return count;
    }

    nvmlDevice_t device_get_handle_by_index(unsigned int index)
    {
        nvmlDevice_t device;
        check(nvmlDeviceGetHandleByIndex(index, &device));
        return device;
    }

    nvmlUtilization_t device_get_utilization_rates(nvmlDevice_t device)
    {
        nvmlUtilization_t utilization;
        check(nvmlDeviceGetUtilizationRates(device, &utilization));
        return utilization;
    }

    // Per-process utilization: returns the most recent sample for the given PID.
    // If no sample is found for the PID, smUtil and memUtil are set to 0.
    nvmlProcessUtilizationSample_t device_get_process_utilization(nvmlDevice_t device, unsigned int target_pid)
    {
        nvmlProcessUtilizationSample_t result_sample{};
        result_sample.pid = target_pid;
        result_sample.smUtil = 0;
        result_sample.memUtil = 0;

        // First call: query required buffer size
        unsigned int count = 0;
        auto ret = nvmlDeviceGetProcessUtilization(device, nullptr, &count, 0);
        if (ret == NVML_ERROR_NOT_FOUND || count == 0)
        {
            // No processes with GPU activity
            return result_sample;
        }
        if (ret != NVML_ERROR_INSUFFICIENT_SIZE && ret != NVML_SUCCESS)
        {
            fprintf(stderr, "NVML GetProcessUtilization (size query): %s\n", nvmlErrorString(ret));
            return result_sample;
        }

        // Second call: fetch samples
        std::vector<nvmlProcessUtilizationSample_t> samples(count);
        ret = nvmlDeviceGetProcessUtilization(device, samples.data(), &count, 0);
        if (ret == NVML_ERROR_NOT_FOUND || count == 0)
        {
            return result_sample;
        }
        if (ret != NVML_SUCCESS)
        {
            fprintf(stderr, "NVML GetProcessUtilization: %s\n", nvmlErrorString(ret));
            return result_sample;
        }

        // Find the most recent sample for target_pid
        unsigned long long latest_ts = 0;
        for (unsigned int i = 0; i < count; i++)
        {
            if (samples[i].pid == target_pid && samples[i].timeStamp > latest_ts)
            {
                latest_ts = samples[i].timeStamp;
                result_sample = samples[i];
            }
        }
        return result_sample;
    }
}
