#pragma once

#include <fstream>
#include <iostream>
#include <string>

struct CpuUsage
{
    int64_t user;
    int64_t nice;
    int64_t system;
    int64_t idle;
    int64_t iowait;
};

CpuUsage read_cpu_usage()
{
    std::ifstream ifs("/proc/stat");
    std::string line;

    if (ifs.fail())
    {
        std::cerr << "failed to read /proc/stat " << std::endl;
        exit(EXIT_FAILURE);
    }

    std::getline(ifs, line);

    CpuUsage cpu_stat{};
    sscanf(line.data(),
           "cpu  %ld %ld %ld %ld %ld",
           &cpu_stat.user,
           &cpu_stat.nice,
           &cpu_stat.system,
           &cpu_stat.idle,
           &cpu_stat.iowait);

    return cpu_stat;
}

CpuUsage diff_cpu_usages(const CpuUsage &a, const CpuUsage &b)
{
    return CpuUsage{
        .user = b.user - a.user,
        .nice = b.nice - a.nice,
        .system = b.system - a.system,
        .idle = b.idle - a.idle,
        .iowait = b.iowait - a.iowait,
    };
}
