#pragma once

#include <unistd.h>

#include <cstdio>

#define HOST_BUFFER_ALIGN (512)

inline void *mb_alloc(size_t size)
{
    // O_DIRECT requires the buffer to be 512-byte aligned
    void *buf = aligned_alloc(HOST_BUFFER_ALIGN, size);
    if (buf == nullptr)
    {
        perror("aligned_alloc");
        exit(EXIT_FAILURE);
    }

    memset(buf, 0, size);

    return buf;
}

inline void mb_pread(int fd, void *buf, size_t nbytes, off_t offset)
{
    ssize_t nread;
    ssize_t s;
    uint8_t *bufp = (uint8_t *)buf;

    nread = 0;
    s = 0;
    while (nread < nbytes)
    {
    retry:
        // std::cout << fd << ", " << nbytes << ", " << offset << std::endl;
        s = pread(fd, bufp + nread, nbytes - nread, offset + nread);
        if (s < 0)
        {
            switch (errno)
            {
            case EINTR:
                goto retry;
            default:
                perror("pread");
                exit(EXIT_FAILURE);
            }
        }
        nread += s;
    }
}

inline void mb_pwrite(int fd, void *buf, size_t nbytes, off_t offset)
{
    ssize_t nwritten;
    ssize_t s;
    uint8_t *bufp = (uint8_t *)buf;

    nwritten = 0;
    s = 0;
    while (nwritten < nbytes)
    {
    retry:
        s = pwrite(fd, bufp + nwritten, nbytes - nwritten, offset + nwritten);
        if (s < 0)
        {
            switch (errno)
            {
            case EINTR:
                goto retry;
            default:
                perror("pread");
                exit(EXIT_FAILURE);
            }
        }
        nwritten += s;
    }
}
