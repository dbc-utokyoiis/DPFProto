#pragma once

#include <cstdio>

#include <cufile.h>

#define CUFILE_IO_ALIGN (512)
#define checkCuFileErrors(val) check_cufile_errors((val), #val, __FILE__, __LINE__)

void check_cufile_errors(CUfileError_t status, const char *const func, const char *const file, const int line)
{
    if (status.err != CU_FILE_SUCCESS)
    {
        fprintf(stderr, "CUFILE error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, status.err, cufileop_status_error(status.err), func);
        exit(EXIT_FAILURE);
    }
}

const char *cufile_status_to_string(CUfileStatus_t status)
{
    switch (status)
    {
    case CUFILE_WAITING:
        return "CUFILE_WAITING";
    case CUFILE_PENDING:
        return "CUFILE_PENDING";
    case CUFILE_INVALID:
        return "CUFILE_INVALID";
    case CUFILE_CANCELED:
        return "CUFILE_CANCELED";
    case CUFILE_COMPLETE:
        return "CUFILE_COMPLETE";
    case CUFILE_TIMEOUT:
        return "CUFILE_TIMEOUT";
    case CUFILE_FAILED:
        return "CUFILE_FAILED";
    default:
        return "UNKNOWN";
    }
}

inline void mb_cufile_driver_open()
{
    checkCuFileErrors(cuFileDriverOpen());
}

inline void mb_cufile_driver_close()
{
    checkCuFileErrors(cuFileDriverClose());
}

inline CUfileHandle_t mb_cufile_handle_register(int fd)
{
    CUfileDescr_t descr;
    memset((void *)&descr, 0, sizeof(CUfileDescr_t));
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t fh;
    checkCuFileErrors(cuFileHandleRegister(&fh, &descr));

    return fh;
}

inline void mb_cufile_handle_deregister(CUfileHandle_t fh)
{
    (void)cuFileHandleDeregister(fh);
}

inline void mb_cufile_buf_register(const void *dev_ptr_base, size_t length)
{
    checkCuFileErrors(cuFileBufRegister(dev_ptr_base, length, 0));
}

inline void mb_cufile_buf_deregister(const void *dev_ptr_base)
{
    checkCuFileErrors(cuFileBufDeregister(dev_ptr_base));
}
