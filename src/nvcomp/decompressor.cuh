
#pragma once

#include <nvcomp.h>
#include <nvcomp/zstd.h>
#include "common/primitive_c.cu"
#include "common/primitive_cuda.cu"
#include <vector>

class Decompressor {

public:
    Decompressor(size_t max_nbatch, size_t max_uncompressed_size) {
        max_nbatch_ = max_nbatch;
        max_uncompressed_size_ = max_uncompressed_size;

        host_uncompressed_bytes_ = static_cast<size_t*>(mb_cuda_host_alloc(max_nbatch * sizeof(size_t)));
        host_uncompressed_ptrs_ = static_cast<void**>(mb_cuda_host_alloc(max_nbatch * sizeof(void*)));

        device_uncompressed_bytes_ = static_cast<size_t*>(mb_cuda_alloc(max_nbatch * sizeof(size_t)));
        device_uncompressed_ptrs_ = static_cast<void**>(mb_cuda_alloc(max_nbatch * sizeof(void*)));

        device_actual_uncompressed_bytes_ = static_cast<size_t*>(mb_cuda_alloc(max_nbatch * sizeof(size_t)));

        host_compressed_bytes_ = static_cast<size_t*>(mb_cuda_host_alloc(max_nbatch * sizeof(size_t)));
        host_compressed_ptrs_ = static_cast<void**>(mb_cuda_host_alloc(max_nbatch * sizeof(void*)));

        device_compressed_bytes_ = static_cast<size_t*>(mb_cuda_alloc(max_nbatch * sizeof(size_t)));
        device_compressed_ptrs_ = static_cast<void**>(mb_cuda_alloc(max_nbatch * sizeof(void*)));

        device_statuses_ = static_cast<nvcompStatus_t*>(mb_cuda_alloc(max_nbatch * sizeof(nvcompStatus_t)));

        nvcompStatus_t status = nvcompBatchedZstdDecompressGetTempSize(
            max_nbatch,
            max_uncompressed_size,
            &decompress_temp_bytes_);
        if (status != nvcompSuccess) {
            std::cerr << "Failed to get temp size for decompression" << std::endl;
            exit(EXIT_FAILURE);
        }

        device_decompress_temp_ = static_cast<void*>(mb_cuda_alloc(decompress_temp_bytes_));
    }


    ~Decompressor() {
        mb_cuda_host_free(host_uncompressed_bytes_);
        mb_cuda_host_free(host_uncompressed_ptrs_);

        mb_cuda_free(device_uncompressed_bytes_);
        mb_cuda_free(device_uncompressed_ptrs_);

        mb_cuda_free(device_actual_uncompressed_bytes_);

        mb_cuda_host_free(host_compressed_bytes_);
        mb_cuda_host_free(host_compressed_ptrs_);

        mb_cuda_free(device_compressed_bytes_);
        mb_cuda_free(device_compressed_ptrs_);

        mb_cuda_free(device_statuses_);
        mb_cuda_free(device_decompress_temp_);
    }


    void set_input_async(
        std::vector<void*> &device_input_ptrs,
        std::vector<size_t> &device_input_sizes,
        CUstream stream
    ) {
        size_t i;
        size_t nbatch = std::min(device_input_ptrs.size(), max_nbatch_);

        for (i = 0; i < nbatch; i++) {
            host_compressed_bytes_[i] = device_input_sizes[i];
            host_compressed_ptrs_[i] = device_input_ptrs[i];

            #if 0
            std::cout << "host_compressed_bytes_[" << i << "]: " << host_compressed_bytes_[i] 
                << ", host_compressed_ptrs_[" << i << "]: " << host_compressed_ptrs_[i] << std::endl;
            #endif
        }
        mb_cuda_memcpy_host_to_device_async(
            device_compressed_bytes_,
            host_compressed_bytes_,
            nbatch * sizeof(size_t),
            stream);

        mb_cuda_memcpy_host_to_device_async(
            device_compressed_ptrs_,
            host_compressed_ptrs_,
            nbatch * sizeof(void*),
            stream);
    }


    // One shot initialization when the workers are created
    void set_output_buffer_async(
        std::vector<void*> &device_output_ptrs,
        CUstream stream
    ) {
        size_t i;
        size_t nbatch = std::min(device_output_ptrs.size(), max_nbatch_);
        size_t chunk_size = max_uncompressed_size_;

        // handling the case that the size of table records is smaller than page size
        for (i = 0; i < nbatch; i++) {
            host_uncompressed_bytes_[i] = chunk_size;
            host_uncompressed_ptrs_[i] = device_output_ptrs[i];

            #if 0
            std::cout << "host_uncompressed_bytes_[" << i << "]: " << host_uncompressed_bytes_[i] 
                << ", host_uncompressed_ptrs_[" << i << "]: " << host_uncompressed_ptrs_[i] << std::endl;
            #endif
        }

        mb_cuda_memcpy_host_to_device_async(
            device_uncompressed_bytes_,
            host_uncompressed_bytes_,
            nbatch * sizeof(size_t),
            stream);

        mb_cuda_memcpy_host_to_device_async(
            device_uncompressed_ptrs_,
            host_uncompressed_ptrs_,
            nbatch * sizeof(void*),
            stream);
    }

    // void decompress(const void* input, size_t input_size, void* output, size_t output_size);
    void decompress_async(size_t nbatch, CUstream stream) {
        // debug_print();

        nvcompStatus_t status = nvcompBatchedZstdDecompressAsync(
            device_compressed_ptrs_,
            device_compressed_bytes_,
            device_uncompressed_bytes_,
            device_actual_uncompressed_bytes_,
            nbatch, // NOTE: use nbatch instead of max_nbatch_
            device_decompress_temp_,
            decompress_temp_bytes_,
            device_uncompressed_ptrs_,
            device_statuses_,
            stream);

        if (status != nvcompSuccess) {
            std::cerr << "Failed to decompress" << std::endl;
            switch (status) {
            case nvcompErrorInvalidValue:
                std::cerr << "nvcompErrorInvalidValue" << std::endl;
                break;
            case nvcompErrorNotSupported:
                std::cerr << "nvcompErrorNotSupported" << std::endl;
                break;
            case nvcompErrorCannotDecompress:
                std::cerr << "nvcompErrorCannotDecompress" << std::endl;
                break;
            case nvcompErrorBadChecksum:
                std::cerr << "nvcompErrorBadChecksum" << std::endl;
                break;
            case nvcompErrorCannotVerifyChecksums:
                std::cerr << "nvcompErrorCannotVerifyChecksums" << std::endl;
                break;
            case nvcompErrorWrongHeaderLength:
                std::cerr << "nvcompErrorWrongHeaderLength" << std::endl;
                break;
            case nvcompErrorChunkSizeTooLarge:
                std::cerr << "nvcompErrorChunkSizeTooLarge" << std::endl;
                break;
            case nvcompErrorCudaError:
                std::cerr << "nvcompErrorCudaError" << std::endl;
                break;
            default:
                break;
            }
            exit(EXIT_FAILURE);
        }
    }

    void debug_print() {
        size_t i;
        std::cout << "max_nbatch_: " << max_nbatch_ << std::endl;

        for (i = 0; i < max_nbatch_; i++) {
            std::cout << "host_compressed_bytes_[" << i << "]: " << host_compressed_bytes_[i] 
                << ", host_compressed_ptrs_[" << i << "]: " << host_compressed_ptrs_[i] << std::endl;
        }

        for (i = 0; i < max_nbatch_; i++) {
            std::cout << "host_uncompressed_bytes_[" << i << "]: " << host_uncompressed_bytes_[i] 
                << ", host_compressed_ptrs_[" << i << "]: " << host_uncompressed_ptrs_[i] << std::endl;
        }
        std::cout << "decompress_temp_bytes_:" << decompress_temp_bytes_ << std::endl;

        //    device_compressed_ptrs_,
        //    device_compressed_bytes_,
        //    device_uncompressed_bytes_,
        //    device_actual_uncompressed_bytes_,
        //    nbatch, // NOTE: use nbatch instead of max_nbatch_
        //    device_decompress_temp_,
        //    decompress_temp_bytes_,
        //    device_uncompressed_ptrs_,
        //    device_statuses_,
 


    }

private:
    size_t max_nbatch_;
    size_t max_uncompressed_size_;

    /* host */
    size_t *host_uncompressed_bytes_;
    void **host_uncompressed_ptrs_;
    size_t *device_uncompressed_bytes_;
    void **device_uncompressed_ptrs_;

    size_t *host_compressed_bytes_;
    void **host_compressed_ptrs_;
    size_t *device_compressed_bytes_;
    void **device_compressed_ptrs_;
    
    size_t *device_actual_uncompressed_bytes_;
    nvcompStatus_t *device_statuses_;
    size_t decompress_temp_bytes_;
    void *device_decompress_temp_;

    // host_uncompressed_bytes: *mut usize,
    // host_uncompressed_ptrs: *mut u64,
    // device_uncompressed_bytes: DeviceMutPtr<usize>,
    // device_uncompressed_ptrs: DeviceMutPtr<DeviceMutPtr<u8>>,
    // host_compressed_bytes: *mut usize,
    // host_compressed_ptrs: *mut u64,
    // device_compressed_bytes: DeviceMutPtr<usize>,
    // device_compressed_ptrs: DeviceMutPtr<DeviceConstPtr<u8>>,
    // device_actual_uncompressed_bytes: DeviceMutPtr<usize>,
    // device_statuses: DeviceMutPtr<nvcomp_sys::nvcompStatus_t>,
    // decomp_temp_bytes: Option<usize>,
    // device_decomp_temp: Option<DeviceMutPtr<u8>>,
};