/*
 * GDS Alignment Test
 *
 * Tests whether cuFileRead succeeds with 512-byte aligned offsets/sizes
 * (vs the commonly assumed 4K alignment requirement).
 *
 * Results are logged to cufile.log for detailed GDS-level diagnostics.
 *
 * Usage:
 *   gds_test <file_or_device>
 *
 * Example:
 *   sudo ./build/gds_test /dev/nvme1n1p1
 *   sudo ./build/gds_test /path/to/some/file
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include "common/primitive_cuda.cu"
#include "common/primitive_cufile.cu"

struct AlignTest {
    const char* label;
    off_t  file_offset;   // offset in the file
    size_t read_size;     // bytes to read
    off_t  buf_offset;    // offset within registered GPU buffer
};

static void run_test(CUfileHandle_t fh, void* dev_buf, size_t buf_size,
                     const AlignTest& t)
{
    printf("  %-40s  file_off=%-8ld  size=%-8zu  buf_off=%-8ld  ... ",
           t.label, (long)t.file_offset, t.read_size, (long)t.buf_offset);
    fflush(stdout);

    // Clear buffer region first
    cudaMemset((char*)dev_buf + t.buf_offset, 0xCC, t.read_size);

    ssize_t nread = cuFileRead(fh, dev_buf, t.read_size,
                               t.file_offset, t.buf_offset);

    if (nread < 0) {
        // cuFileRead returns negative cuFile error code
        printf("FAIL (cuFileRead returned %zd: %s)\n",
               nread, cufileop_status_error((CUfileOpError)(-nread)));
    } else if ((size_t)nread != t.read_size) {
        printf("PARTIAL (requested %zu, got %zd)\n", t.read_size, nread);
    } else {
        // Verify data was actually transferred (not all 0xCC)
        uint8_t sample[16];
        cudaMemcpy(sample, (char*)dev_buf + t.buf_offset,
                   sizeof(sample), cudaMemcpyDeviceToHost);
        int all_cc = 1;
        for (int i = 0; i < 16; i++) {
            if (sample[i] != 0xCC) { all_cc = 0; break; }
        }
        if (all_cc) {
            printf("WARN (read OK but data unchanged — might be zero device region)\n");
        } else {
            printf("OK (%zd bytes, first 4 bytes: %02x %02x %02x %02x)\n",
                   nread, sample[0], sample[1], sample[2], sample[3]);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_or_device>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    // Open with O_RDONLY | O_DIRECT (required for GDS)
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fprintf(stderr, "open(%s) failed: %s\n", path, strerror(errno));
        return 1;
    }

    // CUDA + cuFile init
    mb_cuda_init();
    CUdevice device = mb_cuda_get_device(0);
    CUcontext ctx = mb_cuda_new_context(device);
    mb_cuda_set_context(ctx);
    mb_cufile_driver_open();

    CUfileHandle_t fh = mb_cufile_handle_register(fd);

    // Allocate GPU buffer (1 MiB, well above any test size)
    const size_t buf_size = 1 * 1024 * 1024;
    void* dev_buf = mb_cuda_alloc(buf_size);
    mb_cufile_buf_register(dev_buf, buf_size);

    printf("GDS Alignment Test\n");
    printf("  file: %s\n", path);
    printf("  GPU buffer: %p (%zu bytes)\n\n", dev_buf, buf_size);

    // ──────────────────────────────────────────────
    // Test 1: Baseline — 4K aligned (should always work)
    // ──────────────────────────────────────────────
    printf("=== Test 1: 4K-aligned baseline ===\n");
    AlignTest tests_4k[] = {
        {"4K off, 4K size",       4096,    4096,   0},
        {"4K off, 8K size",       4096,    8192,   0},
        {"4K off, 64K size",      4096,   65536,   0},
        {"8K off, 4K size",       8192,    4096,   0},
        {"0 off, 4K size",           0,    4096,   0},
    };
    for (auto& t : tests_4k) run_test(fh, dev_buf, buf_size, t);
    printf("\n");

    // ──────────────────────────────────────────────
    // Test 2: 512B-aligned file offset (non-4K)
    // ──────────────────────────────────────────────
    printf("=== Test 2: 512B-aligned file offset ===\n");
    AlignTest tests_512_off[] = {
        {"512B off, 512B size",    512,     512,   0},
        {"512B off, 4K size",      512,    4096,   0},
        {"1024B off, 1024B size", 1024,    1024,   0},
        {"1536B off, 512B size",  1536,     512,   0},
        {"2048B off, 2048B size", 2048,    2048,   0},
        {"3584B off, 512B size",  3584,     512,   0},
    };
    for (auto& t : tests_512_off) run_test(fh, dev_buf, buf_size, t);
    printf("\n");

    // ──────────────────────────────────────────────
    // Test 3: 512B-aligned read size (non-4K)
    // ──────────────────────────────────────────────
    printf("=== Test 3: 512B-aligned read size (from 4K offset) ===\n");
    AlignTest tests_512_sz[] = {
        {"4K off, 512B size",     4096,     512,   0},
        {"4K off, 1024B size",    4096,    1024,   0},
        {"4K off, 1536B size",    4096,    1536,   0},
        {"4K off, 2560B size",    4096,    2560,   0},
        {"4K off, 3584B size",    4096,    3584,   0},
        {"4K off, 5120B size",    4096,    5120,   0},
        {"4K off, 27648B size",   4096,   27648,   0},
    };
    for (auto& t : tests_512_sz) run_test(fh, dev_buf, buf_size, t);
    printf("\n");

    // ──────────────────────────────────────────────
    // Test 4: Both offset and size 512B-aligned
    // ──────────────────────────────────────────────
    printf("=== Test 4: Both file_offset and size 512B-aligned ===\n");
    AlignTest tests_both[] = {
        {"512B off, 1024B size",   512,    1024,   0},
        {"1024B off, 3072B size", 1024,    3072,   0},
        {"2048B off, 27648B size",2048,   27648,   0},
        {"3584B off, 28160B size",3584,   28160,   0},
    };
    for (auto& t : tests_both) run_test(fh, dev_buf, buf_size, t);
    printf("\n");

    // ──────────────────────────────────────────────
    // Test 5: 512B-aligned buf_offset
    // ──────────────────────────────────────────────
    printf("=== Test 5: 512B-aligned buf_offset ===\n");
    AlignTest tests_buf[] = {
        {"4K foff, 4K size, 512 boff",  4096,  4096,  512},
        {"4K foff, 4K size, 1024 boff", 4096,  4096, 1024},
        {"4K foff, 4K size, 4K boff",   4096,  4096, 4096},
    };
    for (auto& t : tests_buf) run_test(fh, dev_buf, buf_size, t);
    printf("\n");

    // ──────────────────────────────────────────────
    // Test 6: Stress — realistic compressed page I/O patterns
    // ──────────────────────────────────────────────
    printf("=== Test 6: Realistic compressed page patterns ===\n");
    AlignTest tests_real[] = {
        // Typical compressed page: ~27 KiB at 4K-aligned offset
        {"27648B @ 0 (4K-aligned)",           0, 27648,  0},
        // Same page at 512B-aligned offset (if layout were 512B-aligned)
        {"27648B @ 28160 (512B-aligned)",  28160, 27648,  0},
        // Two pages coalesced without 4K padding gap
        {"55296B @ 0 (2 pages, no gap)",      0, 55296,  0},
    };
    for (auto& t : tests_real) run_test(fh, dev_buf, buf_size, t);
    printf("\n");

    // Cleanup
    mb_cufile_buf_deregister(dev_buf);
    mb_cuda_free(dev_buf);
    mb_cufile_handle_deregister(fh);
    mb_cufile_driver_close();
    close(fd);

    printf("Done. Check cufile.log for detailed GDS diagnostics.\n");
    return 0;
}
