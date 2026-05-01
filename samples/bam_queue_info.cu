// bam_queue_info.cu — Check BaM queue counts for different num_queues values.
//
// Usage:
//   sudo ./build/bam_queue_info /dev/libnvm0 [num_queues]
//
// Default num_queues: 128, 256, 512, 1024 (tests all)
// If num_queues is given, tests only that value.

#include <cstdio>
#include <cstdlib>
#include <cstdint>

// BAM headers
#include "ctrl.h"

static void print_device_caps(const char* path) {
    printf("\n=== Device Capabilities ===\n");
    printf("Device: %s\n", path);

    try {
        Controller ctrl(path, /*ns_id=*/1, /*cudaDevice=*/0, /*queueDepth=*/64, /*numQueues=*/1);

        printf("\n  --- Controller Registers (CAP) ---\n");
        printf("  max_qs (MQES from CAP)  = %u\n", ctrl.ctrl->max_qs);

        printf("\n  --- IDENTIFY CONTROLLER ---\n");
        printf("  max_entries (MQES)      = %u\n", ctrl.info.max_entries);
        printf("  max_data_size (MDTS)    = %zu bytes\n", ctrl.info.max_data_size);
        printf("  max_data_pages          = %zu\n", ctrl.info.max_data_pages);
        printf("  max_out_cmds (MAXCMD)   = %zu\n", ctrl.info.max_out_cmds);
        printf("  sq_entry_size (SQES)    = %zu bytes\n", ctrl.info.sq_entry_size);
        printf("  cq_entry_size (CQES)    = %zu bytes\n", ctrl.info.cq_entry_size);
        printf("  nvme_version            = 0x%08x\n", ctrl.info.nvme_version);
        printf("  page_size               = %zu\n", ctrl.info.page_size);
        printf("  db_stride               = %zu\n", ctrl.info.db_stride);
        printf("  contiguous (CQR)        = %d\n", ctrl.info.contiguous);
        printf("  serial_no               = %.20s\n", ctrl.info.serial_no);
        printf("  model_no                = %.40s\n", ctrl.info.model_no);
        printf("  firmware                = %.8s\n", ctrl.info.firmware);

        printf("\n  --- Namespace ---\n");
        printf("  blk_size (LBA)          = %u\n", ctrl.blk_size);
        printf("  page_size               = %u\n", ctrl.page_size);

        printf("\n  --- Queue Negotiation ---\n");
        printf("  n_sqs (granted by dev)  = %u\n", ctrl.n_sqs);
        printf("  n_cqs (granted by dev)  = %u\n", ctrl.n_cqs);
    } catch (const std::exception& e) {
        fprintf(stderr, "  ERROR: %s\n", e.what());
    }
}

static void test_queues(const char* path, uint32_t num_queues, uint32_t queue_depth) {
    printf("\n=== Testing num_queues=%u, queue_depth=%u ===\n", num_queues, queue_depth);
    printf("Device: %s\n", path);

    try {
        Controller ctrl(path, /*ns_id=*/1, /*cudaDevice=*/0, queue_depth, num_queues);
        printf("  n_sqs  = %u (SQs granted by device)\n", ctrl.n_sqs);
        printf("  n_cqs  = %u (CQs granted by device)\n", ctrl.n_cqs);
        printf("  n_qps  = %u (actual queue pairs created)\n", ctrl.n_qps);
        printf("  page_size = %u\n", ctrl.page_size);
        printf("  blk_size  = %u\n", ctrl.blk_size);
    } catch (const std::exception& e) {
        fprintf(stderr, "  ERROR: %s\n", e.what());
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <device> [num_queues]\n", argv[0]);
        fprintf(stderr, "  device: e.g. /dev/libnvm0\n");
        fprintf(stderr, "  num_queues: if omitted, tests 128/256/512/1024\n");
        return 1;
    }

    const char* path = argv[1];
    const uint32_t queue_depth = 1024;

    cudaSetDevice(0);

    // Always print device capabilities first
    print_device_caps(path);

    if (argc >= 3) {
        uint32_t nq = (uint32_t)atoi(argv[2]);
        test_queues(path, nq, queue_depth);
    } else {
        uint32_t test_values[] = {128, 256, 512, 1024};
        for (uint32_t nq : test_values) {
            test_queues(path, nq, queue_depth);
        }
    }

    return 0;
}
