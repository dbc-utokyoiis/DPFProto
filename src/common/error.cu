#define TRY(expr)                                             \
    if (cudaError_t result = (expr); result != cudaSuccess) { \
        return result;                                        \
    }
