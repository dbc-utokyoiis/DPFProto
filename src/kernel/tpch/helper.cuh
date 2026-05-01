#pragma once

#include <assert.h>

#define ENABLE_GPU_RUNTIME_ASSERTION
#ifdef ENABLE_GPU_RUNTIME_ASSERTION
//#define DEBUG_ASSERT(expr) do { if (!(expr)) { printf( "Assertion failed: %s\n", #expr); abort(); } } while (0)
#define DEBUG_ASSERT(expr) do { assert(expr); } while (0)
#else
#define DEBUG_ASSERT(expr) do { } while (0)
#endif
