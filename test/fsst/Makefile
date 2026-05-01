NVCC        = nvcc
CXX         = g++
CUDA_ARCH   = -gencode arch=compute_80,code=sm_80

FSST_DIR    = ../fsst

CXXFLAGS    = -O3 -std=c++17
NVCCFLAGS   = -O3 -std=c++17 $(CUDA_ARCH) -Xcompiler=-O3

# Check AVX512 support
HAS_AVX512  := $(shell grep -q avx512 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
ifeq ($(HAS_AVX512),1)
  CXXFLAGS  += -mavx512f -mavx512bw -mavx512dq
  NVCC_XCOMPILER = -Xcompiler=-mavx512f,-mavx512bw,-mavx512dq
  FSST_OBJS  = libfsst.o fsst_avx512.o
else
  FSST_OBJS  = libfsst.o
endif

all: fsst_vchar_test

fsst_vchar_test: fsst_vchar_test.cu fsst_host.o cpu_fsst_avx512.o $(FSST_OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) -I$(FSST_DIR) -o $@ $^ -lpthread

fsst_host.o: fsst_host.cpp fsst_host.h fsst_page.h
	$(CXX) $(CXXFLAGS) -I$(FSST_DIR) -c -o $@ $<

cpu_fsst_avx512.o: cpu_fsst_avx512.cpp cpu_fsst_avx512.h
	$(CXX) $(CXXFLAGS) -I$(FSST_DIR) -c -o $@ $<

libfsst.o: $(FSST_DIR)/libfsst.cpp
	$(CXX) $(CXXFLAGS) -I$(FSST_DIR) -c -o $@ $<

fsst_avx512.o: $(FSST_DIR)/fsst_avx512.cpp
	$(CXX) $(CXXFLAGS) -I$(FSST_DIR) -c -o $@ $<

clean:
	rm -f fsst_vchar_test *.o

.PHONY: all clean
