find_package(CUDA REQUIRED)

option(CUDA_BUILD_CC20 "Build with compute capability 2.0 support" OFF)
option(CUDA_BUILD_CC21 "Build with compute capability 2.1 support" OFF)
option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" OFF)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" ON)
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" OFF)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" ON)
option(CUDA_BUILD_CC60 "Build with compute capability 6.0 support" OFF)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" OFF)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" ON)
option(CUDA_BUILD_DEBUG "Build with kernel debug" OFF)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-use_fast_math;")

if (CUDA_BUILD_CC20)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_20;")
endif ()
if (CUDA_BUILD_CC21)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_20,code=sm_21;")
endif ()
if (CUDA_BUILD_CC30)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30;")
endif ()
if(CUDA_BUILD_CC35)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35;")
endif ()
if (CUDA_BUILD_CC50)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_50,code=sm_50;")
endif ()
if (CUDA_BUILD_CC52)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_52,code=sm_52;")
endif ()
if (CUDA_BUILD_CC60)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_60,code=sm_60;")
endif ()
if (CUDA_BUILD_CC61)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_61,code=sm_61;")
endif ()

if (CUDA_BUILD_INFO)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v;-lineinfo")
endif ()

if (CUDA_BUILD_DEBUG)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G")
endif ()

set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
