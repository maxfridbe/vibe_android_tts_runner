cmake_minimum_required(VERSION 3.22)
project(ttsrunner_jni CXX C)

# llama.cpp build knobs: CPU always; OpenCL is the GPU backend.
# On-device findings (SD 8 Gen 2 / Adreno 740):
#  - Vulkan: the proprietary Adreno Vulkan driver fails to compile llama.cpp's
#    compute shaders at all (vk::Device::createComputePipeline: ErrorUnknown
#    during the first talker decode), so it is compiled out.
#  - OpenCL: Qualcomm's Adreno kernels handle the talker LLM fine but SIGABRT
#    inside ggml_cl_compute_forward on the codec ("mmproj") graph — so the JNI
#    layer keeps the codec on CPU (mmproj_use_gpu=false) and only offloads the
#    talker layers.
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL ON CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)    # the TTS codec helper lives in mtmd
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(ANDROID)
    # posix_madvise is missing from the Android libc llama.cpp expects
    set(SHIM "${CMAKE_CURRENT_SOURCE_DIR}/android_posix_shim.h")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -include${SHIM}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include${SHIM}")
    # Android 15+ 16 KB pages vs 4 KB-aligned GGUF tensor offsets break mmap
    set(GGML_MMAP OFF CACHE BOOL "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        # +i8mm enables ggml's int8 matmul kernels (big quant-gemm speedup);
        # needs a ~2021+ SoC (SD 888/8 Gen 1 and later, Tensor, Dimensity
        # 1000+). Older phones SIGILL — drop i8mm here to support them.
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -march=armv8.2-a+dotprod+fp16+i8mm")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16+i8mm")
    endif()
endif()

# ggml-vulkan needs the SPIRV-Headers CMake package; the builder image
# installs it to /usr/local, which the Android toolchain's find-root
# filtering would otherwise hide.
if(EXISTS "/usr/local/share/cmake/SPIRV-Headers")
    list(APPEND CMAKE_FIND_ROOT_PATH "/usr/local")
    set(SPIRV-Headers_DIR "/usr/local/share/cmake/SPIRV-Headers" CACHE PATH "" FORCE)
endif()

# OpenCL headers + dlopen stub: links against the stub at build time, loads
# the vendor libOpenCL.so at runtime (absent on non-Qualcomm devices, where
# ggml then simply reports no GPU and everything runs on CPU).
if(GGML_OPENCL)
    set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
    set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
    set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencl-headers" CACHE PATH "" FORCE)
    set(OpenCL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/opencl-stub/libOpenCL.so" CACHE FILEPATH "" FORCE)
endif()

# llama.cpp sources: pre-fetched into the builder image at /opt/llama_cpp
# (env LLAMA_CPP_SRC), with the Qwen3-TTS CPU/Vulkan fixes already patched in.
# FetchContent fallback for building outside the container.
set(LLAMA_SRC "$ENV{LLAMA_CPP_SRC}")
if(NOT EXISTS "${LLAMA_SRC}/CMakeLists.txt")
    include(FetchContent)
    FetchContent_Declare(llama_cpp
        URL https://github.com/ggml-org/llama.cpp/archive/f9e832c10e9444cb168ddcb579cc62c154f3068b.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(llama_cpp)
    set(LLAMA_SRC ${llama_cpp_SOURCE_DIR})
    message(WARNING "using FetchContent llama.cpp WITHOUT the qwen3tts-fixes patch; "
                    "CPU/Vulkan TTS will crash. Build in the container instead.")
endif()
add_subdirectory(${LLAMA_SRC} llama_build EXCLUDE_FROM_ALL)

add_library(ttsrunner_jni SHARED tts_jni.cpp)
target_include_directories(ttsrunner_jni PRIVATE
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/common
    ${LLAMA_SRC}/tools/mtmd)
if(GGML_OPENCL)
    target_link_libraries(ttsrunner_jni PRIVATE ${OpenCL_LIBRARY})
endif()
target_link_libraries(ttsrunner_jni PRIVATE llama llama-common mtmd android log)
target_compile_options(ttsrunner_jni PRIVATE -fexceptions -frtti)
target_compile_features(ttsrunner_jni PRIVATE cxx_std_17)
