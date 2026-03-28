cmake_minimum_required(VERSION 3.18)
project(qualityscaler_tensorrt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 让用户在 cmake -DTENSORRT_ROOT= 时指定
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "C:/TensorRT-10.16.0.72")
endif()
if(NOT CUDA_ROOT)
    set(CUDA_ROOT "C:/CUDA")
endif()

find_path(TRT_INC NvInfer.h
          PATHS ${TENSORRT_ROOT}/include)
find_library(TRT_LIB nvinfer_10
             PATHS ${TENSORRT_ROOT}/lib)
find_package(CUDA REQUIRED)

add_library(qualityscaler_tensorrt SHARED
            ai_tensorrt_cpp.cpp)

target_include_directories(qualityscaler_tensorrt PRIVATE
                           ${TRT_INC}
                           ${CUDA_INCLUDE_DIRS})

target_link_libraries(qualityscaler_tensorrt PRIVATE
                      ${TRT_LIB}
                      ${CUDA_LIBRARIES})

# Windows 导出符号自动处理
if(MSVC)
    target_compile_definitions(qualityscaler_tensorrt PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# 安装到项目根目录，方便 Go 链接
install(TARGETS qualityscaler_tensorrt
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../..)