cmake_minimum_required(VERSION 3.22)

project(onnxruntime-cuda-cpp-example)

set(CMAKE_CXX_STANDARD 20)

message("ENV ONNXRUNTIME_VERSION = $ENV{ONNXRUNTIME_VERSION}")
message("ENV OPENCV_VERSION = $ENV{OPENCV_VERSION}")

set(ONNXRUNTIME_PATH /onnxruntime-linux-x64-gpu-$ENV{ONNXRUNTIME_VERSION})

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_PATH}/lib)
find_library(ONNXRUNTIME_CUDA_LIB onnxruntime_providers_cuda HINTS ${ONNXRUNTIME_PATH}/lib)
find_library(ONNXRUNTIME_SHARED_LIB onnxruntime_providers_shared HINTS ${ONNXRUNTIME_PATH}/lib)
find_library(ONNXRUNTIME_TENSORRT_LIB onnxruntime_providers_tensorrt HINTS ${ONNXRUNTIME_PATH}/lib)

set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_LIB} ${ONNXRUNTIME_CUDA_LIB} ${ONNXRUNTIME_SHARED_LIB} ${ONNXRUNTIME_TENSORRT_LIB})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

message("Detected OpenCV_VERSION by find_package = ${OpenCV_VERSION}")

add_subdirectory(examples)
