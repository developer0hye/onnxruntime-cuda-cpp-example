cmake_minimum_required(VERSION 3.22)

project(onnxruntime-cuda-cpp-example)

set(CMAKE_CXX_STANDARD 20)

message("ENV ONNXRUNTIME_VERSION = $ENV{ONNXRUNTIME_VERSION}")
message("ENV OPENCV_VERSION = $ENV{OPENCV_VERSION}")

set(ONNXRUNTIME_PATH /onnxruntime-linux-x64-gpu-$ENV{ONNXRUNTIME_VERSION})

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_PATH}/lib)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

message("Detected OpenCV_VERSION by find_package = ${OpenCV_VERSION}")

add_subdirectory(examples)
