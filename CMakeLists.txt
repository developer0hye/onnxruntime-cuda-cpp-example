cmake_minimum_required(VERSION 3.22)

project(onnxruntime-cuda-cpp-example)

set(CMAKE_CXX_STANDARD 17)

message("ENV ONNXRUNTIME_VERSION = $ENV{ONNXRUNTIME_VERSION}")
message("ENV OPENCV_VERSION = $ENV{OPENCV_VERSION}")

set(ONNXRUNTIME_PATH /onnxruntime-linux-x64-gpu-$ENV{ONNXRUNTIME_VERSION})

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_PATH}/lib)
find_package( OpenCV REQUIRED )

message("Detected OpenCV_VERSION by find_package = ${OpenCV_VERSION}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(TARGETS single_image_model_inference fixed_batch_image_model_inference dynamic_batch_image_model_inference)

foreach(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_include_directories(${TARGET} PUBLIC ${ONNXRUNTIME_PATH}/include)
    target_link_libraries(${TARGET} PUBLIC ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})
endforeach()