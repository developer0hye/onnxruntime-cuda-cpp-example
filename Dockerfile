# --------------------------------------------------------------
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with CUDA and CUDNN

# Find the base image from https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL author="Yonghye Kwon" 
LABEL email="developer.0hye@gmail.com"

ENV ONNXRUNTIME_VERSION=1.15.1 \
    OPENCV_VERSION=4.7.0

# Install basic packages for build and development
RUN \
    apt-get update && \
    apt-get install vim -y && \
    apt-get install cmake -y && \
    apt-get install wget -y && \
    apt-get install unzip -y

# Install ONNXRuntime
RUN \
    wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xvf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz && \
    rm onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz

# Install OpenCV
RUN \
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    cd opencv-${OPENCV_VERSION} && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    mkdir build && \
    cd build && \
        cmake -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_INSTALL_PREFIX=/usr/local \
            -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
            -D WITH_CUDA=ON \
            -D WITH_CUDNN=ON \
            -D OPENCV_DNN_CUDA=ON \
            -D CUDA_FAST_MATH=ON \
            -D WITH_CUBLAS=ON \
            -D WITH_CUFFT=ON .. && \
    make -j$(nproc) && \
    make install

RUN mkdir /app/
COPY . /app/

RUN \
    cd app && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --config Release

WORKDIR /app/build/examples