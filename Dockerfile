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
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j4 && \
    make install

RUN mkdir /app/
COPY . /app/
