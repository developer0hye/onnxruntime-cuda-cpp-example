#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    // std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  const std::vector<std::string> filenames = {"bus.jpg", "zidane.jpg"};
  int resize_height = 640;
  int resize_width = 640;

  std::chrono::duration<double> diff_cpu, diff_gpu;

  cv::Mat image_float32;
  cv::cuda::GpuMat image_on_gpu, resized_image_on_gpu,
      padded_resized_image_on_gpu; // to avoid memory allocation in the
                                           // loop
  cv::cuda::GpuMat chw_padded_resized_image_on_gpu(1, 3 * resize_height * resize_width, CV_8U);
  cv::cuda::GpuMat float_chw_padded_resized_image_on_gpu(1, 3 * resize_height * resize_width, CV_32F);
  
  std::vector<cv::cuda::GpuMat> rgb_channels = {
                    cv::cuda::GpuMat(resize_height, resize_width, CV_8U, &(chw_padded_resized_image_on_gpu.ptr()[0 + resize_height * resize_width])),
                    cv::cuda::GpuMat(resize_height, resize_width, CV_8U, &(chw_padded_resized_image_on_gpu.ptr()[resize_height * resize_width + resize_height * resize_width])),
                    cv::cuda::GpuMat(resize_height, resize_width, CV_8U, &(chw_padded_resized_image_on_gpu.ptr()[resize_height * resize_width * 2 + resize_height * resize_width * 3]))
  };

  int gpu_id = 0;

  std::vector<cv::Mat> images;
  for (auto filename : filenames) {
    images.push_back(cv::imread("/app/imgs/" + filename));
  }

  for (bool use_cuda : {false, true}) {
    auto start = std::chrono::high_resolution_clock::now(); // start timing
    for (int i = 0; i < 1000; i++) {

      for (auto image : images) {
        int input_height = image.rows;
        int input_width = image.cols;

        float r = std::min(static_cast<float>(resize_height) / input_height,
                           static_cast<float>(resize_width) / input_width);
        r = std::min(r, 1.0f);

        int non_padded_resize_height = static_cast<int>(input_height * r);
        int non_padded_resize_width = static_cast<int>(input_width * r);
        int top = 0, bottom = 0, left = 0, right = 0;
        bottom = resize_height - non_padded_resize_height;
        right = resize_width - non_padded_resize_width;

        if (use_cuda) {
          image_on_gpu.upload(image);
          cv::cuda::resize(
              image_on_gpu, resized_image_on_gpu,
              cv::Size(non_padded_resize_width, non_padded_resize_height),
              cv::INTER_LINEAR);
          cv::cuda::copyMakeBorder(
              resized_image_on_gpu, padded_resized_image_on_gpu, top, bottom,
              left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
          cv::cuda::cvtColor(padded_resized_image_on_gpu,
                             padded_resized_image_on_gpu, cv::COLOR_BGR2RGB);
          assert(padded_resized_image_on_gpu.rows == resize_height &&
                 padded_resized_image_on_gpu.cols == resize_width);
          
          // hwc to chw
          cv::cuda::split(padded_resized_image_on_gpu, rgb_channels);
          chw_padded_resized_image_on_gpu.convertTo(float_chw_padded_resized_image_on_gpu, CV_32F, 1.0/255.0);
          auto *dataPointer = float_chw_padded_resized_image_on_gpu.ptr<void>();
          
          Ort::MemoryInfo mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, gpu_id, OrtMemType::OrtMemTypeDefault);
          Ort::Value ort_tensor = Ort::Value::CreateTensor(mem_info, 
          dataPointer, 
          3 * resize_height * resize_width * sizeof(float), 
          std::vector<int64_t>{1, 3, resize_height, resize_width}.data(), 
          1,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        } else {
          cv::resize(
              image, image,
              cv::Size(non_padded_resize_width, non_padded_resize_height),
              cv::INTER_LINEAR);
          cv::copyMakeBorder(image, image, top, bottom, left, right,
                             cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
          assert(image.rows == resize_height && image.cols == resize_width);
          cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
          image.convertTo(image_float32, CV_32FC3);
          cv::dnn::blobFromImage(image_float32, image_float32, 1. / 255.);
          
          Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
          OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

          auto *dataPointer = image_float32.ptr<void>();

          Ort::Value ort_tensor = Ort::Value::CreateTensor(mem_info, 
          dataPointer, 
          3 * resize_height * resize_width * sizeof(float), 
          std::vector<int64_t>{1, 3, resize_height, resize_width}.data(), 
          1,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now(); // end timing

    if (use_cuda)
      diff_gpu = end - start;
    else
      diff_cpu = end - start;
  }

  std::cout << "Execution time using CPU: " << diff_cpu.count() << " s\n";
  std::cout << "Execution time using GPU: " << diff_gpu.count() << " s\n";

  return 0;
}