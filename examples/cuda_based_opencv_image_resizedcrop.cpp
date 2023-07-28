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
  const std::vector<std::vector<std::vector<float>>> bounding_boxes_per_image = {
    {{0.0, 0.0, 0.5, 0.5}, {0.5, 0.0, 1.0, 0.5}, {0.0, 0.5, 0.5, 1.0}, {0.5, 0.5, 1.0, 1.0}, {0.25, 0.25, 0.75, 0.75}}, 
    {{0.0, 0.0, 0.5, 0.5}, {0.5, 0.0, 1.0, 0.5}, {0.0, 0.5, 0.5, 1.0}, {0.5, 0.5, 1.0, 1.0}, {0.25, 0.25, 0.75, 0.75}}
    };

  int resize_height = 384;
  int resize_width = 128;

  std::chrono::duration<double> diff_cpu, diff_gpu;
  std::vector<cv::cuda::GpuMat> images_on_gpu; // to avoid memory allocation in the
  cv::cuda::GpuMat resized_cropped_image_on_gpu;

  int gpu_id = 0;

  std::vector<cv::Mat> images;
  for (auto filename : filenames) {
    images.push_back(cv::imread("/app/imgs/" + filename));
    cv::cuda::GpuMat image_on_gpu;
    image_on_gpu.upload(images.back());
    images_on_gpu.push_back(image_on_gpu);
  }

  for (bool use_cuda : {false, true}) {
    auto start = std::chrono::high_resolution_clock::now(); // start timing
    for (int i = 0; i < 1000; i++) {
      for (int j = 0; j < images.size(); j++)
      {
        cv::Mat image = images[j];

        int input_height = image.rows;
        int input_width = image.cols;

        if(use_cuda)
        {
          std::vector<cv::cuda::GpuMat> cropped_images;
          for(int k = 0; k < bounding_boxes_per_image[j].size(); k++)
          {
            auto box = bounding_boxes_per_image[j][k];
            int x1 = box[0] * input_width;
            int y1 = box[1] * input_height;
            int x2 = box[2] * input_width;
            int y2 = box[3] * input_height;

            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(input_width, x2);
            y2 = std::min(input_height, y2);

            int cropped_width = x2 - x1;
            int cropped_height = y2 - y1;
            
            if(cropped_width < 0 || cropped_height < 0 || cropped_width >= input_width || cropped_height >= input_height)
            {
              std::cout << "Invalid bounding box" << std::endl;
              continue;
            }
            
            cv::cuda::GpuMat cropped_image = images_on_gpu[j](cv::Rect(x1, y1, cropped_width, cropped_height));
            cv::cuda::resize(cropped_image, resized_cropped_image_on_gpu, cv::Size(resize_width, resize_height));
            cropped_images.push_back(std::move(resized_cropped_image_on_gpu));

            //save images to disk
            // cv::Mat resized_cropped_image_cpu;
            // resized_cropped_image_on_gpu.download(resized_cropped_image_cpu);
            // std::stringstream ss;
            // ss << "/app/imgs/gpu_resized_cropped_image_" << j << "_" << k << ".jpg";
            // cv::imwrite(ss.str(), resized_cropped_image_cpu);
          }
        }
        else
        {
          std::vector<cv::Mat> cropped_images;
          for(auto box:bounding_boxes_per_image[j])
          {
            int x1 = box[0] * input_width;
            int y1 = box[1] * input_height;
            int x2 = box[2] * input_width;
            int y2 = box[3] * input_height;

            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(input_width-1, x2);
            y2 = std::min(input_height-1, y2);

            int cropped_width = x2 - x1;
            int cropped_height = y2 - y1;
            
            if(cropped_width < 0 || cropped_height < 0 || cropped_width >= input_width || cropped_height >= input_height)
            {
              std::cout << "Invalid bounding box" << std::endl;
              continue;
            }

            cv::Mat cropped_image = image(cv::Rect(x1, y1, cropped_width, cropped_height));
            cv::resize(cropped_image, cropped_image, cv::Size(resize_width, resize_height));
            cropped_images.push_back(cropped_image);
            
            //save images to disk
            // std::stringstream ss;
            // ss << "/app/imgs/cpu_resized_cropped_image_" << j << "_" << cropped_images.size() << ".jpg";
            // cv::imwrite(ss.str(), cropped_image);
          }
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