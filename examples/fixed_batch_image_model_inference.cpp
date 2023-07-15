#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

int argmax(const std::vector<float> &v) {
  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

std::vector<float> postprocess(const std::vector<float> &tensor_values) {
  // Compute softmax
  std::vector<float> softmax_values(tensor_values.size());
  float max_value =
      *std::max_element(tensor_values.begin(), tensor_values.end());
  float sum = 0.0f;

  for (std::size_t i = 0; i < tensor_values.size(); ++i) {
    softmax_values[i] = std::exp(tensor_values[i] - max_value);
    sum += softmax_values[i];
  }

  for (std::size_t i = 0; i < softmax_values.size(); ++i) {
    softmax_values[i] /= sum;
  }

  return softmax_values;
}

// // pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v) {
  int total = 1;
  for (auto &i : v)
    total *= i;
  return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data,
                         const std::vector<std::int64_t> &shape) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(),
                                            shape.data(), shape.size());
  return tensor;
}

class OnnxModel {
private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session{nullptr};
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::int64_t> input_shapes;
  std::vector<std::int64_t> output_shapes;

  std::vector<const char *> input_names_char;
  std::vector<const char *> output_names_char;
  int gpu_id;

public:
  // Initialize OnnxModel
  OnnxModel(const std::string &model_file, bool use_cuda = false,
            int gpu_id = 0)
      : env(ORT_LOGGING_LEVEL_WARNING, "single_image_model_inference"),
        gpu_id(gpu_id) {

    if (use_cuda) {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
          session_options, gpu_id));
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = Ort::Session(env, model_file.c_str(), session_options);
    setup();
  }

  // set up the model
  void setup() {
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
      input_names.emplace_back(
          session.GetInputNameAllocated(i, allocator).get());
      input_shapes =
          session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    for (auto &s : input_shapes) {
      if (s < 0) {
        s = 1;
      }
    }

    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
      output_names.emplace_back(
          session.GetOutputNameAllocated(i, allocator).get());
      output_shapes =
          session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    std::cout << "Input names: " << input_names[0] << std::endl;
    std::cout << "Input shape: " << print_shape(input_shapes) << std::endl;
    std::cout << "Total number of elements: " << calculate_product(input_shapes)
              << std::endl;

    std::cout << "Output names: " << output_names[0] << std::endl;
    std::cout << "Output shape: " << print_shape(output_shapes) << std::endl;
    std::cout << "Total number of elements: "
              << calculate_product(output_shapes) << std::endl;

    // pass data through model
    input_names_char.resize(input_names.size());
    std::transform(std::begin(input_names), std::end(input_names),
                   std::begin(input_names_char),
                   [&](const std::string &str) { return str.c_str(); });

    output_names_char.resize(output_names.size());
    std::transform(std::begin(output_names), std::end(output_names),
                   std::begin(output_names_char),
                   [&](const std::string &str) { return str.c_str(); });
  }

  std::vector<float> preprocess(const std::vector<cv::Mat> &images) {
    auto input_shape = input_shapes;

    if (input_shape.size() != 4) {
      throw std::runtime_error("Invalid input shape. Expected format: [batch, "
                               "channel, height, width]");
    }

    if (input_shape[0] != images.size()) {
      throw std::runtime_error(
          "Invalid batch size. Expected: " + std::to_string(input_shape[0]) +
          ", Got: " + std::to_string(images.size()));
    }

    auto total_number_elements = calculate_product(input_shape);
    std::vector<float> batch_input_tensor_values(total_number_elements);

    int channel = static_cast<int>(input_shape[1]);
    if (channel != 1) {
      throw std::runtime_error(
          "Invalid number of channels. Expected: 1, Got: " +
          std::to_string(channel));
    }

    int height = static_cast<int>(input_shape[2]);
    int width = static_cast<int>(input_shape[3]);

    for (std::size_t i = 0; i < images.size(); i++) {
      cv::Mat resized_image;
      cv::resize(images[i], resized_image, cv::Size(width, height));
      cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2GRAY);

      cv::Mat image_float32;
      resized_image.convertTo(image_float32, CV_32FC1);
      image_float32 /= 255.0f;

      // Mat to vector<float>
      cv::dnn::blobFromImage(image_float32, image_float32);

      std::vector<float> image_float32_vec;
      image_float32_vec.assign(image_float32.begin<float>(),
                               image_float32.end<float>());

      // Copy the vector to the batch_input_tensor_values
      std::copy(image_float32_vec.begin(), image_float32_vec.end(),
                batch_input_tensor_values.begin() +
                    i * image_float32_vec.size());
    }

    return batch_input_tensor_values;
  }

  std::vector<std::vector<float>>
  operator()(const std::vector<cv::Mat> &images) {
    // Preprocess the input image
    try {
      std::vector<float> input_tensor_values = preprocess(images);

      Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
          OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

      std::vector<Ort::Value> input_tensors;
      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          mem_info, input_tensor_values.data(), input_tensor_values.size(),
          input_shapes.data(), input_shapes.size()));

      auto output_tensors =
          session.Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                      input_tensors.data(), input_names_char.size(),
                      output_names_char.data(), output_names_char.size());

      // double-check the dimensions of the output tensors
      // NOTE: the number of output tensors is equal to the number of output
      // nodes specifed in the Run() call
      assert(output_tensors.size() == output_names.size() &&
             output_tensors[0].IsTensor());

      // Convert the output tensor to a vector
      auto *output_tensor =
          output_tensors.front().GetTensorMutableData<float>();

      std::vector<float> flatten_batch_output_tensor_values(
          output_tensor,
          output_tensor + calculate_product(session.GetOutputTypeInfo(0)
                                                .GetTensorTypeAndShapeInfo()
                                                .GetShape()));

      auto num_classes = output_shapes[1];
      std::vector<std::vector<float>> batch_output_tensor_values;
      for (std::size_t i = 0; i < images.size(); i++) {
        std::vector<float> output_tensor_values(
            flatten_batch_output_tensor_values.begin() + i * num_classes,
            flatten_batch_output_tensor_values.begin() + (i + 1) * num_classes);
        batch_output_tensor_values.emplace_back(
            postprocess(output_tensor_values));
      }

      return batch_output_tensor_values;
    } catch (const Ort::Exception &exception) {
      std::cout << "ERROR running model inference: " << exception.what()
                << std::endl;
      exit(-1);
    }
  }
};

int main(int argc, char *argv[]) {
  std::string model_file = "/app/models/mnist_b4.onnx";
  const std::vector<std::string> filenames = {"0.png", "7.png", "8.png",
                                              "9.png"};

  // Add expected output vector.
  const std::vector<int> expected_output_vector = {0, 7, 8, 9};

  bool test_passed = true;
  std::chrono::duration<double> diff_cpu, diff_gpu;

  for (bool use_cuda : {false, true}) {
    OnnxModel model(model_file, use_cuda);
    auto start = std::chrono::high_resolution_clock::now(); // start timing

    std::vector<cv::Mat> images;
    for (auto filename : filenames) {
      images.push_back(cv::imread("/app/imgs/" + filename));
    }

    for (int i = 0; i < 100; i++) {
      auto batch_output = model(images); // softmax values
      for (int j = 0; j < batch_output.size(); j++) {
        auto output = batch_output[j];
        std::cout << "Output softmax values:\n";
        for (float val : output) {
          std::cout << val << " ";
        }
        int max_index = argmax(output);

        std::cout << "\nArgmax index: " << max_index << std::endl;
        std::cout << "Max probability: " << output[max_index] << std::endl;

        // Compare the expected and the actual output.
        if (max_index != expected_output_vector[j]) {
          test_passed = false;
          std::cerr << "Test failed! For file " << filenames[j]
                    << ", expected: " << expected_output_vector[j]
                    << ", but got: " << max_index << std::endl;
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

if (test_passed) {
  std::cout << "\033[32mAll tests passed successfully.\033[0m" << std::endl;
} else {
  std::cout << "\033[31mSome tests failed.\033[0m" << std::endl;
  return -1;
}

return 0;
}