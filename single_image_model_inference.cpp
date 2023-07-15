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
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;
  std::vector<std::string> output_names;

  std::vector<const char *> input_names_char;
  std::vector<const char *> output_names_char;

public:
  // Initialize OnnxModel
  OnnxModel(const std::string &model_file)
      : env(ORT_LOGGING_LEVEL_WARNING, "single_image_model_inference"),
        session(env, model_file.c_str(), session_options) {
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
    }

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

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

  std::vector<float> preprocess(const cv::Mat &image) {
    auto input_shape = input_shapes;
    auto total_number_elements = calculate_product(input_shape);

    if (input_shape.size() != 4) {
      throw std::runtime_error("Invalid input shape. Expected format: [batch, "
                               "channel, height, width]");
    }

    int height = static_cast<int>(input_shape[2]);
    int width = static_cast<int>(input_shape[3]);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2GRAY);

    cv::Mat image_float32;
    resized_image.convertTo(image_float32, CV_32FC1);
    image_float32 /= 255.0f;

    // Mat to vector<float>
    std::vector<float> input_tensor_values(width * height, 0);
    input_tensor_values.assign((float *)image_float32.datastart,
                               (float *)image_float32.dataend);
    return input_tensor_values;
  }

  std::vector<float> operator()(const cv::Mat &image) {
    // Preprocess the input image
    try {
      auto input_shape = input_shapes;
      auto total_number_elements = calculate_product(input_shape);

      std::vector<float> input_tensor_values = preprocess(image);

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
      std::vector<float> output_tensor_values(
          output_tensor,
          output_tensor + calculate_product(session.GetOutputTypeInfo(0)
                                                .GetTensorTypeAndShapeInfo()
                                                .GetShape()));
      return postprocess(output_tensor_values);
    } catch (const Ort::Exception &exception) {
      std::cout << "ERROR running model inference: " << exception.what()
                << std::endl;
      exit(-1);
    }
  }
};

int main(int argc, ORTCHAR_T *argv[]) {
  std::string model_file = "/app/mnist-12.onnx";
  OnnxModel model(model_file);
  cv::Mat input = cv::imread("/app/7.png");
  for (int i = 0; i < 100; i++) {
    auto output = model(input); // softmax values

    std::cout << "Output softmax values:\n";
    for (float val : output) {
      std::cout << val << " ";
    }

    int max_index = argmax(output);

    std::cout << "\nArgmax index: " << max_index << std::endl;
    std::cout << "Max probability: " << output[max_index] << std::endl;
  }

  return 0;
}
