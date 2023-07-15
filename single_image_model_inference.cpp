#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <cassert>
#include <array>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm> // For std::max_element
#include <opencv2/opencv.hpp>

using namespace std;

int argmax(const std::vector<float>& v) {
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

class Model {
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::Session session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_char_;
    std::vector<const char*> output_names_char_;

    std::vector<std::vector<std::int64_t>> input_shapes_;
    std::vector<std::vector<std::int64_t>> output_shapes_;

    Ort::Value preprocess(const cv::Mat& image) {
        const auto& input_shape = input_shapes_[0];
        if (input_shape.size() != 4) {
            throw std::runtime_error("Invalid input shape. Expected format: [batch, channel, height, width]");
        }
        
        int height = static_cast<int>(input_shape[2]);
        int width = static_cast<int>(input_shape[3]);
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(width, height));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2GRAY);

        cv::Mat image_float32;
        resized_image.convertTo(image_float32, CV_32FC1);
        image_float32 /= 255.0f;
        std::vector<float> input_tensor_values((float*)image_float32.data, (float*)image_float32.data + width * height);

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), input_tensor_values.size(), input_shapes_[0].data(), input_shapes_[0].size());
        return tensor;
    }

   std::vector<float> postprocess(const std::vector<float>& tensor_values) {
        // Compute softmax
        std::vector<float> softmax_values(tensor_values.size());
        float max_value = *std::max_element(tensor_values.begin(), tensor_values.end());
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


public:
    Model(const std::string& model_path) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "model-runner"), 
      session_(env_, model_path.c_str(), session_options_) {

        for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
            input_names_.emplace_back(session_.GetInputNameAllocated(i, allocator_).get());
            input_shapes_.emplace_back(session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
            output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator_).get());
            output_shapes_.emplace_back(session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        input_names_char_.resize(input_names_.size());
        std::transform(std::begin(input_names_), std::end(input_names_), std::begin(input_names_char_),
                 [&](const std::string& str) { return str.c_str(); });
        
        output_names_char_.resize(output_names_.size());
        std::transform(std::begin(output_names_), std::end(output_names_), std::begin(output_names_char_),
                 [&](const std::string& str) { return str.c_str(); });

        // Assuming model has 1 input node and 1 output node.
        assert(input_names_.size() == 1 && output_names_.size() == 1);
    

        std::cout << "Input Node Name/Shape (" << input_names_.size() << "):" << std::endl;
        for (std::size_t i = 0; i < input_names_.size(); i++) {
            std::cout << "\t" << input_names_.at(i) << " : ";
            printShape(input_shapes_.at(i));
        }

        std::cout << "Output Node Name/Shape (" << output_names_.size() << "):" << std::endl;
        for (std::size_t i = 0; i < output_names_.size(); i++) {
            std::cout << "\t" << output_names_.at(i) << " : ";
            printShape(output_shapes_.at(i));
        }
    }

    // Getter functions for input/output names and shapes
    const std::vector<std::string>& getInputNames() const { return input_names_; }
    const std::vector<std::string>& getOutputNames() const { return output_names_; }
    const std::vector<std::vector<std::int64_t>>& getInputShapes() const { return input_shapes_; }
    const std::vector<std::vector<std::int64_t>>& getOutputShapes() const { return output_shapes_; }
    
    void printShape(const std::vector<std::int64_t>& shape) const {
        std::cout << "[";
        for (auto it = shape.begin(); it != shape.end(); ++it) {
            std::cout << *it;
            if (it + 1 != shape.end()) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // Further functions for handling the input and output tensors,
    // running the model, etc. could be added here.
    std::vector<float> operator()(const cv::Mat& image) {
        // Preprocess the input image
        try {
            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(preprocess(image));

            auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names_char_.data(), input_tensors.data(), 
                                            input_names_char_.size(), output_names_char_.data(), output_names_char_.size());

            // double-check the dimensions of the output tensors
            // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
            assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

                    // Convert the output tensor to a vector
            auto* output_tensor = output_tensors.front().GetTensorMutableData<float>();
            std::vector<float> output_tensor_values(output_tensor, output_tensor + std::accumulate(output_shapes_[0].begin(), output_shapes_[0].end(), 1, std::multiplies<std::int64_t>()));

            // Postprocess the output tensor
            return postprocess(output_tensor_values);
        } catch (const Ort::Exception& exception) {
            std::cout << "ERROR running model inference: " << exception.what() << std::endl;
            exit(-1);
        } 
    }
};


int main()
{
    std::cout << "single_image_model_inference.cpp" << std::endl;
    std::cout << std::fixed << std::setprecision(8);

    std::string model_path = "/app/mnist-12.onnx";
    Model model(model_path);

    cv::Mat input = cv::imread("/app/7.png");
    auto output = model(input); // softmax values

    // Find the index of the maximum softmax value
    int max_index = argmax(output);

    std::cout << "Output softmax values:\n";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << "\nArgmax index: " << max_index << std::endl;
    std::cout << "Max probability: " << output[max_index] << std::endl;

    return 0;
}
