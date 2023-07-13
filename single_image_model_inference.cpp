#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <cassert>
#include <array>
#include <vector>
#include <numeric>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;

class Model {
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::Session session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<std::int64_t>> input_shapes_;
    std::vector<std::vector<std::int64_t>> output_shapes_;
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

        // Assuming model has 1 input node and 1 output node.
        assert(input_names_.size() == 1 && output_names_.size() == 1);
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
};


int main()
{
    std::cout << "single_image_model_inference.cpp" << std::endl;

    std::string model_path = "/app/mnist-12.onnx";
    Model model(model_path);

    std::cout << "Input Node Name/Shape (" << model.getInputNames().size() << "):" << std::endl;
    for (std::size_t i = 0; i < model.getInputNames().size(); i++) {
        std::cout << "\t" << model.getInputNames().at(i) << " : ";
        model.printShape(model.getInputShapes().at(i));
    }

    std::cout << "Output Node Name/Shape (" << model.getOutputNames().size() << "):" << std::endl;
    for (std::size_t i = 0; i < model.getOutputNames().size(); i++) {
        std::cout << "\t" << model.getOutputNames().at(i) << " : ";
        model.printShape(model.getOutputShapes().at(i));
    }

    return 0;
}
