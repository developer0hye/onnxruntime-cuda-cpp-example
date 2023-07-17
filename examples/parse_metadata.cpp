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


int main(int argc, char *argv[]) {
  std::string model_file = "/app/models/yolov8n.onnx";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "parse_metadata");
  Ort::SessionOptions session_options;
  Ort::Session session{nullptr};
  Ort::AllocatorWithDefaultOptions allocator;

  session = Ort::Session(env, model_file.c_str(), session_options);

  auto model_metadata = session.GetModelMetadata();
  auto custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator);
  std::cout << "Model Metadata: " << std::endl;
  for (auto &key : custom_metadata_map_keys) {
    std::string key_str = key.get();
    std::string value_str = model_metadata.LookupCustomMetadataMapAllocated(key_str.c_str(), allocator).get();
    std::cout << "key: " << key_str << " value: " << value_str << std::endl;
  }

  return 0;
}