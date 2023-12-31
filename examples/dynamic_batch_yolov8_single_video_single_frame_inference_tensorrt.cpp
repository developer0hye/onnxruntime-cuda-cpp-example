#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

float iou(const std::vector<float> &boxA, const std::vector<float> &boxB) {
  // The format of box is [top_left_x, top_left_y, bottom_right_x,
  // bottom_right_y]
  const float eps = 1e-6;
  float iou = 0.f;
  float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  float x1 = std::max(boxA[0], boxB[0]);
  float y1 = std::max(boxA[1], boxB[1]);
  float x2 = std::min(boxA[2], boxB[2]);
  float y2 = std::min(boxA[3], boxB[3]);
  float w = std::max(0.f, x2 - x1);
  float h = std::max(0.f, y2 - y1);
  float inter = w * h;
  iou = inter / (areaA + areaB - inter + eps);
  return iou;
}

void nms(std::vector<std::vector<float>> &boxes, const float iou_threshold) {
  // The format of boxes is [[top_left_x, top_left_y, bottom_right_x,
  // bottom_right_y, score, class_id], ...] Sorting "score + class_id" is to
  // ensure that the boxes with the same class_id are grouped together and
  // sorted by score
  std::sort(boxes.begin(), boxes.end(),
            [](const std::vector<float> &boxA, const std::vector<float> &boxB) {
              return boxA[4] + boxA[5] > boxB[4] + boxB[5];
            });
  for (int i = 0; i < boxes.size(); ++i) {
    if (boxes[i][4] == 0.f) {
      continue;
    }
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (boxes[i][5] != boxes[j][5]) {
        break;
      }
      if (iou(boxes[i], boxes[j]) > iou_threshold) {
        boxes[j][4] = 0.f;
      }
    }
  }
  std::erase_if(boxes,
                [](const std::vector<float> &box) { return box[4] == 0.f; });
}

std::vector<std::vector<float>> postprocess(
    const std::vector<float> &tensor_values, //[num_outputs x num_anchors]
    const std::vector<std::int64_t> &input_shape,
    const std::vector<std::int64_t> &pad_hw,
    const std::vector<std::int64_t> &output_shape,
    const float score_threshold = 0.25f,
    const float nms_iou_threshold = 0.45f) {

  /*
  tensor_values:

  cx1 cx2 cx3 cx4 ... cxN
  cy1 cy2 cy3 cy4 ... cyN
  w1 w2 w3 w4 ... wN
  h1 h2 h3 h4 ... hN
  c11 c12 c13 c14 ... c1N
  c21 c22 c23 c24 ... c2N
  c31 c32 c33 c34 ... c3N
  ...
  cM1 cM2 cM3 cM4 ... cMN

  transposed_tensor_values:

  cx1 cy1 w1 h1 c11 c21 c31 ... cM1
  cx2 cy2 w2 h2 c12 c22 c32 ... cM2
  cx3 cy3 w3 h3 c13 c23 c33 ... cM3
  ...
  cxN cyN wN hN c1N c2N c3N ... cMN

  bboxes:

  top_left_x1 top_left_y1 bottom_right_x1 bottom_right_y1 max_score1
  max_score_index1 top_left_x2 top_left_y2 bottom_right_x2 bottom_right_y2
  max_score2 max_score_index2
  ...
  top_left_xN top_left_yN bottom_right_xN bottom_right_yN max_scoreN
  max_score_indexN
  */

  std::vector<float> transposed_tensor_values(
      tensor_values.size());          //[num_anchors x num_outputs]
  auto num_outputs = output_shape[1]; // 4 + num_classes, [x1, y1, w, h,
                                      // class1_score, class2_score, ...]
  auto num_anchors = output_shape[2];

  for (std::size_t i = 0; i < num_anchors; i++) {
    for (std::size_t j = 0; j < num_outputs; j++) {
      transposed_tensor_values[i * num_outputs + j] =
          tensor_values[j * num_anchors + i];
    }
  }

  const int boxes_num_outputs = 6;
  std::vector<std::vector<float>> boxes;
  for (int i = 0; i < num_anchors; i++) {
    float max_score = 0.0f;
    int max_score_index = 0;
    for (int j = 4; j < num_outputs; j++) {
      if (transposed_tensor_values[i * num_outputs + j] > max_score) {
        max_score = transposed_tensor_values[i * num_outputs + j];
        max_score_index = j;
      }
    }

    if (max_score < score_threshold) {
      continue;
    }

    float cx =
        transposed_tensor_values[i * num_outputs + 0] /
        (static_cast<float>(input_shape[3]) - static_cast<float>(pad_hw[1]));
    float cy =
        transposed_tensor_values[i * num_outputs + 1] /
        (static_cast<float>(input_shape[2]) - static_cast<float>(pad_hw[0]));
    float w =
        transposed_tensor_values[i * num_outputs + 2] /
        (static_cast<float>(input_shape[3]) - static_cast<float>(pad_hw[1]));
    float h =
        transposed_tensor_values[i * num_outputs + 3] /
        (static_cast<float>(input_shape[2]) - static_cast<float>(pad_hw[0]));

    float x1 = std::clamp(cx - w / 2., 0., 1.);
    float y1 = std::clamp(cy - h / 2., 0., 1.);
    float x2 = std::clamp(cx + w / 2., 0., 1.);
    float y2 = std::clamp(cy + h / 2., 0., 1.);
    boxes.push_back({x1, y1, x2, y2, max_score, max_score_index - 4});
  }
  nms(boxes, nms_iou_threshold);
  return boxes;
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

struct OrtTensorRTProviderOptionsV2 {
  int device_id;                                // cuda device id.
  int has_user_compute_stream;                  // indicator of user specified CUDA compute stream.
  void* user_compute_stream;                    // user specified CUDA compute stream.
  int trt_max_partition_iterations;             // maximum iterations for TensorRT parser to get capability
  int trt_min_subgraph_size;                    // minimum size of TensorRT subgraphs
  size_t trt_max_workspace_size;                // maximum workspace size for TensorRT.
  int trt_fp16_enable;                          // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
  int trt_int8_enable;                          // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
  const char* trt_int8_calibration_table_name;  // TensorRT INT8 calibration table name.
  int trt_int8_use_native_calibration_table;    // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
  int trt_dla_enable;                           // enable DLA. Default 0 = false, nonzero = true
  int trt_dla_core;                             // DLA core number. Default 0
  int trt_dump_subgraphs;                       // dump TRT subgraph. Default 0 = false, nonzero = true
  int trt_engine_cache_enable;                  // enable engine caching. Default 0 = false, nonzero = true
  const char* trt_engine_cache_path;            // specify engine cache path
  int trt_engine_decryption_enable;             // enable engine decryption. Default 0 = false, nonzero = true
  const char* trt_engine_decryption_lib_path;   // specify engine decryption library path
  int trt_force_sequential_engine_build;        // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
  int trt_context_memory_sharing_enable;        // enable context memory sharing between subgraphs. Default 0 = false, nonzero = true
  int trt_layer_norm_fp32_fallback;             // force Pow + Reduce ops in layer norm to FP32. Default 0 = false, nonzero = true
  int trt_timing_cache_enable;                  // enable TensorRT timing cache. Default 0 = false, nonzero = true
  int trt_force_timing_cache;                   // force the TensorRT cache to be used even if device profile does not match. Default 0 = false, nonzero = true
  int trt_detailed_build_log;                   // Enable detailed build step logging on TensorRT EP with timing for each engine build. Default 0 = false, nonzero = true
  int trt_build_heuristics_enable;              // Build engine using heuristics to reduce build time. Default 0 = false, nonzero = true
  int trt_sparsity_enable;                      // Control if sparsity can be used by TRT. Default 0 = false, 1 = true
  int trt_builder_optimization_level;           // Set the builder optimization level. WARNING: levels below 3 do not guarantee good engine performance, but greatly improve build time.  Default 3, valid range [0-5]
  int trt_auxiliary_streams;                    // Set maximum number of auxiliary streams per inference stream. Setting this value to 0 will lead to optimal memory usage. Default -1 = heuristics
  const char* trt_tactic_sources;               // pecify the tactics to be used by adding (+) or removing (-) tactics from the default
                                                // tactic sources (default = all available tactics) e.g. "-CUDNN,+CUBLAS" available keys: "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS"
  const char* trt_extra_plugin_lib_paths;       // specify extra TensorRT plugin library paths
  const char* trt_profile_min_shapes;           // Specify the range of the input shapes to build the engine with
  const char* trt_profile_max_shapes;           // Specify the range of the input shapes to build the engine with
  const char* trt_profile_opt_shapes;           // Specify the range of the input shapes to build the engine with
  int trt_cuda_graph_enable;                    // Enable CUDA graph in ORT TRT
};

class OnnxModel {
private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session{nullptr};
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;

  std::vector<const char *> input_names_char;
  std::vector<const char *> output_names_char;

  std::map<int, std::string> class_map;
  int num_classes;
  std::vector<int> imgsz; // h, w
  int gpu_id;
  bool preprocess_on_gpu;

  cv::cuda::GpuMat image_on_gpu, resized_image_on_gpu,
      padded_resized_image_on_gpu;
  cv::cuda::GpuMat chw_padded_resized_image_on_gpu;
  cv::cuda::GpuMat float_chw_padded_resized_image_on_gpu;

public:
  // Initialize OnnxModel
  OnnxModel(const std::string &model_file, bool use_cuda = false, bool use_trt = false,
            bool preprocess_on_gpu = false, int gpu_id = 0)
      : env(ORT_LOGGING_LEVEL_WARNING, "single_image_model_inference"),
        gpu_id(gpu_id), preprocess_on_gpu(preprocess_on_gpu), num_classes(0) {

    if (use_cuda) {
      if (use_trt) {
        int device_id = gpu_id;
        int trt_max_partition_iterations = 1000;
        int trt_min_subgraph_size = 1;
        size_t trt_max_workspace_size = 1 << 30;
        bool trt_fp16_enable = true;
        bool trt_int8_enable = false;
        std::string trt_int8_calibration_table_name = "";
        bool trt_int8_use_native_calibration_table = false;
        bool trt_dla_enable = false;
        int trt_dla_core = 0;
        bool trt_dump_subgraphs = false;
        bool trt_engine_cache_enable = false;
        std::string trt_engine_cache_path = "";
        bool trt_engine_decryption_enable = false;
        std::string trt_engine_decryption_lib_path = "";
        bool trt_force_sequential_engine_build = false;
        bool trt_context_memory_sharing_enable = false;
        bool trt_layer_norm_fp32_fallback = false;
        bool trt_timing_cache_enable = false;
        bool trt_force_timing_cache = false;
        bool trt_detailed_build_log = false;
        bool trt_build_heuristics_enable = false;
        bool trt_sparsity_enable = false;
        int trt_builder_optimization_level = 3;
        int trt_auxiliary_streams = -1;
        std::string trt_tactic_sources = "";
        std::string trt_extra_plugin_lib_paths = "";
        std::string trt_profile_min_shapes = "images:1x3x640x640";
        std::string trt_profile_max_shapes = "images:64x3x640x640";
        std::string trt_profile_opt_shapes = "images:8x3x640x640";
        bool trt_cuda_graph_enable = false;

        OrtTensorRTProviderOptionsV2 tensorrt_options;
        tensorrt_options.device_id = device_id;
        tensorrt_options.has_user_compute_stream = 0;
        tensorrt_options.user_compute_stream = nullptr;
        tensorrt_options.trt_max_partition_iterations = trt_max_partition_iterations;
        tensorrt_options.trt_min_subgraph_size = trt_min_subgraph_size;
        tensorrt_options.trt_max_workspace_size = trt_max_workspace_size;
        tensorrt_options.trt_fp16_enable = trt_fp16_enable;
        tensorrt_options.trt_int8_enable = trt_int8_enable;
        tensorrt_options.trt_int8_calibration_table_name = trt_int8_calibration_table_name.c_str();
        tensorrt_options.trt_int8_use_native_calibration_table = trt_int8_use_native_calibration_table;
        tensorrt_options.trt_dla_enable = trt_dla_enable;
        tensorrt_options.trt_dla_core = trt_dla_core;
        tensorrt_options.trt_dump_subgraphs = trt_dump_subgraphs;
        tensorrt_options.trt_engine_cache_enable = trt_engine_cache_enable;
        tensorrt_options.trt_engine_cache_path = trt_engine_cache_path.c_str();
        tensorrt_options.trt_engine_decryption_enable = trt_engine_decryption_enable;
        tensorrt_options.trt_engine_decryption_lib_path = trt_engine_decryption_lib_path.c_str();
        tensorrt_options.trt_force_sequential_engine_build = trt_force_sequential_engine_build;
        tensorrt_options.trt_context_memory_sharing_enable = trt_context_memory_sharing_enable;
        tensorrt_options.trt_layer_norm_fp32_fallback = trt_layer_norm_fp32_fallback;
        tensorrt_options.trt_timing_cache_enable = trt_timing_cache_enable;
        tensorrt_options.trt_force_timing_cache = trt_force_timing_cache;
        tensorrt_options.trt_detailed_build_log = trt_detailed_build_log;
        tensorrt_options.trt_build_heuristics_enable = trt_build_heuristics_enable;
        tensorrt_options.trt_sparsity_enable = trt_sparsity_enable;
        tensorrt_options.trt_builder_optimization_level = trt_builder_optimization_level;
        tensorrt_options.trt_auxiliary_streams = trt_auxiliary_streams;
        tensorrt_options.trt_tactic_sources = trt_tactic_sources.c_str();
        tensorrt_options.trt_extra_plugin_lib_paths = trt_extra_plugin_lib_paths.c_str();
        tensorrt_options.trt_profile_min_shapes = trt_profile_min_shapes.c_str();
        tensorrt_options.trt_profile_max_shapes = trt_profile_max_shapes.c_str();
        tensorrt_options.trt_profile_opt_shapes = trt_profile_opt_shapes.c_str();
        tensorrt_options.trt_cuda_graph_enable = trt_cuda_graph_enable;

        session_options.AppendExecutionProvider_TensorRT_V2(tensorrt_options);

        // OrtTensorRTProviderOptions trt_options{};
        // trt_options.device_id = gpu_id;
        // trt_options.trt_fp16_enable = 1;
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, gpu_id));
      }
      
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
    fetch_model_metadata();
    setup();
  }

  // set up the model
  void setup() {
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
      input_names.emplace_back(
          session.GetInputNameAllocated(i, allocator).get());
      input_shape =
          session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    for (auto &s : input_shape) {
      if (s < 0) {
        s = 1;
      }
    }

    input_shape[2] = imgsz[0];
    input_shape[3] = imgsz[1];

    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
      output_names.emplace_back(
          session.GetOutputNameAllocated(i, allocator).get());
      output_shape =
          session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    for (auto &s : output_shape) {
      if (s < 0) {
        s = 1;
      }
    }

    output_shape[1] = 4 + num_classes;

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    std::cout << "Input names: " << input_names[0] << std::endl;
    std::cout << "Input shape: " << print_shape(input_shape) << std::endl;
    std::cout << "Total number of elements: " << calculate_product(input_shape)
              << std::endl;

    std::cout << "Output names: " << output_names[0] << std::endl;
    std::cout << "Output shape: " << print_shape(output_shape) << std::endl;
    std::cout << "Total number of elements: " << calculate_product(output_shape)
              << std::endl;

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

  void fetch_model_metadata() {
    // Fetch model metadata
    auto model_metadata = session.GetModelMetadata();
    auto custom_metadata_map_keys =
        model_metadata.GetCustomMetadataMapKeysAllocated(allocator);
    std::cout << "Model Metadata: " << std::endl;
    for (auto &key : custom_metadata_map_keys) {
      std::string key_str = key.get();
      std::string value_str =
          model_metadata
              .LookupCustomMetadataMapAllocated(key_str.c_str(), allocator)
              .get();
      std::cout << "key: " << key_str << " value: " << value_str << std::endl;
      if (key_str == "names") {
        std::stringstream ss(value_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
          int key;
          std::string value;
          std::stringstream pairStream(item);
          std::string pairItem;

          if (std::getline(pairStream, pairItem, ':')) {
            pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), '{'),
                           pairItem.end());
            pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), ' '),
                           pairItem.end());
            key = stoi(pairItem);
          }

          if (std::getline(pairStream, pairItem, ':')) {
            pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), '\''),
                           pairItem.end());
            pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), ' '),
                           pairItem.end());
            pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), '}'),
                           pairItem.end());
            value = pairItem;
          }

          class_map.insert(std::make_pair(key, value));
        }
        num_classes = class_map.size();
      } else if (key_str == "imgsz") {
        std::stringstream ss(value_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
          item.erase(std::remove(item.begin(), item.end(), '['), item.end());
          item.erase(std::remove(item.begin(), item.end(), ']'), item.end());

          imgsz.push_back(std::stoi(item));
        }
      }
    }

    assert(num_classes > 0);
    std::cout << "Number of classes: " << num_classes << std::endl;
    std::cout << "Class map: " << std::endl;
    for (auto &pair : class_map) {
      std::cout << pair.first << " : " << pair.second << std::endl;
    }

    std::cout << "Image size: " << std::endl;
    for (auto &item : imgsz) {
      std::cout << item << " ";
    }
    std::cout << std::endl;
  }

  std::tuple<std::vector<float>, std::vector<std::vector<int64_t>>>
  preprocess(const std::vector<cv::Mat> &images) {
    input_shape[0] = images.size();

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
    if (channel != 3) {
      throw std::runtime_error(
          "Invalid number of channels. Expected: 3, Got: " +
          std::to_string(channel));
    }

    int height = static_cast<int>(input_shape[2]);
    int width = static_cast<int>(input_shape[3]);

    std::vector<std::vector<int64_t>> batch_pad_hw;
    for (std::size_t i = 0; i < images.size(); i++) {

      cv::Mat image = images[i].clone();

      int old_height = image.rows;
      int old_width = image.cols;

      float r = std::min(static_cast<float>(height) / old_height,
                         static_cast<float>(width) / old_width);
      r = std::min(r, 1.0f);

      int new_height = static_cast<int>(old_height * r);
      int new_width = static_cast<int>(old_width * r);

      cv::resize(image, image, cv::Size(new_width, new_height),
                 cv::INTER_LINEAR);

      int top = 0, bottom = 0, left = 0, right = 0;
      bottom = height - new_height;
      right = width - new_width;

      cv::copyMakeBorder(image, image, top, bottom, left, right,
                         cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

      assert(image.rows == height && image.cols == width);
      batch_pad_hw.push_back({bottom, right});

      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

      cv::Mat image_float32;
      image.convertTo(image_float32, CV_32FC3);
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
    return {batch_input_tensor_values, batch_pad_hw};
  }

  std::tuple<cv::cuda::GpuMat, std::vector<std::vector<int64_t>>>
  preprocess_cuda(const std::vector<cv::Mat> &images) {
    int resize_height = static_cast<int>(input_shape[2]);
    int resize_width = static_cast<int>(input_shape[3]);

    if (chw_padded_resized_image_on_gpu.cols *
            chw_padded_resized_image_on_gpu.rows *
            chw_padded_resized_image_on_gpu.channels() !=
        input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]) {
      chw_padded_resized_image_on_gpu = cv::cuda::GpuMat(
          1, input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3],
          CV_8U);
    }

    std::vector<std::vector<int64_t>> batch_pad_hw;

    for (int i = 0; i < images.size(); i++) {
      cv::Mat image = images[i];
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
      batch_pad_hw.push_back({bottom, right});

      image_on_gpu.upload(image);
      cv::cuda::resize(
          image_on_gpu, resized_image_on_gpu,
          cv::Size(non_padded_resize_width, non_padded_resize_height),
          cv::INTER_LINEAR);
      cv::cuda::copyMakeBorder(
          resized_image_on_gpu, padded_resized_image_on_gpu, top, bottom, left,
          right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
      cv::cuda::cvtColor(padded_resized_image_on_gpu,
                         padded_resized_image_on_gpu, cv::COLOR_BGR2RGB);
      assert(padded_resized_image_on_gpu.rows == resize_height &&
             padded_resized_image_on_gpu.cols == resize_width);

      std::vector<cv::cuda::GpuMat> rgb_channels = {
          cv::cuda::GpuMat(
              resize_height, resize_width, CV_8U,
              &(chw_padded_resized_image_on_gpu
                    .ptr()[0 + resize_height * resize_width * 3 * i])),
          cv::cuda::GpuMat(resize_height, resize_width, CV_8U,
                           &(chw_padded_resized_image_on_gpu
                                 .ptr()[resize_height * resize_width +
                                        resize_height * resize_width * 3 * i])),
          cv::cuda::GpuMat(
              resize_height, resize_width, CV_8U,
              &(chw_padded_resized_image_on_gpu
                    .ptr()[resize_height * resize_width * 2 +
                           resize_height * resize_width * 3 * i]))};

      cv::cuda::split(padded_resized_image_on_gpu, rgb_channels);
    }
    chw_padded_resized_image_on_gpu.convertTo(
        float_chw_padded_resized_image_on_gpu, CV_32F, 1.0 / 255.0);

    return {float_chw_padded_resized_image_on_gpu, batch_pad_hw};
  }

  std::vector<std::vector<std::vector<float>>>
  operator()(const std::vector<cv::Mat> &images) {
    // Preprocess the input image
    try {
      input_shape[0] = images.size(); // equals to batch size
      std::vector<Ort::Value> output_tensors;
      std::vector<std::vector<int64_t>> batch_pad_hw;

      if (preprocess_on_gpu) {
        auto [input_tensor_values, batch_pad_hw_inner_block] =
            preprocess_cuda(images);
        batch_pad_hw = std::move(batch_pad_hw_inner_block);
        auto *dataPointer = input_tensor_values.ptr<void>();

        Ort::MemoryInfo mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator,
                                 gpu_id, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor(
            mem_info, dataPointer,
            input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] *
                sizeof(float),
            input_shape.data(), input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        output_tensors = std::move(
            session.Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                        input_tensors.data(), input_names_char.size(),
                        output_names_char.data(), output_names_char.size()));
      } else {
        auto [input_tensor_values, batch_pad_hw_inner_block] =
            preprocess(images);
        batch_pad_hw = std::move(batch_pad_hw_inner_block);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            mem_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()));

        output_tensors = std::move(
            session.Run(Ort::RunOptions{nullptr}, input_names_char.data(),
                        input_tensors.data(), input_names_char.size(),
                        output_names_char.data(), output_names_char.size()));
      }

      // double-check the dimensions of the output tensors
      // NOTE: the number of output tensors is equal to the number of output
      // nodes specifed in the Run() call
      assert(output_tensors.size() == output_names.size() &&
             output_tensors[0].IsTensor());

      // Convert the output tensor to a vector
      auto *output_tensor =
          output_tensors.front().GetTensorMutableData<float>();

      Ort::TypeInfo output_type_info = output_tensors.front().GetTypeInfo();
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      output_shape = output_tensor_info.GetShape();

      auto num_outputs = output_shape[1];
      auto num_anchors = output_shape[2];

      std::vector<float> flatten_batch_output_tensor_values(
          output_tensor, output_tensor + calculate_product(output_shape));

      std::vector<std::vector<std::vector<float>>> batch_output_tensor_values;
      for (std::size_t i = 0; i < images.size(); i++) {
        std::vector<float> output_tensor_values(
            flatten_batch_output_tensor_values.begin() +
                (i * num_outputs * num_anchors),
            flatten_batch_output_tensor_values.begin() +
                ((i + 1) * num_outputs * num_anchors));
        batch_output_tensor_values.emplace_back(postprocess(
            output_tensor_values, input_shape, batch_pad_hw[i], output_shape));
      }

      return batch_output_tensor_values;
    } catch (const Ort::Exception &exception) {
      std::cout << "ERROR running model inference: " << exception.what()
                << std::endl;
      exit(-1);
    }
  }

  void set_preprocess_on_gpu(bool preprocess_on_gpu) {
    this->preprocess_on_gpu = preprocess_on_gpu;
  }

  std::string get_class_name(int class_id) { return class_map[class_id]; }
  const std::vector<std::int64_t>& get_input_shape() { return input_shape; }
};

int main(int argc, char *argv[]) { 
  std::string model_file = "/app/models/yolov8n.onnx";
  std::string filename = "/app/videos/desk.mp4";

  std::chrono::duration<double> diff_gpu, diff_gpu_trt;
  bool use_cuda = true;
  bool preprocess_on_gpu = true;

  for (bool use_trt : {false, true}) {
    OnnxModel model(model_file, use_cuda, use_trt, preprocess_on_gpu);

    model.set_preprocess_on_gpu(preprocess_on_gpu);

    //warmup
    model({cv::Mat(model.get_input_shape()[2], model.get_input_shape()[3], CV_8UC3, cv::Scalar(0, 0, 0))});

    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
      std::cout << "Error opening video stream or file" << std::endl;
      return -1;
    }
    
    std::vector<cv::Mat> images(1);
    auto start = std::chrono::high_resolution_clock::now(); // start timing
    while (true) {
      cv::Mat frame;
      cap >> frame;
      if (frame.empty())
        break;
      images[0] = std::move(frame);
      auto batch_output = model(images);
      for (int j = 0; j < batch_output.size(); j++) {
        auto output = batch_output[j];
        std::cout << "Output size: " << output.size() << std::endl;
        for (int k = 0; k < output.size(); k++) {
          float x1 = output[k][0] * images[j].cols;
          float y1 = output[k][1] * images[j].rows;
          float x2 = output[k][2] * images[j].cols;
          float y2 = output[k][3] * images[j].rows;

          float score = output[k][4];
          int class_id = output[k][5];

          std::string class_name = model.get_class_name(class_id);
          std::cout << "x1: " << x1 << " y1: " << y1 << " x2: " << x2
                    << " y2: " << y2 << " score: " << score
                    << " class_id: " << class_id
                    << " class_name: " << class_name << std::endl;
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now(); // end timing
    cap.release();
    
    if (use_trt)
      diff_gpu_trt = end - start;
    else
      diff_gpu = end - start;
  }

  std::cout << "Execution time using GPU: " << diff_gpu.count() << " s\n";
  std::cout << "Execution time using GPU-trt: " << diff_gpu_trt.count() << " s\n";

  return 0;
}