# onnxruntime-cuda-cpp-example

I am still working on this...

Will it be finished?

# Examples

- Image Classification
    - single_image_model_inference
    - fixed_batch_image_model_inference
    - dynamic_batch_image_model_inference
- Object Detection
    - dynamic_batch_yolov8
    - dynamic_batch_yolov8_tensorrt
    - dynamic_batch_yolov8_with_cuda_based_opencv_image_preprocessing
    - dynamic_batch_yolov8_single_video_single_frame_inference
    - dynamic_batch_yolov8_single_video_single_frame_inference_tensorrt
    - dynamic_batch_yolov8_single_video_multiple_frames_inference
    - dynamic_batch_yolov8_single_video_multiple_frames_inference_tensorrt
    - dynamic_batch_yolov8_multiple_videos_inference_per_video(not yet implemented)
    - dynamic_batch_yolov8_multiple_videos_parallel_inference_with_queue(not yet implemented)
- Utilities
    - parse_metadata
    - cuda_based_opencv_image_preprocessing

# Setup

```bash
git clone https://github.com/developer0hye/onnxruntime-cuda-cpp-example
cd onnxruntime-cuda-cpp-example
docker build . -t onnxruntime-cuda:1.0
sudo docker run -it --runtime=nvidia --gpus all onnxruntime-cuda:1.0
```

# Run

(on a container)
```bash
./{example_you_want_to_run}
```
