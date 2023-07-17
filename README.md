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
- Utilities
    - parse_metadata

# Setup

```bash
git clone https://github.com/developer0hye/onnxruntime-cuda-cpp-example
cd onnxruntime-cuda-cpp-example
docker build . -t onnxruntime-cuda:1.0
sudo docker run -it --runtime=nvidia --gpus all onnxruntime-cuda:1.0
```

# Run

(on a container)
```
cd ./build/examples/
./single_image_model_inference
./fixed_batch_image_model_inference
./dynamic_batch_image_model_inference
```
