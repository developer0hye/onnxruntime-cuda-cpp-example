# onnxruntime-cuda-cpp-example

I am still working on this...

Will it be finished?

# Setup

```bash
git clone https://github.com/developer0hye/onnxruntime-cuda-cpp-example
cd onnxruntime-cuda-cpp-example
docker build . -t onnxruntime-cuda:1.0
sudo docker run -it --runtime=nvidia --gpus all onnxruntime-cuda:1.0
```

(on a container)
```
cd build
./single_image_model_inference
./fixed_batch_image_model_inference
./dynamic_batch_image_model_inference
```
