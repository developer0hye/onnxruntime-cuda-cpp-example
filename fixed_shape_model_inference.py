import onnxruntime
import cv2
import numpy as np

np.set_printoptions(suppress=True) # suppress scientific notation

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum()

model = onnxruntime.InferenceSession("mnist-12.onnx", providers=["CUDAExecutionProvider"])
model_input_shape = model.get_inputs()[0].shape
model_input_name = model.get_inputs()[0].name

input = cv2.imread("7.png") # (28, 28, 3)
input = cv2.resize(input, (model_input_shape[3], model_input_shape[2])) # (28, 28, 3)
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) # (28, 28)
input = input[np.newaxis, np.newaxis, :, :] # (1, 1, 28, 28)
input = input.astype(np.float32) / 255.0

output = model.run(None, {model_input_name: input}) # list of numpy.ndarray
output = output[0] # logits
output = softmax(output) # softmax

print(f"output: {output}, predicted number: {np.argmax(output)}, predicted probability: {np.max(output)}")
