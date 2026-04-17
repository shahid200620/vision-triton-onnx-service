import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

pt_model = YOLO("yolov8s.pt")
session = ort.InferenceSession("yolov8s.onnx")

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    pt_result = pt_model.model(x)
    pt_output = pt_result[0].cpu().numpy()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

onnx_output = session.run(
    [output_name],
    {input_name: x.cpu().numpy()}
)[0]

if np.allclose(pt_output, onnx_output, atol=1e-4):
    print("Validation successful: ONNX output matches PyTorch output")
else:
    raise AssertionError("Validation failed")
