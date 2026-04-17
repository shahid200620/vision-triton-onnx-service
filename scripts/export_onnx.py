from ultralytics import YOLO
import os

model = YOLO("yolov8s.pt")

model.export(
format="onnx",
opset=12,
dynamic=True,
imgsz=640
)

if os.path.exists("yolov8s.onnx"):
    print("Export completed successfully")
else:
    print("Export failed")
