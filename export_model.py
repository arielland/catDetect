"""
Run this ONCE on your Windows PC to export the YOLOv8n model to ONNX format.
The resulting yolov8n.onnx file (~12MB) is then copied to the RPi.

Usage (on Windows):
    pip install ultralytics
    python export_model.py

Then copy to RPi:
    scp yolov8n.onnx relland@catdetector.local:~/catDetect/
"""

from ultralytics import YOLO

print("Downloading yolov8n.pt and exporting to ONNX...")
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, simplify=True)
print("\nDone! Copy yolov8n.onnx to your RPi:")
print("  scp yolov8n.onnx relland@catdetector.local:~/catDetect/")
