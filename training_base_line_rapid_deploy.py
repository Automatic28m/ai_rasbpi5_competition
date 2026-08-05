'''
1. Baseline Rapid Deploy
Run this first to secure a working prototype. It trains fast and exports a quantized (FP16) model for maximum processing speed on the Raspberry Pi 5
'''

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting Baseline Rapid Deploy...")
model.train(
    data="dataset.yaml",
    epochs=100,
    batch=8,
    imgsz=320,
    patience=20,
    device='cpu',
    workers=4,
    optimizer='AdamW',
    lr0=0.001
)

# Export as an FP16 ONNX model for the smallest possible file size
best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    half=True,
    end2end=False
)