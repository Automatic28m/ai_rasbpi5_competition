'''
2. High-Res Feature Extractor
This script focuses on the 70% accuracy requirement by processing larger images to capture more granular details of the target object.
'''


from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting High-Res Feature Extractor...")
model.train(
    data="dataset.yaml",
    epochs=100,
    batch=4,           # Lowered batch size to compensate for larger images
    imgsz=480,
    patience=15,
    device='cpu',
    workers=4,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,

    scale=0.3,
    fliplr=0.5,
    degrees=10.0,
    hsv_v=0.3,
    mosaic=0.0
)

# Export at 320 to maintain high FPS during the Pi 5 testing phase
best_model = YOLO(
    "./runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=480,
    half=True,
    end2end=False
)