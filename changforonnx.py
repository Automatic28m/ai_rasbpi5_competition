from ultralytics import YOLO

best_model = YOLO(
    "runs/detect/train-16/weights/best.pt",
    task="detect"
)

# บังคับให้ใช้ detection head แบบปกติ
best_model.model.model[-1].end2end = False

best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    end2end=False
)