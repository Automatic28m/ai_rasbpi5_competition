from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting High-Res Feature Extractor Training...")
model.train(
    data="dataset.yaml",
    epochs=150,
    batch=16,
    imgsz=480,         # ขยายภาพขึ้นเพื่อความแม่นยำ 
    patience=30,
    device='cpu',
    workers=4,
    lr0=0.001,
    cos_lr=True
)

# บีบขนาดภาพกลับมาที่ 320 ตอน Export เพื่อให้ประมวลผลบนบอร์ดได้ FPS สูง
best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320
)[cite: 2]