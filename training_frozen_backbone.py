from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting Frozen Backbone Training...")
model.train(
    data="dataset.yaml",
    epochs=100,
    batch=16,
    imgsz=480,         
    patience=20,
    device='cpu',
    workers=4,
    optimizer="AdamW",
    lr0=0.005,         # ปรับ LR สูงขึ้นเล็กน้อยได้เพราะ Backbone ปลอดภัย
    freeze=10          # ล็อคน้ำหนัก 10 เลเยอร์แรกของโมเดล
)

best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    half=True          # บีบอัดเป็น FP16
)[cite: 2]