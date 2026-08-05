from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting Baseline Rapid Deploy Training...")
model.train(
    data="dataset.yaml",
    epochs=100,
    batch=16,            # RAM 16GB รับมือ batch 16 ได้สบาย
    imgsz=320,           # ภาพเล็ก CPU คำนวณจบไว
    patience=20,
    device='cpu',        # บังคับรัน CPU
    workers=4            # ใช้ Core ของ i5 ช่วยโหลดข้อมูล
)

# แปลงเป็น ONNX แบบ FP16 เพื่อให้ไฟล์เล็กที่สุด
best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    half=True
)[cite: 2]