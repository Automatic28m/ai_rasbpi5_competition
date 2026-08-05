from ultralytics import YOLO

# โหลดโมเดล YOLO26 ตามโจทย์ (สมมติใช้ตัว nano เพื่อความรวดเร็ว)
model = YOLO("yolo26n.pt")

results = model.train(
    data="dataset.yaml",
    epochs=100,          # วางเป้าหมายไว้ที่ 100 รอบ
    batch=16,            # ใช้ RAM 16GB ให้คุ้มค่า
    imgsz=320,           # ขนาดภาพเล็กเพื่อให้ CPU ประมวลผลไว
    patience=20,         # หยุดอัตโนมัติหาก 20 รอบหลังสุดโมเดลไม่เก่งขึ้น
    device='cpu',        # บังคับรันบน CPU เพื่อความชัวร์ ไม่ให้ Error
    workers=4            # ใช้ Core ของ i5-13400F ช่วยโหลดข้อมูล
)

# ==========================================
# 4. Export Model เป็น ONNX
# ==========================================
# ดึงน้ำหนักโมเดลที่ดีที่สุดจากการ Train รอบล่าสุดมาแปลงไฟล์
best_model = YOLO("runs/detect/train/weights/best.pt")

best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    half=True,  # <-- เพิ่มบรรทัดนี้เพื่อลดขนาดโมเดลลงครึ่งหนึ่ง (FP16)
    data="dataset.yaml"  # <-- ต้องใช้ไฟล์ config เพื่อให้โมเดล Calibrate ค่าความแม่นยำ
)
