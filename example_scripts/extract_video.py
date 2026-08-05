import cv2
import os

# สร้างโฟลเดอร์เก็บภาพ
os.makedirs('dataset', exist_ok=True)

# โหลดวิดีโอ (เปลี่ยนชื่อไฟล์ตามจริง)
cap = cv2.VideoCapture('video_dataset.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS)) # หาค่า FPS ของวิดีโอ
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    
    # เซฟภาพทุกๆ 1 วินาที (เมื่อ count หาร fps ลงตัว)
    if count % fps == 0:
        cv2.imwrite(f'dataset/img_{count}.jpg', frame)
        
    count += 1

cap.release()
print("Extract เสร็จสิ้น!")