import cv2, os

os.makedirs('extract_dataset', exist_ok=True)

cap = cv2.VideoCapture('./dataset/3MinRawVideo.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % fps == 0:
        cv2.imwrite(f'extract_dataset/img_{count}.jpg', frame)
        
    count += 1
    
cap.release()
print("Finish")