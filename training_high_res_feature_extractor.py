'''
2. High-Res Feature Extractor
This script focuses on the 70% accuracy requirement by processing larger images to capture more granular details of the target object.
'''
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(
    data="dataset.yaml",
    epochs=100,
    batch=4,
    imgsz=480,
    patience=20,
    device="cpu",
    workers=4,
    optimizer="AdamW",
    lr0=0.001, # Learning Rate
    cos_lr=True, # COsine Annealing: Reducing Learning Rate slowly
    
    scale=0.3, # Shrink/expand +/- randomly 30%
    fliplr=0.5, # Randomly flip images left/right 50%
    degree=10.0, # Randomly rotate images +/- 10 degree
    hsv_v=0.3, # Randomly adjust brightness +/- 30%
    mosaic=0.0 # Turn off 4 images combining
)

best_model = YOLO("./run/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12, # ONNX Operation version
    simplify=True, # Cut off complex mathemetic for small file size
    dynamic=False, # Static input shape for more FPS on Pi5
    imgsz=480,
    end2end=False # Turn off Non-Maximum Suppression
)