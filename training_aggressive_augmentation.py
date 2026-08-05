'''
3. Aggressive Augmentation
The best defense against a tiny dataset. 
This script artificially expands your data by manipulating, rotating, and blending images together, 
forcing the model to learn the object rather than memorizing the background.
'''


from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting Aggressive Augmentation...")
model.train(
    data="dataset.yaml",
    epochs=200,        # Extended epochs for learning through distortions
    batch=8,
    imgsz=320,
    patience=50,       # Higher patience to tolerate loss fluctuations
    device='cpu',
    workers=4,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    mosaic=1.0,
    degrees=10,
    translate=0.1
)

best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    end2end=False
)