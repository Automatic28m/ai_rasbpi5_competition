'''
4. Frozen Backbone
Highly recommended for this specific scenario. 
Freezing the first 10 layers saves massive CPU computation time, 
allowing you to train on larger images without slowing down the process. 
It also prevents your small dataset from overwriting the model's core edge-detection abilities.
'''


from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Starting Frozen Backbone Training...")
model.train(
    data="dataset.yaml",
    epochs=100,
    batch=8,
    imgsz=480,
    patience=20,
    device='cpu',
    workers=4,
    optimizer='AdamW',
    lr0=0.005,         # Slightly higher LR is safe when the backbone is frozen
    freeze=10
)

best_model = YOLO("runs/detect/train/weights/best.pt")
best_model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=320,
    half=True,
    end2end=False
)