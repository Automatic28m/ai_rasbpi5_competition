import os
import shutil
from torch.utils.data import random_split

imgs = [f for f in os.listdir('images') if f.endswith('.jpg')]
train_ds, val_ds = random_split(imgs, [0.8, 0.2])
for ds, name in [(train_ds, 'train'), (val_ds, 'val')]:
    os.makedirs(f'dataset/images/{name}', exist_ok=True)
    os.makedirs(f'dataset/labels/{name}', exist_ok=True)
    for img in ds:
        shutil.copy(f'images/{img}', f'dataset/images/{name}/{img}')
        lbl = img.replace('.jpg', '.txt')
        if os.path.exists(f'label/{lbl}'):
            shutil.copy(f'label/{lbl}', f'dataset/labels/{name}/{lbl}')
print("Split dataset successfully!")