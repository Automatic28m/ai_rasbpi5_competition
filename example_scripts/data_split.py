import os, shutil
from torch.utils.data import random_split

# Missing random

imgs = [f for f in os.listdir('raw_images') if f.endswith('.jpg')]
train_ds, val_ds = random_split(imgs, [0.8, 0.2])


for ds, name in [(train_ds, 'train'), (val_ds, 'val')]:
    os.makedirs(f'dataset/images/{name}', exist_ok=True)
    os.makedirs(f'dataset/labels/{name}', exist_ok=True)
    for img in ds:
        shutil.copy(f'raw_images/{img}', f'dataset/images/{name}/{img}')
        lbl = img.replace('.jpg', '.txt')
        if os.path.exists(f'raw_labels/{lbl}'):
            shutil.copy(f'raw_labels/{lbl}', f'dataset/labels/{name}/{lbl}')