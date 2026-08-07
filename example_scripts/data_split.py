import os
import shutil
from torch.utils.data import random_split

imgs = [f for f in os.listdir('labeled_ds/images') if f.endswith('.jpg')] # source dataset
train_ds, val_ds = random_split(imgs, [0.8, 0.2])
for ds, name in [(train_ds, 'train'), (val_ds, 'val')]:
    os.makedirs(f'dataset/images/{name}', exist_ok=True) # splitted dataset folder
    os.makedirs(f'dataset/labels/{name}', exist_ok=True) # splitted dataset folder
    for img in ds:
        shutil.copy(f'labeled_ds/images/{img}', f'dataset/images/{name}/{img}') # copy source dataset to splitted folder
        lbl = img.replace('.jpg', '.txt')
        if os.path.exists(f'labeled_ds/labels/{lbl}'): # source dataset
            shutil.copy(f'labeled_ds/labels/{lbl}', f'dataset/labels/{name}/{lbl}') # copy source dataset to splitted folder
print("Split dataset successfully!")