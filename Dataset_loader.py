import torch
import os
from PIL import Image


class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        print(root_dir)
        self.root_dir = root_dir
        self.paths = os.listdir(root_dir)[:200]
        print(self.paths[0])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.paths[index])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
