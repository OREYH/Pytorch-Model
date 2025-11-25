import os, random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ImageTransform:
    """Transforms that match diffusers DDPMPipeline defaults for CelebA-HQ 64x64."""
    def __init__(self, image_size=64):
        base = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        self.data_transform = {
            'train': transforms.Compose([transforms.RandomHorizontalFlip()] + base),
            'valid': transforms.Compose(base)
        }
    
    def __call__(self, image, phase='train'):
        return self.data_transform[phase](image)


class Denormalization(nn.Module):
    """Maps tensors from [-1, 1] back to [0, 1] for visualization."""
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor):
        tensor = tensor.detach().cpu()
        tensor = 0.5 * tensor + 0.5
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        if tensor.dim() == 4:  ## [B, C, H, W] -> [B, H, W, C]
            tensor = tensor.permute(0, 2, 3, 1)
        elif tensor.dim() == 3:  ## [C, H, W] -> [H, W, C]
            tensor = tensor.permute(1, 2, 0)
        
        return tensor


class CelebAHQDataset(Dataset):
    """Simple dataset that reads image files directly (no class labels)."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


def _list_image_files(data_path):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_paths = []
    for root, _, files in os.walk(data_path):
        for name in files:
            if name.lower().endswith(exts):
                image_paths.append(os.path.join(root, name))
    return sorted(image_paths)


def create_loader(data_path, image_size=64, batch_size=64, valid_ratio=0.0, num_workers=4, seed=0):
    """
    Returns DataLoaders for CelebA-HQ images normalized to [-1, 1].
    - data_path: directory containing raw images (any folder layout is fine).
    - valid_ratio: fraction used for validation split (0 disables validation).
    """
    image_paths = _list_image_files(data_path)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found under {data_path}")
    
    random.Random(seed).shuffle(image_paths)
    
    split = int(len(image_paths) * valid_ratio)
    valid_paths = image_paths[:split]
    train_paths = image_paths[split:] if split > 0 else image_paths
    
    transform = ImageTransform(image_size)
    train_dataset = CelebAHQDataset(train_paths, lambda img: transform(img, 'train'))
    valid_dataset = CelebAHQDataset(valid_paths, lambda img: transform(img, 'valid')) if split > 0 else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    
    return train_loader, valid_loader


if __name__ == '__main__':
    ## quick sanity check (counts only)
    train_loader, valid_loader = create_loader(data_path='./data/celebhq', valid_ratio=0.0, batch_size=8)
    print(f"train batches: {len(train_loader)}")
    if valid_loader is not None:
        print(f"valid batches: {len(valid_loader)}")
