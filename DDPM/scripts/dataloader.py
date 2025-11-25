import os, glob
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class ImageTransform():
    """ CelebA-HQ를 128x128 사이즈의 이미지로 줄이기 """
    def __init__(self, image_size=128, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        base = [
            transforms.Resize(image_size, interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(), # [B, C, H, W], (0 ~ 1)
            transforms.Normalize(mean=mean, std=std)]
        
        self.data_transform = {
            'train': transforms.Compose([transforms.RandomHorizontalFlip()] + base),
            'valid': transforms.Compose(base)}
    
    def __call__(self, image, phase='train'):
        return self.data_transform[phase](image)

class Denormalization(nn.Module):
    """ [-1, 1] -> [0, 1] 인 [B, C, H, W] 형태로 되돌림 """
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor:torch.Tensor):
        tensor = tensor.detach().cpu() # gradient 추적 중단 후 cpu 변환
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        if tensor.dim() == 4: ## [B, C, H, W] -> [B, H, W, C]
            tensor = tensor.permute(0, 2, 3, 1)
        elif tensor.dim() == 3: ## [C, H, W] -> [H, W, C]
            tensor = tensor.permute(1, 2, 0)
        else:
            raise ValueError(f"tensor dim expected 3 or 4, got {tensor.dim()} (shape={tuple(tensor.shape)})")

        return tensor

class CelebAHQDataset(Dataset):
    """ CelebA-HQ Dataset을 불러옴 """
    def __init__(self, image_paths, transform=None, phase='train'):
        self.image_paths = image_paths
        self.transform   = transform
        self.phase       = phase
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image, self.phase)
        
        return image

def create_loader(data_dir='./data/CelebA-HQ', image_size=128, batch_size=64, valid_ratio=0.3, seed=42):
    """ 데이터로더 생성 """
    # 1. 이미지 경로 리스트 불러오기
    data_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    assert len(data_paths), f"{data_dir}에 데이터가 존재하지 않습니다."
    ## 데이터 경로 분할
    random.Random(seed).shuffle(data_paths)
    train_num = int(len(data_paths) * (1 - valid_ratio))
    train_paths = data_paths[:train_num]
    valid_paths = data_paths[train_num:]
    
    # 2. 데이터셋 생성
    image_transform = ImageTransform(image_size)
    train_dataset = CelebAHQDataset(train_paths, transform=image_transform, phase='train')
    valid_dataset = CelebAHQDataset(valid_paths, transform=image_transform, phase='valid')
    
    # 3. 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

def plot_images(images):
    """ 데이터셋 이미지 시각화 """
    denorm = Denormalization()
    figure = plt.figure(figsize=(16, 24))
    
    for idx in range(24):
        figure.add_subplot(4, 6, idx + 1)
        image = denorm(images[idx])
        plt.imshow(image)
        plt.axis('off')
    
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)