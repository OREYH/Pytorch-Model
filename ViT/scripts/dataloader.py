""" 데이터로더 생성 """

#### 라이브러리 호출 ####
import os, math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class ImageTransform():
    """ 이미지 전처리 """
    def __init__(self):
        base_transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: (tensor - 0.5) / 0.5) ]
        
        self.data_transform = {
            'train': transforms.Compose([transforms.RandomHorizontalFlip()] + base_transform),
            'valid': transforms.Compose(base_transform)}
    
    def __call__(self, image, phase):
        tensor = self.data_transform[phase](image)
        return tensor

class Denormalization(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor):
        tensor = tensor.detach().cpu()
        tensor = 0.5 * tensor + 0.5
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        if tensor.dim() == 4: # [B, C, H, W]
            tensor = tensor.permute(0, 2, 3, 1) # [B, H, W, C]
        else: # [C, H, W]
            tensor = tensor.permute(1, 2, 0) # [H, W, C]
        
        return tensor

def create_loader(data_path='./data', valid_ratio=0.5, batch_size=128):
    """ 데이터로더 생성 """
    # 0. 경로 생성
    os.makedirs(data_path, exist_ok=True)
    
    # 1. 데이터셋 생성
    image_transform = ImageTransform()
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True,
                        transform=lambda img: image_transform(img, 'train'))
    other_dataset = datasets.CIFAR10(root=data_path, train=False, download=True,
                        transform=lambda img: image_transform(img, 'valid'))
    
    valid_dataset, test_dataset = random_split(other_dataset, [valid_ratio, 1-valid_ratio])
    
    class_to_idx = train_dataset.class_to_idx
    
    print("  train, valid, test datasets 데이터 개수  ".center(60, '='))
    print(len(train_dataset), len(valid_dataset), len(test_dataset), sep=', ')
    
    # 2. DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader, class_to_idx

def plot_images(images, labels, class_to_idx):
    """ 이미지 확인 """
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    denorm = Denormalization()
    figure = plt.figure(figsize=(16, 24))
    
    for idx in range(24):
        figure.add_subplot(4, 6, idx + 1)
        image = denorm(images[idx])
        plt.imshow(image)
        plt.title(idx_to_class[labels[idx].item()])
        plt.axis('off')
    
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)

def plot_test_images(images, labels, preds, class_to_idx, num_samples=25):
    """ test images로 test 결과 시각화 """
    images = images.cpu()
    labels = labels.cpu()
    preds  = preds.cpu()
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    # 이미지 index 뽑기
    num_samples = min(num_samples, len(images))
    img_indices = torch.randperm(len(images))[:num_samples]
    
    denorm = Denormalization()
    figure = plt.figure(figsize=(24, 28))
    rows = math.ceil(math.sqrt(num_samples))
    cols = math.ceil(num_samples / rows)
    
    for idx, img_idx in enumerate(img_indices):
        figure.add_subplot(rows, cols, idx + 1)
        
        image = denorm(images[img_idx])
        label = labels[img_idx].item()
        pred  = preds[img_idx].item()
        color = 'blue' if label == pred else 'red'
        title = f"label: {idx_to_class[label]} | pred: {idx_to_class[pred]}"
        
        plt.imshow(image)
        plt.title(title, color=color, fontsize=12)
        plt.axis('off')
    
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)

if __name__ == '__main__':
    data_path = './data'
    image_transform = ImageTransform()
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True,
                        transform=lambda img: image_transform(img, 'train'))
    other_dataset = datasets.CIFAR10(root=data_path, train=False, download=True,
                        transform=lambda img: image_transform(img, 'valid'))
    
    print(train_dataset.class_to_idx)
