""" ViT Train / Eval / Test """

#### 라이브러리 호출 ####
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast

def train(model, data_loader, criterion, optimizer, scheduler, scaler, device, max_norm=1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # gradient None으로 초기화
    optimizer.zero_grad(set_to_none=True)
    for images, labels in tqdm(data_loader, total=len(data_loader), 
                               desc='train', position=1, leave=False, colour='#9b59b6'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # FP16 / FP32 자동 섞기
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images) # [B, num_classes]
            loss    = criterion(outputs, labels)
        
        # loss scaling: underflow 방지
        scaler.scale(loss).backward()
        
        # gradient unscale: gradient를 기존 크기로 되돌림
        scaler.unscale_(optimizer)
        
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # 파라미터 업데이트
        scaler.step(optimizer)
        
        # scale 조정: 학습 안정화
        scaler.update()
        
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        
        total_loss    += loss.item() * images.size(0)
        total_correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
        total_samples += images.size(0)
    
    train_loss = total_loss / max(1, total_samples)
    train_acc  = total_correct / max(1, total_samples)
    last_lr    = scheduler.get_last_lr()[0] if scheduler is not None else None
    
    return train_loss, train_acc, last_lr

@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for images, labels in tqdm(data_loader, total=len(data_loader),
                               desc='eval', position=1, leave=False, colour='#3e8ede'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        
        total_loss    += loss.item() * images.size(0)
        total_correct += (torch.argmax(outputs, dim=-1) == labels).sum().item()
        total_samples += images.size(0)
    
    eval_loss = total_loss / max(1, total_samples)
    eval_acc  = total_correct / max(1, total_samples)
    
    return eval_loss, eval_acc


@torch.no_grad()
def test(model, images, labels, device):
    """ 테스트 루프 """
    model.eval()
    
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    with autocast(device_type=device.type, enabled=device.type == 'cuda'):
        outputs = model(images)
    
    preds = torch.argmax(outputs, dim=-1).cpu()
        
    return preds
