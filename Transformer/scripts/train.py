#### 라이브러리 호출 ####
import math
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast

class NoamScheduler():
    """ Noam Warmup Scheduler """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.factor       = factor
        self.step_num     = 0
        self.current_lr   = 0.0
    
    def step(self):
        self.step_num += 1
        args1 = self.step_num ** -0.5
        args2 = self.step_num * (self.warmup_steps ** -1.5)
        lr = self.factor * (self.d_model ** -0.5) * min(args1, args2)
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return self.current_lr

def train(model, data_loader, criterion, optimizer, scheduler, scaler, device,
          accumulation_steps=1, max_norm=1.0, pad_id=3):
    """ 모델 학습 """
    model.train()
    total_loss = 0.0
    total_tokens = 0 # pad_id를 제외한 실제 token 개수
    
    # gradient를 0 대신 아예 None으로 만들기
    # 속도에 이점이 있음
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in tqdm(enumerate(data_loader, start=1), desc='train', total=len(data_loader), leave=False, position=1, colour='#9b59b6'):
        # 1. 데이터 불러오기
        # non_blocking=True -> 속도 개선
        src        = batch.src_ids.to(device, non_blocking=True) 
        tgt_in     = batch.tgt_in_ids.to(device, non_blocking=True)
        tgt_out    = batch.tgt_out_ids.to(device, non_blocking=True)
        src_pad    = batch.src_pad_mask.to(device, non_blocking=True)
        tgt_pad    = batch.tgt_pad_mask.to(device, non_blocking=True)
        num_tokens = tgt_out.ne(pad_id).sum().item()
        
        # 2. 모델 연산
        # autocast: FP16 / FP32 자동 섞기
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            # logits: [B, tgt_len, vocab_size]
            # tgt_out: [B, tgt_len]
            logits = model(src, tgt_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            # logits: [B * tgt_len, vocab_size] / tgt_out: [B * tgt_len]
            batch_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)) 
        
        # 3. gradient 계산
        loss = batch_loss / accumulation_steps
        scaler.scale(loss).backward() # loss scaling
        
        # 4. 파라미터 업데이트
        update_step = batch_idx % accumulation_steps == 0 or batch_idx == len(data_loader)
        if update_step:
            # loss scaling으로 인해 scaling된 gradient를 원래 크기로 복원
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer) # 파라미터 업데이트
            scaler.update()        # scale 조정
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss   += batch_loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    
    return avg_loss, ppl, scheduler.get_last_lr()

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, pad_id=3):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in tqdm(data_loader, desc='eval', leave=False, total=len(data_loader), position=1, colour='#3e8ede'):
        src        = batch.src_ids.to(device, non_blocking=True)
        tgt_in     = batch.tgt_in_ids.to(device, non_blocking=True)
        tgt_out    = batch.tgt_out_ids.to(device, non_blocking=True)
        src_pad    = batch.src_pad_mask.to(device, non_blocking=True)
        tgt_pad    = batch.tgt_pad_mask.to(device, non_blocking=True)
        num_tokens = tgt_out.ne(pad_id).sum().item()
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(src, tgt_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            batch_loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        
        total_loss   += batch_loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    
    return avg_loss, ppl

@torch.no_grad()
def inference(model, tokenizer, sentence, device, max_len=64):
    """ Sentence를 넣어 Decoding하는 함수 """
    # Auto-Regressive Decoder
    model.eval()
    src_ids = torch.tensor([tokenizer.EncodeAsIds(sentence)], dtype=torch.long, device=device) # [B, src_len]
    src_pad = src_ids.eq(tokenizer.pad_id()) # [B, src_len]
    tgt_ids = torch.tensor([[tokenizer.bos_id()]], dtype=torch.long, device=device) # [B, 1]
    
    for _ in range(max_len):
        tgt_pad    = tgt_ids.eq(tokenizer.pad_id())
        logits     = model(src_ids, tgt_ids, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        # tgt_ids: [B, current_len]
        tgt_ids    = torch.cat([tgt_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
        if next_token == tokenizer.eos_id():
            break
    
    out_ids = tgt_ids[0, 1:].tolist()
    ## eos_id 제외
    if tokenizer.eos_id() in out_ids:
        out_ids = out_ids[:out_ids.index(tokenizer.eos_id())]

    return tokenizer.DecodeIds(out_ids)