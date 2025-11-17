""" Transformer 기계학습기 메인 스크립트 """

#### 라이브러리 호출 ####
import os
import random
import torch
import argparse
import tomli
from tqdm.auto import tqdm

from torch import nn, optim, amp

from scripts.dataloader import load_tokenizer, create_loader
from scripts.model import Transformer
from scripts.train import train, evaluate, inference, NoamScheduler
from scripts.notify_ntfy import ntfy_notify # 알림 모듈

def set_seed(seed):
    """ 재현성을 위한 시드 고정 """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer 한-일 기계번역기')
    parser.add_argument('--config', type=str, default='config.toml', help='학습/추론 파라미터 TOML 경로')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], help='훈련/추론 중 하나 선택(train or inference)')
    
    return parser.parse_args()

def load_config(toml_path):
    with open(toml_path, 'rb') as file:
        return tomli.load(file)

@ntfy_notify(title='Transformer 학습 & 합성 데이터 생성', notify_on='both')
def main():
    args = parse_args()
    cfg  = load_config(args.config)
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_path = os.path.join(data_cfg['data_dir'], 'tokenizer', 'sp_kor_jpn.model')
    tokenizer = load_tokenizer(tokenizer_path)
    
    train_path = os.path.join(data_cfg['data_dir'], 'text', 'chat_train.txt')
    valid_path = os.path.join(data_cfg['data_dir'], 'text', 'chat_valid.txt')
    
    train_loader = create_loader(train_path, tokenizer, 
                                 src_len=data_cfg['src_len'], tgt_len=data_cfg['tgt_len'],
                                 batch_size=train_cfg['batch_size'], shuffle=True)
    valid_loader = create_loader(valid_path, tokenizer,
                                 src_len=data_cfg['src_len'], tgt_len=data_cfg['tgt_len'],
                                 batch_size=train_cfg['val_batch_size'], shuffle=False)
    
    model = Transformer(vocab_size=tokenizer.GetPieceSize(), pad_id=tokenizer.pad_id(), 
                        **model_cfg).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id(),
                                    label_smoothing=train_cfg['label_smoothing'])
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9,
                           weight_decay=train_cfg['weight_decay'])
    scheduler = NoamScheduler(optimizer, model_cfg['d_model'], 
                              warmup_steps=train_cfg['warmup_steps'])
    scaler    = amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    if args.mode == 'train':
        # 학습 저장 경로 생성
        os.makedirs(train_cfg['save_dir'], exist_ok=True)
        best_path = os.path.join(train_cfg['save_dir'], 'best.pt')
        best_valid = float('inf')
        
        for epoch in tqdm(range(1, train_cfg['epochs'] + 1), desc='train', total=train_cfg['epochs'], colour='#BAFF1A'):
            train_loss, train_ppl, lr = train(model, train_loader, criterion, optimizer, scheduler, scaler, device, 
                                              accumulation_steps=train_cfg['accumulation_steps'], 
                                              max_norm=train_cfg['grad_clip'], pad_id=tokenizer.pad_id())
            valid_loss, valid_ppl = evaluate(model, valid_loader, criterion, device, pad_id=tokenizer.pad_id())
            
            print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
                  f"valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.2f} lr={lr:.6f}")
            
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': {'step': scheduler.step_num, 'lr': scheduler.current_lr},
                    'scaler_state': scaler.state_dict(),
                    'config': cfg }, best_path)
                print(f"  -> 새로운 최고 모델 저장: {best_path}")
    
    if args.mode == 'inference':
        best_path = os.path.join(train_cfg['save_dir'], 'best.pt')
        ckpt = torch.load(best_path, map_location=device)
        cfg = ckpt['config']
        model.load_state_dict(ckpt['model_state'])
        
        translation = inference(model, tokenizer, infer_cfg['sample_text'], device, max_len=infer_cfg['max_infer_len'])
        print(f"입력: {infer_cfg['sample_text']}")
        print(f"번역: {translation}")

if __name__ == '__main__':
    main()