""" ViT Classification Script """

#### 라이브러리 호출 ####
import os
import random
import argparse
import torch

from torch import nn, optim, amp
from tqdm.auto import tqdm

from scripts.dataloader import create_loader
from scripts.model import VisionTransformer
from scripts.train import train, evaluate, test


def set_seed(seed):
    """ 모든 랜덤 시드 고정 """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Vision Transformer CIFAR-10 학습')
    parser.add_argument('--data_path', type=str, default='./data', help='CIFAR10 데이터 경로')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='검증 데이터 비율 (0~1)')
    parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=20, help='에폭 수')
    parser.add_argument('--lr', type=float, default=3e-4, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight Decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient Clipping Max Norm')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='모델 저장 경로')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    ## 모델 하이퍼파라미터
    parser.add_argument('--img_size', type=int, default=32, help='입력 이미지 크기')
    parser.add_argument('--patch_size', type=int, default=4, help='패치 크기')
    parser.add_argument('--embed_dim', type=int, default=192, help='패치 임베딩 차원')
    parser.add_argument('--depth', type=int, default=6, help='Encoder 블록 수')
    parser.add_argument('--num_heads', type=int, default=3, help='Multi-Head Attention 헤드 수')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP 확장 비율')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention Dropout')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## DataLoader 생성
    train_loader, valid_loader, test_loader, class_to_idx = create_loader(
        data_path=args.data_path,
        valid_ratio=args.valid_ratio,
        batch_size=args.batch_size
    )
    
    ## 모델/손실/옵티마이저/스케줄러
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=len(class_to_idx),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    ## 학습 루프
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, 'vit_best.pt')
    best_acc = 0.0
    
    for epoch in tqdm(range(1, args.epochs + 1), desc='epochs', total=args.epochs, colour='#BAFF1A'):
        train_loss, train_acc, lr = train(model, train_loader, criterion, optimizer, scheduler, scaler, device, 
                                          max_norm=args.grad_clip)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
              f"valid_loss={valid_loss:.4f} valid_acc={valid_acc*100:.2f}% lr={lr:.6f}")
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'class_to_idx': class_to_idx,
                'args': vars(args)
            }, best_path)
            print(f"  -> 새 베스트 모델 저장: {best_path}")
    
    ## 테스트 (베스트 모델 로드)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc*100:.2f}%")


if __name__ == '__main__':
    main()
