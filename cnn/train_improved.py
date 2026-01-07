import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast  # 高速化用: 混合精度学習
import torch.backends.cudnn as cudnn
from dataset import CaptchaDataset
from model import CaptchaCNN
from train_utils import validate  # validateはそのまま利用
import os
import json
import time

# --- RTX3080向け 高速化設定 ---
CONFIG = {
    "name": "p8_sgd_fast_amp",
    "lr": 1e-2,
    "batch_size": 128,      # 32 -> 128 に増量 (3080なら余裕なはずです)
    "dropout": 0.5,
    "opt": "sgd",
    "epochs": 50,
    "weight_decay": 1e-4,
    "num_workers": 4,       # 0 -> 4 に変更 (CPU並列読み込み)
    "pin_memory": True      # GPU転送高速化
}

DATA_DIR = "./study"
MODELS_DIR = "models"
RESULTS_FILE = "training_log_fast.json"

# 高速化のために学習ループをここで再定義 (AMP対応)
def train_one_epoch_amp(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct_chars = 0
    total_chars = 0
    correct_captchas = 0
    total_captchas = 0
    
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True) # non_blockingで転送待ちを減らす
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # AMP (Automatic Mixed Precision) Context
        with autocast():
            outputs = model(images)
            # 形状変更: (Batch, Length, Classes) -> (Batch*Length, Classes)
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            labels_flat = labels.view(-1)
            loss = criterion(outputs_flat, labels_flat)
        
        # Scalerで逆伝播 (FP16の勾配消失を防ぐ)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 統計計算 (ここはCPU負荷になるので必要最小限に)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 2)
        
        correct_chars += (preds == labels).sum().item()
        total_chars += labels.numel()
        correct_captchas += (preds == labels).all(dim=1).sum().item()
        total_captchas += images.size(0)
        
    epoch_loss = running_loss / total_captchas
    char_acc = correct_chars / total_chars
    captcha_acc = correct_captchas / total_captchas
    
    return epoch_loss, char_acc, captcha_acc

def run_fast_training():
    print(f"Starting High-Performance Training on RTX 3080")
    print(f"Config: {CONFIG}")
    
    # 0. CUDNN Benchmark有効化 (固定サイズ画像で高速化)
    cudnn.benchmark = True
    
    # 1. Data Prep
    full_dataset = CaptchaDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # num_workers と pin_memory を有効化
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True # Windowsでのオーバーヘッド削減
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=True
    )
    
    # 2. Model Setup
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    model = CaptchaCNN().to(device)
    model.dropout.p = CONFIG['dropout']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                          lr=CONFIG['lr'], 
                          momentum=0.9, 
                          weight_decay=CONFIG['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler() # AMP用スケーラー
    
    best_acc = 0.0
    history = []
    
    start_time = time.time()
    
    # 3. Training Loop
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        
        # AMP Training
        train_loss, train_char_acc, train_captcha_acc = train_one_epoch_amp(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validation (通常通り)
        val_loss, val_char_acc, val_captcha_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_captcha_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} [{epoch_time:.2f}s] - "
              f"Loss: {train_loss:.4f} - TrainAcc: {train_captcha_acc:.4f} - ValAcc: {val_captcha_acc:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_captcha_acc,
            "epoch_time": epoch_time
        })
        
        if val_captcha_acc > best_acc:
            best_acc = val_captcha_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model_fast.pth"))
            
    total_time = time.time() - start_time
    print(f"\nAll finished in {total_time:.1f}s. Best Acc: {best_acc:.4f}")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Windowsのマルチプロセスエラー回避のため、freeze_supportが必要な場合があるが、
    # if __name__ == "__main__": の中であれば通常は大丈夫です。
    run_fast_training()