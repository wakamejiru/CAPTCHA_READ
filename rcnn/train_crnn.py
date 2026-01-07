import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.backends import cudnn
import os
import json
import time
import numpy as np

# 既存のデータセットを利用
from dataset import CaptchaDataset, CHARSET
# 新しいモデルを利用
from model_crnn import CRNN

# --- 設定 ---
CONFIG = {
    "name": "crnn_ctc_512_dropout",
    "lr": 0.001,
    "batch_size": 128,     
    "img_h": 32,
    "hidden_size": 512,    # 大きなサイズを維持
    "epochs": 100,         # Dropoutがあるため、学習時間を確保 (50 -> 100)
    "num_workers": 8,
    "pin_memory": True,
    "dropout": 0.5         # 50%のニューロンをランダムに無効化して鍛える
}

DATA_DIR = "./study"
MODELS_DIR = "models"
RESULTS_FILE = "training_log_crnn.json"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_one_epoch_ctc(model, dataloader, criterion, optimizer, device, converter_dict):
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        
        # ラベルの準備 (CTC用: blank=0とするため、既存ラベルを+1する)
        # datasetがindex(0~35)を返すと仮定。CTC用に1~36にシフト
        targets = labels.to(device) + 1 
        
        optimizer.zero_grad()
        
        preds = model(images)  # (W, B, C)
        
        # CTC Lossの入力サイズ計算
        preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
        
        # target_lengths計算 (固定長なら定数だが、可変長対応しておく)
        target_lengths = torch.IntTensor([labels.size(1)] * batch_size).to(device) # (B)
        
        # Log Softmax
        log_probs = nn.functional.log_softmax(preds, dim=2)
        
        # Loss計算 (log_probs, targets, input_lengths, target_lengths)
        # targetsは1次元の連結ベクトルか、(B, L)のTensor。CTCLossの仕様に合わせる
        # torch.nn.CTCLoss: targets can be (B, L)
        loss = criterion(log_probs, targets, preds_size, target_lengths)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        
    return running_loss / len(dataloader.dataset)

def decode_greedy(preds, charset):
    # preds: (W, B, C) -> argmax -> (W, B) -> Transpose (B, W)
    preds = preds.permute(1, 0, 2) # [B, W, C]
    _, preds_index = preds.max(2)
    preds_index = preds_index.cpu().detach().numpy()
    
    decoded_text = []
    for i in range(preds_index.shape[0]):
        text = ""
        prev_char = -1
        for char_idx in preds_index[i]:
            # Blank (0) または 重複文字をスキップ
            if char_idx != 0 and char_idx != prev_char:
                # index 1 が charset[0] に対応
                if 0 <= char_idx - 1 < len(charset):
                    text += charset[char_idx - 1]
            prev_char = char_idx
        decoded_text.append(text)
    return decoded_text

def validate_ctc(model, dataloader, criterion, device, charset):
    model.eval()
    running_loss = 0.0
    correct_captchas = 0
    total_captchas = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            targets = labels.to(device) + 1
            
            preds = model(images) # (W, B, C)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
            target_lengths = torch.IntTensor([labels.size(1)] * batch_size).to(device)
            
            log_probs = nn.functional.log_softmax(preds, dim=2)
            loss = criterion(log_probs, targets, preds_size, target_lengths)
            running_loss += loss.item() * batch_size
            
            # デコードして正解率計算
            pred_texts = decode_greedy(preds, charset)
            
            # ラベルをテキストに戻す (datasetの実装に依存するが、ここでは簡易的に)
            label_texts = []
            labels_np = labels.cpu().numpy()
            for i in range(labels_np.shape[0]):
                t = "".join([charset[idx] for idx in labels_np[i]])
                label_texts.append(t)
            
            for pt, lt in zip(pred_texts, label_texts):
                if pt == lt:
                    correct_captchas += 1
            total_captchas += batch_size
            
    return running_loss / total_captchas, correct_captchas / total_captchas

def run_crnn_training():
    print(f"Starting CRNN Training (Blog Replication)")
    
    # Data Prep
    full_dataset = CaptchaDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model: クラス数は CHARSET + Blank(1)
    n_class = len(CHARSET) + 1
    model = CRNN(CONFIG['img_h'], 3, n_class, CONFIG['hidden_size']).to(device)
    model.apply(weights_init) # 初期化
    
    # CTC Loss (blank=0 を指定)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_acc = 0.0
    history = []
    
    for epoch in range(CONFIG['epochs']):
        start = time.time()
        
        train_loss = train_one_epoch_ctc(model, train_loader, criterion, optimizer, device, None)
        val_loss, val_acc = validate_ctc(model, val_loader, criterion, device, CHARSET)
        
        scheduler.step(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} [{elapsed:.1f}s] Loss: {train_loss:.4f} ValAcc: {val_acc:.4f}")
        
        history.append({"epoch": epoch+1, "loss": train_loss, "val_acc": val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model_crnn.pth"))
            
    print(f"Best Accuracy: {best_acc:.4f}")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    cudnn.benchmark = True
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_crnn_training()