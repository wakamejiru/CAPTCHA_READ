import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CaptchaDataset
from model import CaptchaCNN
from train_utils import train_one_epoch, validate
import os
import json

# Configuration Patterns
PATTERNS = [
    {"name": "p1_base", "lr": 1e-3, "batch_size": 32, "dropout": 0.5, "opt": "adam", "epochs": 10},
    {"name": "p2_lr_low", "lr": 1e-4, "batch_size": 32, "dropout": 0.5, "opt": "adam", "epochs": 10},
    {"name": "p3_lr_mid", "lr": 5e-4, "batch_size": 32, "dropout": 0.5, "opt": "adam", "epochs": 10},
    {"name": "p4_bs_small", "lr": 1e-3, "batch_size": 16, "dropout": 0.5, "opt": "adam", "epochs": 10},
    {"name": "p5_bs_large", "lr": 1e-3, "batch_size": 64, "dropout": 0.5, "opt": "adam", "epochs": 10},
    {"name": "p6_drop_low", "lr": 1e-3, "batch_size": 32, "dropout": 0.3, "opt": "adam", "epochs": 10},
    {"name": "p7_drop_high", "lr": 1e-3, "batch_size": 32, "dropout": 0.7, "opt": "adam", "epochs": 10},
    {"name": "p8_sgd", "lr": 1e-3, "batch_size": 32, "dropout": 0.5, "opt": "sgd", "epochs": 10},
    {"name": "p9_long", "lr": 1e-3, "batch_size": 32, "dropout": 0.5, "opt": "adam", "epochs": 15},
    {"name": "p10_wd", "lr": 1e-3, "batch_size": 32, "dropout": 0.5, "opt": "adam", "epochs": 10, "weight_decay": 1e-4}
]

DATA_DIR = "./study"
MODELS_DIR = "models"
RESULTS_FILE = "experiment_results.json"

def run_experiment(config):
    print(f"Running Experiment: {config['name']}")
    
    # 1. Prepare Data
    full_dataset = CaptchaDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # 2. Setup Model & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaCNN().to(device)
    
    # Adjust dropout if possible
    model.dropout.p = config['dropout']
    
    criterion = nn.CrossEntropyLoss()
    
    if config['opt'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0))
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config.get('weight_decay', 0))
        
    best_acc = 0.0
    
    # 3. Training Loop
    history = []
    for epoch in range(config['epochs']):
        train_loss, train_char_acc, train_captcha_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_char_acc, val_captcha_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_captcha_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_captcha_acc:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_captcha_acc
        })
        
        if val_captcha_acc > best_acc:
            best_acc = val_captcha_acc
            save_path = os.path.join(MODELS_DIR, f"best_model_{config['name']}.pth")
            torch.save(model.state_dict(), save_path)
            
    return {
        "config": config,
        "best_val_acc": best_acc,
        "history": history
    }

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_results = []
    
    for config in PATTERNS:
        result = run_experiment(config)
        all_results.append(result)
        
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("All experiments completed.")
