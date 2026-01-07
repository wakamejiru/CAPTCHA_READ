import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CaptchaDataset, decode_prediction, CHARSET
from model import CaptchaCNN
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json  # 追加

MODEL_PATH = "../models/best_model_fast.pth"
DATA_DIR = "../test"
CONFUSION_MATRIX_FILE = "../confusion_matrix.png"
CONFUSION_MATRIX_CSV = "../confusion_matrix.csv"
METRICS_FILE = "../accuracy_metrics.json"  # 追加: 結果保存用ファイル名

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CaptchaCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Load Data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    dataset = CaptchaDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    correct_chars = 0
    total_chars = 0
    correct_captchas = 0
    total_captchas = 0
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device) # (Batch, Length)
            
            outputs = model(images) # (Batch, Length, Classes)
            _, preds = torch.max(outputs, 2)
            
            # Flatten for char-level verification
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
            
            # Accuracy stats
            correct_chars += (preds == labels).sum().item()
            total_chars += labels.numel()
            
            # Check if entire captcha is correct (all characters in the sequence match)
            # preds: (Batch, Length), labels: (Batch, Length)
            row_match = (preds == labels).all(dim=1)
            correct_captchas += row_match.sum().item()
            total_captchas += images.size(0)
            
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    captcha_acc = correct_captchas / total_captchas if total_captchas > 0 else 0
    
    print(f"\n--- Evaluation Results ---")
    print(f"Character Accuracy: {char_acc:.4f}")
    print(f"Captcha Accuracy (All-or-Nothing): {captcha_acc:.4f}")
    
    # --- 追加: 結果をJSONに保存 ---
    metrics = {
        "character_accuracy": char_acc,
        "captcha_accuracy": captcha_acc,
        "total_captchas": total_captchas,
        "correct_captchas": correct_captchas,
        "total_chars": total_chars,
        "correct_chars": correct_chars
    }
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_FILE}")
    # ---------------------------

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CHARSET)))
    
    # Save as CSV
    cm_df = pd.DataFrame(cm, index=list(CHARSET), columns=list(CHARSET))
    cm_df.to_csv(CONFUSION_MATRIX_CSV)
    print(f"Confusion Matrix saved to {CONFUSION_MATRIX_CSV}")
    
    # Plot
    try:
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm_df, annot=False, cmap='Blues')
        plt.title("Character Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(CONFUSION_MATRIX_FILE)
        print(f"Confusion Matrix plot saved to {CONFUSION_MATRIX_FILE}")
    except Exception as e:
        print(f"Could not save plot (seaborn might not be working): {e}")

if __name__ == "__main__":
    evaluate()