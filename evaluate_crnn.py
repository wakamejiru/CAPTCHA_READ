import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CaptchaDataset, CHARSET
from model_crnn import CRNN  # 新しいモデル定義を読み込む
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 設定 ---
MODEL_PATH = "models/best_model_crnn.pth" # 新しいモデルのパス
DATA_DIR = "./test"  # テストデータの場所
METRICS_FILE = "accuracy_metrics_crnn.json"
CONFUSION_MATRIX_FILE = "confusion_matrix_crnn.png"
IMG_H = 32
HIDDEN_SIZE = 512

def decode_greedy(preds, charset):
    """
    CRNNの出力 (Time, Batch, Class) をテキストに変換する
    """
    # (W, B, C) -> (B, W, C) -> argmax -> (B, W)
    preds = preds.permute(1, 0, 2)
    _, preds_index = preds.max(2)
    preds_index = preds_index.cpu().detach().numpy()
    
    decoded_text = []
    for i in range(preds_index.shape[0]):
        text = ""
        prev_char = -1
        for char_idx in preds_index[i]:
            # 0はBlank、かつ連続する同じ文字は無視するのがCTCのルール
            if char_idx != 0 and char_idx != prev_char:
                # index 1 が charset[0] に対応 (0はblank)
                if 0 <= char_idx - 1 < len(charset):
                    text += charset[char_idx - 1]
            prev_char = char_idx
        decoded_text.append(text)
    return decoded_text

def evaluate():
    print(f"Loading CRNN model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの準備 (学習時と同じパラメータで初期化)
    n_class = len(CHARSET) + 1 # +1 for blank
    model = CRNN(IMG_H, 3, n_class, HIDDEN_SIZE).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    # 重みのロード
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Load Error: {e}")
        print("モデル構造と重みファイルが一致していません。train_crnn.pyで学習したモデルを使用してください。")
        return

    model.eval()
    
    # データ読み込み
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    dataset = CaptchaDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    correct_captchas = 0
    total_captchas = 0
    
    # 詳細分析用のリスト
    results = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # 推論
            preds = model(images) # Output: (Width, Batch, Classes)
            
            # デコード (数値 -> 文字列)
            pred_texts = decode_greedy(preds, CHARSET)
            
            # 正解ラベルのデコード
            labels_np = labels.cpu().numpy()
            true_texts = []
            for i in range(labels_np.shape[0]):
                # datasetの実装に依存しますが、通常はCHARSETのインデックス配列
                t = "".join([CHARSET[idx] for idx in labels_np[i]])
                true_texts.append(t)
            
            # 比較
            for pred_text, true_text in zip(pred_texts, true_texts):
                is_correct = (pred_text == true_text)
                if is_correct:
                    correct_captchas += 1
                
                results.append({
                    "True": true_text,
                    "Predicted": pred_text,
                    "Correct": is_correct
                })
                
            total_captchas += len(images)
            
    # 集計結果
    acc = correct_captchas / total_captchas if total_captchas > 0 else 0
    print(f"\n--- Evaluation Results (CRNN) ---")
    print(f"Total Images: {total_captchas}")
    print(f"Correct Captchas: {correct_captchas}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # JSON保存
    metrics = {
        "model": "CRNN",
        "captcha_accuracy": acc,
        "total_captchas": total_captchas,
        "correct_captchas": correct_captchas
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_FILE}")
    
    # 間違った画像の上位パターンなどをCSV保存
    df = pd.DataFrame(results)
    df.to_csv("evaluation_details.csv", index=False)
    print("Detailed results saved to evaluation_details.csv")

    # 文字単位の混同行列は、文字数が可変であるため単純には描画できませんが、
    # 長さが一致する場合のみの文字正解率などを出すと参考になります
    match_len = df[df['True'].str.len() == df['Predicted'].str.len()]
    print(f"Length Match Count: {len(match_len)} / {len(df)}")

if __name__ == "__main__":
    evaluate()