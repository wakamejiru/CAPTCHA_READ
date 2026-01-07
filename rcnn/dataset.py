import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# 文字セット定義
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {char: idx for idx, char in enumerate(CHARSET)}

def encode_label(label):
    # 安全策: CHARSETにある文字だけを変換
    return [CHAR2IDX[c] for c in label if c in CHAR2IDX]

def decode_prediction(preds):
    return "".join([CHARSET[p] for p in preds])

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        
        # 1. 画像ファイルのリストを取得（JPGとPNG両方に対応）
        valid_extensions = ('.jpg', '.jpeg', '.png')
        self.image_files = [f for f in os.listdir(data_dir) 
                            if f.lower().endswith(valid_extensions)]
        
        # 2. ラベルの取得
        self.labels = []
        for f in self.image_files:
            # 拡張子を除去 ("abc_123.jpg" -> "abc")
            filename_without_ext = os.path.splitext(f)[0]
            # "_" で分割してラベル部分を取得
            label_part = filename_without_ext.split('_')[0]
            self.labels.append(label_part)
        
        # ファイルが見つからない場合の警告
        if len(self.image_files) == 0:
            print(f"Warning: No image files (jpg/png) found in {data_dir}")
        
        # 3. 画像変換設定
        self.img_w = 160 
        self.img_h = 32
        
        if transform:
            self.transform = transform
        else:
            # データ拡張（回転やノイズ）を削除し、リサイズとTensor変換のみにする
            self.transform = transforms.Compose([
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            # RGB (3チャンネル) に強制変換して読み込む
            # これにより train_crnn.py の nc=3 設定と整合します
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

        label_str = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(encode_label(label_str), dtype=torch.long)
        
        return image, label