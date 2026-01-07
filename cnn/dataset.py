import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# 修正ポイント: 合計が60文字になるように調整します。
# 例: 大文字小文字数字(62文字)から '0' と 'O' (または 'I' と 'l') を除いたものなど。
# ここでは推測に基づき、一般的に使われる60文字セットを定義します。
# ---------------------------------------------------------
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ" 
# ※上記は一例です。数字(10) + 小文字(26) + 大文字(24 = I, O抜き) = 60文字

CAPTCHA_LENGTH = 5  
NUM_CLASSES = len(CHARSET) # ここが 60 になる必要があります

def decode_prediction(preds):
    return "".join([CHARSET[p] for p in preds])

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # 拡張子が画像のものだけを取得
        self.image_filenames = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((50, 200)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

        self.char_to_idx = {char: idx for idx, char in enumerate(CHARSET)}

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # ファイル名からラベルを取得
        label_str = os.path.splitext(img_name)[0]
        
        label = torch.zeros(CAPTCHA_LENGTH, dtype=torch.long)
        for i, char in enumerate(label_str):
            if i < CAPTCHA_LENGTH:
                # 文字がCHARSETにない場合の安全策
                label[i] = self.char_to_idx.get(char, 0)
                
        return image, label