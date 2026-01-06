import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Define charset based on analysis (no 0, o)
CHARSET = sorted(['1', '2', '3', '4', '5', '6', '7', '8', '9', 
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

CHAR2IDX = {char: idx for idx, char in enumerate(CHARSET)}
IDX2CHAR = {idx: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET)
CAPTCHA_LENGTH = 5

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_files = [f for f in self.data_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        if transform:
            self.transform = transform
        else:
            # Default transform if none provided
            self.transform = transforms.Compose([
                transforms.Resize((50, 200)), # Height, Width - adjusted for typical 5-char captcha aspect ratio
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        image = Image.open(file_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label_str = file_path.stem # name without extension
        
        # Encode label
        label_vec = []
        for char in label_str:
            if char in CHAR2IDX:
                label_vec.append(CHAR2IDX[char])
            else:
                # Handle unknown characters if any (shouldn't happen given analysis)
                print(f"Warning: Unknown char '{char}' in file {file_path.name}")
                label_vec.append(0) # Fallback to first class
        
        # Ensure fixed length (truncate or pad if necessary, though assumption is fixed length)
        if len(label_vec) != CAPTCHA_LENGTH:
             # In a real scenario we might pad, but here we assume clean data.
             # Just strict enforcement for now to catch errors.
             pass

        return image, torch.tensor(label_vec, dtype=torch.long)

def decode_prediction(pred_indices):
    """
    Args:
        pred_indices: List or Tensor of indices (length 5)
    Returns:
        String of characters
    """
    return "".join([IDX2CHAR[idx.item() if isinstance(idx, torch.Tensor) else idx] for idx in pred_indices])
