import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import CaptchaDataset
from model import CaptchaCNN
from train_utils import train_one_epoch
import random

def dry_run():
    print("Starting Dry Run...")
    
    # 1. Prepare Data (Small subset)
    full_dataset = CaptchaDataset("./study")
    if len(full_dataset) == 0:
        print("Error: No images found in ./study")
        return

    subset_indices = random.sample(range(len(full_dataset)), min(100, len(full_dataset)))
    train_dataset = Subset(full_dataset, subset_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CaptchaCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Run 1 Epoch
    print("Running training for 1 epoch...")
    loss, char_acc, captcha_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    print(f"Dry Run Result: Loss={loss:.4f}, CharAcc={char_acc:.4f}, CaptchaAcc={captcha_acc:.4f}")
    if loss > 0:
        print("Dry run successful!")
    else:
        print("Dry run produced 0 loss, which is suspicious unless perfectly overfit instantly.")

if __name__ == "__main__":
    dry_run()
