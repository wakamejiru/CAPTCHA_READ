import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import decode_prediction, CAPTCHA_LENGTH

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_chars = 0
    total_chars = 0
    correct_captchas = 0
    total_captchas = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device) # (Batch, Length)
        
        optimizer.zero_grad()
        outputs = model(images) # (Batch, Length, Classes)
        
        # Calculate loss
        # Flatten outputs and labels to use CrossEntropyLoss
        # outputs: (Batch * Length, Classes)
        # labels: (Batch * Length)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Accuracy Calculation
        _, preds = torch.max(outputs, 2) # (Batch, Length)
        
        # Per-character accuracy
        correct_chars += (preds == labels).sum().item()
        total_chars += labels.numel()
        
        # Whole-captcha accuracy
        # (preds == labels).all(dim=1) -> Boolean tensor of shape (Batch,)
        correct_captchas += (preds == labels).all(dim=1).sum().item()
        total_captchas += images.size(0)
        
        pbar.set_postfix({'loss': loss.item()})
        
    epoch_loss = running_loss / total_captchas
    char_acc = correct_chars / total_chars
    captcha_acc = correct_captchas / total_captchas
    
    return epoch_loss, char_acc, captcha_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_chars = 0
    total_chars = 0
    correct_captchas = 0
    total_captchas = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 2)
            correct_chars += (preds == labels).sum().item()
            total_chars += labels.numel()
            correct_captchas += (preds == labels).all(dim=1).sum().item()
            total_captchas += images.size(0)
            
    avg_loss = running_loss / total_captchas
    char_acc = correct_chars / total_chars
    captcha_acc = correct_captchas / total_captchas
    
    return avg_loss, char_acc, captcha_acc
