import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import NUM_CLASSES, CAPTCHA_LENGTH

class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        
        # Feature Extractor (Backbone)
        # Input: (3, 50, 200)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # -> (32, 25, 100)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # -> (64, 12, 50)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # -> (128, 6, 25)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2) # -> (256, 3, 12)
        
        # Fully Connected Layers
        # Flatten size = 256 * 3 * 12 = 9216
        self.fc_dim = 256 * 3 * 12
        self.fc1 = nn.Linear(self.fc_dim, 1024)
        self.dropout = nn.Dropout(0.5)
        
        # Output Heads - one for each character
        self.heads = nn.ModuleList([
            nn.Linear(1024, NUM_CLASSES) for _ in range(CAPTCHA_LENGTH)
        ])

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, self.fc_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Collect outputs from each head
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
            
        # Stack outputs -> (Batch, Length, Classes)
        return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    # Smoke test
    model = CaptchaCNN()
    dummy_input = torch.randn(1, 3, 50, 200)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: (1, 5, 60)
