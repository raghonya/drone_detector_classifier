# model.py
import torch.nn as nn

class DroneClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),   # ← kernel_size 5
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),  # ← kernel_size 5
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),                                 # длина 256 → 128 → 64 ⇒ 32*64 = 2048
            nn.Linear(32 * 64, 128),                      # ← hidden 128
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
