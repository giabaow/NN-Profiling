import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10, layers=3):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(nn.ReLU())
            input_dim = hidden_dim
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim*mlp_ratio, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: seq_len x batch x embed_dim
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x