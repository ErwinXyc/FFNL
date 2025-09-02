import torch
import torch.nn as nn
import torch.nn.functional as F
from base_modules import *

class FishFormerLite_Simplified(nn.Module):
    def __init__(self, num_classes=4, dropout=0.2):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 24, 3, padding=1, bias=False), 
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            TiedSELayer(24, B=4, reduction=8)
        )
        
        # Stage I
        self.stage1 = nn.Sequential(
            GhostBottleneck(24, 32, 32, stride=1, se_ratio=0.1),
            EfficientDepthwiseBlock(32, 32, expansion_ratio=4),
            TiedSELayer(32, B=4, reduction=8)
        )
        
        # Stage II
        self.stage2 = nn.Sequential(
            LightweightDownsample(32, 48),
            GhostBottleneck(48, 64, 64, stride=1, se_ratio=0.1),
            EfficientDepthwiseBlock(64, 64, expansion_ratio=4),
            TiedSELayer(64, B=4, reduction=8),
            nn.Conv2d(64, 72, 1, bias=False),  
            nn.BatchNorm2d(72),
            nn.SiLU(inplace=True)
        )
        
        # Shortcut
        self.shortcut12 = nn.Sequential(
            nn.Conv2d(32, 72, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(72)
        )
        
        # Stage III
        self.stage3 = nn.Sequential(
            LightweightDownsample(72, 96),
            nn.Conv2d(96, 96, 3, padding=1, bias=False), 
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            TiedSELayer(96, B=4, reduction=8),
            LightweightPatchAttention(96, patch_size=4),
            LightweightSPP(96, 104)
        )
        
        # Stage IV
        self.stage4 = nn.Sequential(
            nn.Conv2d(104, 112, 1, bias=False), 
            nn.BatchNorm2d(112),
            nn.SiLU(inplace=True),
            GhostBottleneck(112, 112, 112, stride=1, se_ratio=0.1),
            EfficientDepthwiseBlock(112, 112, expansion_ratio=4),
            TiedSELayer(112, B=4, reduction=8)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(112, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1) + self.shortcut12(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        out = self.global_pool(x4).squeeze(-1).squeeze(-1)
        out = self.dropout(out)
        return self.fc(out)