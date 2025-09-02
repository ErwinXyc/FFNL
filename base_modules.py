import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# GhostModule
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        init_channels = int(oup / ratio)
        new_channels = oup - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True) if relu else nn.Identity(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :x1.shape[1] + x2.shape[1], :, :]

# LightweightSqueeze-and-Excite
class LightweightSE(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super().__init__()
        reduced_chs = max(1, int(in_chs * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_chs, reduced_chs, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_chs, in_chs, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w

# GhostBottleneck
class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.):
        super().__init__()
        self.stride = stride
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=dw_kernel_size//2, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs)
            )
        else:
            self.dw_conv = nn.Identity()
        self.se = LightweightSE(mid_chs, se_ratio) if se_ratio > 0 else nn.Identity()
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        if in_chs == out_chs and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=dw_kernel_size//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, bias=False),
                nn.BatchNorm2d(out_chs)
            )
    def forward(self, x):
        res = x
        x = self.ghost1(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(res)
        return x

# EfficientDepthwiseBlock
class EfficientDepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expansion_ratio
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # Pointwise linear
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if (stride == 1 and in_channels == out_channels) else nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)



# LightweightChannelAttention
class LightweightChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# LightweightSPP
class LightweightSPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        total_channels = in_channels * 3  # in_channels + 2*in_channels
        self.conv = nn.Conv2d(total_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)
    def forward(self, x):
        size = x.size()
        pool1 = F.adaptive_avg_pool2d(x, (1, 1))
        pool2 = F.adaptive_avg_pool2d(x, (2, 2))
        
        pool1 = F.interpolate(pool1, size=(size[2], size[3]), mode='bilinear', align_corners=False)
        pool2 = F.interpolate(pool2, size=(size[2], size[3]), mode='bilinear', align_corners=False)
        
        out = torch.cat([x, pool1, pool2], dim=1)
        out = self.conv(out)
        out = self.relu(self.bn(out))
        return out


# LightweightPatchAttention
class LightweightPatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=8, reduction=8):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(in_channels)
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size, self.patch_size
        pad_h = pad_w = 0
        if H % ph != 0 or W % pw != 0:
            pad_h = (ph - H % ph) % ph
            pad_w = (pw - W % pw) % pw
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        x_patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        nH, nW = x_patches.shape[2], x_patches.shape[3]
        x_patches = x_patches.contiguous().view(B, C, nH * nW, ph, pw)
        x_patches = x_patches.permute(0,2,3,4,1).contiguous()  # (B, N, ph, pw, C)
        x_patches = self.norm(x_patches)
        x_patches = x_patches.permute(0,4,1,2,3).contiguous()  # (B, C, N, ph, pw)
        patch_mean = x_patches.mean(dim=(3,4))  # (B, C, N)
        attn_weights = self.attn(patch_mean.transpose(1,2))  # (B, N, C)
        attn_weights = attn_weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1)  # (B, C, N, 1, 1)
        x_patches = x_patches * attn_weights
        x_patches = x_patches.view(B, C, nH, nW, ph, pw)
        x_patches = x_patches.permute(0,1,2,4,3,5).contiguous()
        x = x_patches.view(B, C, H, W)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H-pad_h, :W-pad_w]
        return x

# LightweightDownsample
class LightweightDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

    
# Tied Block Squeeze and Excitation Layer
class TiedSELayer(nn.Module):
    '''Tied Block Squeeze and Excitation Layer - Enhanced Version'''
    def __init__(self, channel, B=4, reduction=8):
        super(TiedSELayer, self).__init__()
        assert channel % B == 0
        self.B = B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channel = channel//B
        self.norm = nn.LayerNorm(channel)
        self.fc = nn.Sequential(
                nn.Linear(channel, max(2, channel//reduction)),
                nn.SiLU(inplace=True),
                nn.Linear(max(2, channel//reduction), channel),
                nn.Tanh() 
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b*self.B, c//self.B)
        y = self.norm(y)  
        y = self.fc(y).view(b, c, 1, 1)
        return x * y + x * 0.5  

# SEAttention
class SEAttention(nn.Module):
    """标准SE注意力模块 """
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
