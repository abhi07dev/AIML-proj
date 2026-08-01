"""
Model architecture for the deepfake detector.

This is the exact architecture from Deep_Fake_detection_final.ipynb:
a dual-stream network that combines
  1) a spatial stream (EfficientNet-B4 + attention) that looks at pixel-level
     manipulation artifacts, and
  2) a frequency stream (a small CNN over the FFT magnitude spectrum) that
     looks for GAN spectral artifacts invisible to the human eye.

IMPORTANT: train.py and app.py both import DeepfakeDetector from here, so a
checkpoint produced by train.py will always load correctly in app.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AttentionBlock(nn.Module):
    """Spatial attention to focus on manipulated regions."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)

    def forward(self, x):
        attn = F.relu(self.conv1(x))
        attn = torch.sigmoid(self.conv2(attn))
        return x * attn


class FrequencyBranch(nn.Module):
    """
    FFT frequency branch - detects GAN spectral artifacts
    invisible to the human eye but characteristic of generated images.
    """

    def __init__(self, out_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 16, out_features), nn.ReLU(True), nn.Dropout(0.3),
        )

    def forward(self, x):
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        fft = torch.fft.fftshift(torch.fft.fft2(gray))
        mag = torch.log(torch.abs(fft) + 1e-8).expand(-1, 3, -1, -1)
        return self.net(mag)


class DeepfakeDetector(nn.Module):
    """
    Dual-stream EfficientNet-B4 deepfake detector.
    Stream 1: Spatial (pixel manipulation) | Stream 2: Frequency (GAN artifacts)
    """

    def __init__(self, num_classes=2, dropout=0.4, pretrained=True, freeze_layers=3):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)
        self.spatial_backbone = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_layers > 0:
            for layer in list(self.spatial_backbone.children())[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.attention = AttentionBlock(1792)
        self.freq = FrequencyBranch(256)
        self.classifier = nn.Sequential(
            nn.Linear(1792 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        fm = self.attention(self.spatial_backbone(x))
        spat = self.spatial_pool(fm).flatten(1)
        freq = self.freq(x)
        return self.classifier(torch.cat([spat, freq], 1))
