import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(CNNFeatureExtractor, self).__init__()

        # 严格对应论文表 I 的结构，添加 padding=1 保持尺寸比例
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 输入维度: [Batch, 3, 64, 48]
        x = self.block1(x)  # 输出: [Batch, 512, 32, 24]
        x = self.block2(x)  # 输出: [Batch, 256, 16, 12]
        x = self.block3(x)  # 输出: [Batch, 128, 8, 6]
        return x