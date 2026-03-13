import torch
import torch.nn as nn
from cnn import CNNFeatureExtractor
from transformer import TransformerClassifier


class CompoundFaultNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CompoundFaultNet, self).__init__()

        # 实例化组件
        self.cnn_extractor = CNNFeatureExtractor(in_channels=3)
        # 经过 3 次 $2 \times 2$ 最大池化，64x48 -> 8x6，共 48 个 Patch
        self.transformer_classifier = TransformerClassifier(
            cnn_out_channels=128,
            seq_length=48,
            d_model=24,
            nhead=8,
            num_layers=4,
            num_classes=num_classes
        )

    def forward(self, x):
        # 1. 局部特征提取
        cnn_features = self.cnn_extractor(x)

        # 2. 全局特征解耦与分类
        outputs = self.transformer_classifier(cnn_features)

        return outputs


