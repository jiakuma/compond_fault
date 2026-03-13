import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, cnn_out_channels=128, seq_length=48, d_model=24, nhead=8, num_layers=4, num_classes=3):
        super(TransformerClassifier, self).__init__()

        # 1. Patch Projection: 将 CNN 通道映射为 Transformer 所需的 d_model
        self.projection = nn.Linear(cnn_out_channels, d_model)

        # 2. Class Token 与 位置编码 (Positional Encoding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 长度需要加 1，因为多了一个 cls_token
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length + 1, d_model))

        # 3. Transformer Encoder 块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,  # 论文中指定的隐藏层维度
            dropout=0.1,
            activation='relu',
            batch_first=True  # 设定 Batch 在第一维，方便操作
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 4. 多标签分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Sigmoid()  # 关键：解耦复合故障
        )
    def forward(self, x):
        batch_size, channels, h, w = x.shape

        # 展平空间维度：[Batch, 128, 8, 6] -> [Batch, 128, 48]
        x = x.view(batch_size, channels, h * w)
        # 转置序列：[Batch, 48, 128]
        x = x.transpose(1, 2)
        # 投影到 d_model：[Batch, 48, 24]
        x = self.projection(x)
        # 拼接 Class Token：[Batch, 49, 24]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 加上位置编码
        x = x + self.pos_embedding
        # 通过 Transformer 提取全局相关性
        x = self.transformer(x)
        # 取出 cls_token 的输出作为整张图的全局特征表示
        cls_out = x[:, 0, :]
        # 输出多标签预测概率
        out = self.classifier(cls_out)
        return out