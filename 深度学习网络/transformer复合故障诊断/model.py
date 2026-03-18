import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # 多头注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # 自注意力机制
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerFaultDiagnosis(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1, num_classes=4):
        super(TransformerFaultDiagnosis, self).__init__()
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        batch_size = x.size(0)
        
        # 调整形状：(batch_size, sequence_length, 1)
        x = x.permute(0, 2, 1)
        
        # 嵌入层
        x = self.embedding(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 调整形状为Transformer所需：(sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 调整形状：(batch_size, d_model, sequence_length)
        x = x.permute(1, 2, 0)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class WaveletTransformerFaultDiagnosis(nn.Module):
    def __init__(self, num_classes=4):
        super(WaveletTransformerFaultDiagnosis, self).__init__()
        # 小波卷积层
        self.wavelet_conv = nn.Conv1d(1, 32, kernel_size=16, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 特征降维
        self.dim_reduction = nn.Linear(32, 64)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(64)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(64, 4, 128, 0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 2)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        batch_size = x.size(0)
        
        # 小波卷积
        x = self.wavelet_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 调整形状：(batch_size, sequence_length, 32)
        x = x.permute(0, 2, 1)
        
        # 特征降维
        x = self.dim_reduction(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 调整形状为Transformer所需：(sequence_length, batch_size, 64)
        x = x.permute(1, 0, 2)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 调整形状：(batch_size, 64, sequence_length)
        x = x.permute(1, 2, 0)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x