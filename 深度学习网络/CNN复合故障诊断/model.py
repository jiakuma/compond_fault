import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFaultDiagnosis(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNFaultDiagnosis, self).__init__()
        # 第一层卷积：输入通道1，输出通道16，卷积核大小16，步长2
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # 第二层卷积：输入通道16，输出通道32，卷积核大小16，步长2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # 第三层卷积：输入通道32，输出通道64，卷积核大小16，步长2
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # 第三层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class CNNFaultDiagnosisWithAttention(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNFaultDiagnosisWithAttention, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # 卷积特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x