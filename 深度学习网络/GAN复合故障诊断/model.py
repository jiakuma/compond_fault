import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=1, sequence_length=1024):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        
        # 全连接层
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, sequence_length * output_dim)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
    
    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        
        # 调整形状：(batch_size, output_dim, sequence_length)
        x = x.view(x.size(0), -1, self.sequence_length)
        
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim=1, sequence_length=1024):
        super(Discriminator, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=16, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim, sequence_length)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = torch.sigmoid(self.fc2(x))
        
        return x


class GANFaultDiagnosis(nn.Module):
    def __init__(self, latent_dim=100, input_dim=1, sequence_length=1024, num_classes=4):
        super(GANFaultDiagnosis, self).__init__()
        # 生成器
        self.generator = Generator(latent_dim, input_dim, sequence_length)
        
        # 特征提取器（基于判别器结构）
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=16, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(16, 32, kernel_size=16, stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_dim, sequence_length)
        
        # 特征提取
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def generate(self, z):
        # 生成样本
        return self.generator(z)


class ConditionalGANFaultDiagnosis(nn.Module):
    def __init__(self, latent_dim=100, input_dim=1, sequence_length=1024, num_classes=4):
        super(ConditionalGANFaultDiagnosis, self).__init__()
        self.num_classes = num_classes
        
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, sequence_length * input_dim),
            nn.Tanh()
        )
        
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Conv1d(input_dim + num_classes, 16, kernel_size=16, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(16, 32, kernel_size=16, stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=16, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(16, 32, kernel_size=16, stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_dim, sequence_length)
        return self.classifier(x)
    
    def generate(self, z, labels):
        # z shape: (batch_size, latent_dim)
        # labels shape: (batch_size, num_classes) or (batch_size,)
        
        if labels.dim() == 1:
            labels = F.one_hot(labels, self.num_classes).float()
        
        # 拼接潜在向量和标签
        z = torch.cat([z, labels], dim=1)
        
        # 生成样本
        x = self.generator(z)
        x = x.view(x.size(0), -1, self.generator[6].out_features // self.generator[6].in_features)
        
        return x
    
    def discriminate(self, x, labels):
        # x shape: (batch_size, input_dim, sequence_length)
        # labels shape: (batch_size, num_classes) or (batch_size,)
        
        if labels.dim() == 1:
            labels = F.one_hot(labels, self.num_classes).float()
        
        # 扩展标签维度并拼接到输入
        labels = labels.unsqueeze(2).repeat(1, 1, x.size(2))
        x = torch.cat([x, labels], dim=1)
        
        # 判别
        return self.discriminator(x)