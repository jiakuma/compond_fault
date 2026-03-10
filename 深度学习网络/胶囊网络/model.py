import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 非线性压缩函数 (Squashing)
def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    # 加上 1e-8 防止除以 0 的梯度爆炸问题
    return scale * x / torch.sqrt(squared_norm + 1e-8)


# 2. 基于 Morlet 小波的一维可学习小波卷积层 (WavConv1d)
class WavConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0):
        super(WavConv1d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 生成固定时间序列 t (范围 -1 到 1)
        # 作为一个常数 buffer，它不会在反向传播中被更新
        t = torch.linspace(-1, 1, kernel_size).view(1, 1, kernel_size)
        self.register_buffer('t', t)

        # 修改为：
        # a: 缩放参数，扩大带宽探索范围
        self.a = nn.Parameter(torch.rand(out_channels, 1, 1) * 5.0 + 0.1)

        # w: 中心频率参数，原先乘以 10，现在改为乘以 150，逼迫模型学习高频冲击
        self.w = nn.Parameter(torch.rand(out_channels, 1, 1) * 150.0 + 1.0)

    def forward(self, x):
        # 确保缩放参数 a 不为 0，防止除零错误
        a_safe = torch.clamp(self.a, min=1e-5)

        # 根据公式动态生成小波卷积核权重
        t_scaled = self.t / a_safe
        # wavelet_weights = cos(w*t) * exp(-0.5 * (t/a)^2)
        wavelet_weights = torch.cos(self.w * self.t) * torch.exp(-0.5 * (t_scaled ** 2))

        # 执行 1D 卷积操作
        # wavelet_weights 形状恰好是 (out_channels, in_channels=1, kernel_size)
        out = F.conv1d(x, wavelet_weights, stride=self.stride, padding=self.padding)

        return out


# 3. 一维主胶囊层 (Primary Capsules 1D)
class PrimaryCaps1d(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dim, kernel_size, stride):
        super(PrimaryCaps1d, self).__init__()
        self.capsule_dim = capsule_dim
        # 使用一维卷积提取特征，输出通道数 = 胶囊数量 * 胶囊维度
        self.conv = nn.Conv1d(in_channels, out_channels * capsule_dim,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # x shape: (batch_size, in_channels, sequence_length)
        out = self.conv(x)
        # out shape: (batch_size, out_channels * capsule_dim, new_seq_length)

        # 将空间维度和通道维度合并，重塑为胶囊的形式
        batch_size = out.size(0)
        out = out.view(batch_size, -1, self.capsule_dim)
        # out shape: (batch_size, num_capsules, capsule_dim)

        return squash(out)


# 4. 分类胶囊层与动态路由 (Class Capsules with Dynamic Routing)
class ClassCaps(nn.Module):
    def __init__(self, num_capsules, in_dim, num_classes, out_dim, num_routing=3):
        super(ClassCaps, self).__init__()
        self.num_routing = num_routing
        # 变换矩阵 W: 形状为 (1, 底层胶囊数, 类别数, 输出维度, 输入维度)
        self.W = nn.Parameter(torch.randn(1, num_capsules, num_classes, out_dim, in_dim) * 0.01)

    def forward(self, x):
        # x shape: (batch_size, num_capsules, in_dim)
        batch_size = x.size(0)

        # 为了矩阵乘法扩充维度
        # x_expanded shape: (batch_size, num_capsules, 1, in_dim, 1)
        x_expanded = x.unsqueeze(2).unsqueeze(4)

        # W_expanded shape: (batch_size, num_capsules, num_classes, out_dim, in_dim)
        W_expanded = self.W.repeat(batch_size, 1, 1, 1, 1)

        # 计算预测向量 u_hat: W * x
        # u_hat shape: (batch_size, num_capsules, num_classes, out_dim, 1)
        u_hat = torch.matmul(W_expanded, x_expanded)
        # 压缩掉最后多余的维度 -> (batch_size, num_capsules, num_classes, out_dim)
        u_hat = u_hat.squeeze(-1)

        # --- 动态路由开始 ---
        # 初始化先验概率 b_ij 为 0, shape: (batch_size, num_capsules, num_classes, 1)
        b_ij = torch.zeros(batch_size, u_hat.size(1), u_hat.size(2), 1).to(x.device)

        for i in range(self.num_routing):
            # 1. 计算路由权重 c_ij
            c_ij = F.softmax(b_ij, dim=2)

            # 2. 聚合高层输入向量 s_j
            # s_j shape: (batch_size, 1, num_classes, out_dim)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # 3. Squashing 激活得到 v_j
            v_j = squash(s_j, dim=-1)

            # 4. 更新 b_ij (最后一次迭代无需更新)
            if i < self.num_routing - 1:
                # u_hat 和 v_j 点积
                agreement = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + agreement

        # 返回最终每个类别的胶囊向量, shape: (batch_size, num_classes, out_dim)
        # 同时也返回动态路由权重 c_ij，这对于后续论文中做可解释性分析（Backward Tracking）至关重要
        return v_j.squeeze(1), c_ij.squeeze(-1)


# 5. 完整的 WavCapsNet 模型
# 5. 完整的 WavCapsNet 模型
class WavCapsNet(nn.Module):
    def __init__(self):
        super(WavCapsNet, self).__init__()
        # 第一层：小波卷积层
        # 采用了优化策略：感受野 512，通道数 128
        self.wav_conv = WavConv1d(out_channels=128, kernel_size=512, stride=16)

        # 第二层：主胶囊层
        # 【修改点 1】：in_channels 必须与上一层的 out_channels (128) 保持一致
        self.primary_caps = PrimaryCaps1d(in_channels=128, out_channels=32,
                                          capsule_dim=8, kernel_size=16, stride=2)

        # --- 张量维度与胶囊数量重新推算 ---
        # 1. WavConv1d 输出长度 = (1024 - 512) // 16 + 1 = 33
        # 2. PrimaryCaps 输出长度 = (33 - 16) // 2 + 1 = 9
        # 3. num_capsules = 9 (序列长度) * 32 (通道数) = 288

        # 第三层：分类胶囊层
        # 【修改点 2】：num_capsules 改为 288，同时确认 num_classes 是 4（加入了正常数据）
        self.class_caps = ClassCaps(num_capsules=288, in_dim=8, num_classes=4, out_dim=16)

    def forward(self, x):
        # 物理特征提取
        x = self.wav_conv(x)

        # 胶囊封装
        x = self.primary_caps(x)

        # 动态路由分类
        v_j, c_ij = self.class_caps(x)

        # 计算分类概率
        probs = torch.norm(v_j, dim=-1)

        return probs, c_ij


# 6. 边缘损失函数 (Margin Loss)
class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, v_mag, labels):
        # v_mag: (batch_size, num_classes), 胶囊的输出模长
        # labels: (batch_size,), 真实的类别索引

        T_k = F.one_hot(labels, num_classes=v_mag.size(1)).float()

        L_present = T_k * F.relu(self.m_plus - v_mag) ** 2
        L_absent = (1 - T_k) * F.relu(v_mag - self.m_minus) ** 2

        loss = L_present + self.lambda_ * L_absent
        return loss.sum(dim=1).mean()