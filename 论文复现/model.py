import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 修改后的 MVD-Net 网络架构
# ==========================================
class MVDNet(nn.Module):
    def __init__(self, input_len=2048, latent_dim=128, K=2):
        super(MVDNet, self).__init__()
        self.K = K
        self.input_len = input_len

        # 1. Encoder: 提取特征 h
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),  # 增加 BN 提升训练稳定性
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten()
        )

        # 【修改点 1】动态推断 encoded_dim，解决硬伤
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_len)
            dummy_output = self.encoder_conv(dummy_input)
            self.encoded_dim = dummy_output.shape[1]
            # 计算卷积层的输出尺寸，用于 Decoder 还原
            self.conv_out_channels = 32
            self.conv_out_len = self.encoded_dim // self.conv_out_channels

        # 2. Parallel FCs (K 个独立的潜空间分支)
        self.fc_mu = nn.ModuleList([nn.Linear(self.encoded_dim, latent_dim) for _ in range(K)])
        self.fc_logvar = nn.ModuleList([nn.Linear(self.encoded_dim, latent_dim) for _ in range(K)])

        # 【修改点 2】独立 Decoders: 由 MLP 改为 ConvTranspose1d，捕获冲击特征
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, self.encoded_dim),
                nn.ReLU(),
                nn.Unflatten(1, (self.conv_out_channels, self.conv_out_len)),
                # 逆卷积层 1
                nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                # 逆卷积层 2 (对应 Encoder 的 kernel 64, stride 8)
                nn.ConvTranspose1d(16, 1, kernel_size=64, stride=8, padding=28, output_padding=7),
            ) for _ in range(K)
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None):
        """
        x: 输入信号 (B, 1, L)
        y: 标签 (B,)。如果是单故障则为故障索引 (0或1)，如果是复合故障则为 -1
        """
        batch_size = x.size(0)
        h = self.encoder_conv(x)

        mus, logvars, x_hats = [], [], []
        for k in range(self.K):
            mu_k = self.fc_mu[k](h)
            logvar_k = self.fc_logvar[k](h)
            z_k = self.reparameterize(mu_k, logvar_k)

            x_k = self.decoders[k](z_k)
            # 确保长度与输入完全一致
            x_hats.append(x_k[:, :, :self.input_len])
            mus.append(mu_k)
            logvars.append(logvar_k)

        # 【修改点 3】Source-Selective Activation (SSA) 权重逻辑
        # 初始化权重：默认为均分 (适用于复合故障)
        weights = torch.full((batch_size, self.K), 1.0 / self.K).to(x.device)

        if y is not None:
            # 根据标签 y 修改权重 w_k (论文公式 14)
            for i in range(batch_size):
                label = y[i].item()
                if label >= 0:  # 单故障样本：只给对应通道权重 1
                    weights[i] = 0.0
                    weights[i, label] = 1.0

        # 加权融合重构：x_fused = Σ (w_k * x_hat_k)
        x_fused = torch.zeros_like(x)
        for k in range(self.K):
            # 将 weights 扩展维度以匹配 (B, 1, L)
            w_k = weights[:, k].view(-1, 1, 1)
            x_fused += w_k * x_hats[k]

        return x_hats, x_fused, mus, logvars

    def get_mask_separation(self, x):
        """
        【修改点 4】新增论文 3.4 节的 Mask 分离推理接口
        """
        self.eval()
        with torch.no_grad():
            x_hats, _, _, _ = self.forward(x)

            # 计算 Soft Mask (基于预测波形的绝对值)
            abs_x_hats = [torch.abs(xh) for xh in x_hats]
            sum_abs = torch.stack(abs_x_hats).sum(dim=0) + 1e-8

            separated_signals = []
            for k in range(self.K):
                mask_k = abs_x_hats[k] / sum_abs
                # 最终输出 = Mask * 原始输入
                separated_signals.append(mask_k * x)

        return separated_signals