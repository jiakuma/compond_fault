import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from model import MVDNet


# %%
# ==========================================
# 1. 数据加载类 (使用你提供的版本)
# ==========================================
class BJTUMVDSpecificLoader(Dataset):
    def __init__(self, base_path, task='TaskA', sample_len=2048):
        self.sample_len = sample_len
        # 根据论文任务映射 BJTU 标签 [cite: 216, 708]
        if task == 'TaskA':
            self.target_labels = ['M0_G0_LA2_RA0', 'M0_G0_LA3_RA0']  # OR, BF
            self.component = 'leftaxlebox'
            self.channel = 'CH17'
        else:
            self.target_labels = ['M0_G5_LA0_RA0', 'M0_G2_LA0_RA0']  # IR, WT
            self.component = 'gearbox'
            self.channel = 'CH11'

        # 1. 加载两类单故障数据
        self.s1_data = self._get_samples_from_structure(base_path, self.target_labels[0])
        self.s2_data = self._get_samples_from_structure(base_path, self.target_labels[1])

        # 验证单故障样本量是否足够
        if len(self.s1_data) < 300 or len(self.s2_data) < 300:
            print(f"警告: 单故障样本不足300个 (S1: {len(self.s1_data)}, S2: {len(self.s2_data)})")

        # 2. 模拟合成混合样本 (Label -1) [cite: 170, 171]
        self.mixed_data = []
        num_mix = 900  # 论文要求合成 900 个混合样本
        for _ in range(num_mix):
            i, j = np.random.randint(0, len(self.s1_data)), np.random.randint(0, len(self.s2_data))
            self.mixed_data.append(self.s1_data[i] + self.s2_data[j])
        self.mixed_data = np.array(self.mixed_data)

        # 3. 汇总并截取论文要求的样本量 [cite: 225]
        # 训练集: 300个单故障1 + 300个单故障2 + 900个混合样本
        self.samples = np.concatenate([self.s1_data[:300], self.s2_data[:300], self.mixed_data])
        self.labels = np.concatenate([np.zeros(300), np.ones(300), np.full(900, -1)])

    def _get_samples_from_structure(self, base_path, label):
        all_segments = []
        label_path = os.path.join(base_path, label)

        # 锁定想要观察的样本地段，比如换成 'Sample_5'
        target_sample = 'Sample_1'

        sample_path = os.path.join(label_path, target_sample)
        if not os.path.exists(sample_path):
            print(f"错误: 找不到 {target_sample}")
            return np.array([])

        # 筛选 20Hz 且对应部件的 CSV
        target_files = [f for f in os.listdir(sample_path)
                        if "20Hz_0kN" in f and self.component in f]

        for file_name in target_files:
            df = pd.read_csv(os.path.join(sample_path, file_name))
            sig = df[self.channel].values

            # Z-score 归一化
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

            # 切分为 2048 点的样本
            for i in range(len(sig) // self.sample_len):
                seg = sig[i * self.sample_len: (i + 1) * self.sample_len]
                all_segments.append(seg[np.newaxis, :])

        return np.array(all_segments, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


# %%
def run_training(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行环境: {device}")

    # 1. 初始化数据集和加载器
    dataset = BJTUMVDSpecificLoader(data_path, task='TaskA')
    # 增加 drop_last=True 防止最后一个 batch 只有一个样本导致 BatchNorm 出错
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    # 2. 初始化模型
    # 注意：现在 MVDNet 内部会自动推断维度
    model = MVDNet(K=2, input_len=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 训练循环
    for epoch in range(100):
        model.train()
        total_epoch_loss = 0
        total_rec_loss = 0
        total_kl_loss = 0

        for x, y in train_loader:
            # 数据移动到设备，x 为 (B, 1, 2048)，y 为 (B,)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 【修改点 1】直接传入 y，利用模型内部的 SSA 机制
            # 返回值：x_hats (列表), x_fused (融合后的重构), mus, logvars
            x_hats, x_fused, mus, logvars = model(x, y)

            # 【修改点 2】损失函数计算
            # 重建损失：针对融合后的信号 x_fused 与 原始信号 x 计算 MSE
            # 使用 'mean' 更好控制学习率，若使用 'sum' 建议调小 alpha/beta
            loss_rec = F.mse_loss(x_fused, x, reduction='mean')

            # KL 散度损失：计算所有通道的 KL 散度之和
            loss_kl = 0
            for k in range(model.K):
                mu_k, lv_k = mus[k], logvars[k]
                # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_k = -0.5 * torch.mean(1 + lv_k - mu_k.pow(2) - lv_k.exp())
                loss_kl += kl_k

            # 总损失平衡 (根据论文建议调整超参数)
            # alpha=0.1 (KL 约束), beta=1.0 (重构强度)
            alpha = 0.1
            loss = 1.0 * loss_rec + alpha * loss_kl

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_rec_loss += loss_rec.item()
            total_kl_loss += loss_kl.item()

        avg_loss = total_epoch_loss / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/100] | Loss: {avg_loss:.4f} | Rec: {total_rec_loss / len(train_loader):.4f} | KL: {total_kl_loss / len(train_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "mvd_net_final.pth")
    print("训练结束。模型已保存为 mvd_net_final.pth")


# %%
# ==========================================
# 启动执行
# ==========================================
if __name__ == "__main__":
    # 使用你的实际数据路径
    DATA_PATH = r"E:\BaiduNetdiskDownload\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\BJTU_RAO_Bogie_Datasets"
    run_training(DATA_PATH)