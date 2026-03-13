import os
import numpy as np
import pandas as pd
import pywt
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def data_load(file_path):
    """读取指定路径的 CSV 文件并提取 CH17 列数据"""
    df = pd.read_csv(file_path, usecols=['CH17'])
    return df.iloc[:, 0].values


def wav_trans(fs, wavelet_name, data):
    """执行连续小波变换 (CWT)"""
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(data, scales, wavelet_name, sampling_period=1 / fs)
    amplitude_scalogram = np.abs(coefficients)
    return frequencies, amplitude_scalogram


def create_samples(data, sample_length=8192, num_samples=100):
    """使用重叠滑动窗口切分一维序列"""
    samples = []
    total_length = len(data)

    if total_length < sample_length:
        raise ValueError(f"数据总长度 ({total_length}) 小于设定的窗口长度 ({sample_length})")

    step_size = (total_length - sample_length) // (num_samples - 1) if num_samples > 1 else 0

    for i in range(num_samples):
        start_idx = i * step_size
        end_idx = start_idx + sample_length
        if end_idx > total_length:
            break
        samples.append(data[start_idx:end_idx])

    return np.array(samples)


def preprocess_pipeline(samples, fs, wavelet_name='morl'):
    """CWT -> Resize -> Z-score 标准化 -> 3通道扩展"""
    processed_images = []
    for seq in samples:
        _, amp = wav_trans(fs, wavelet_name, seq)
        img_resized = cv2.resize(amp, (48, 64), interpolation=cv2.INTER_LINEAR)
        img_normalized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
        img_3_channel = np.stack((img_normalized,) * 3, axis=0)
        processed_images.append(img_3_channel)

    return np.array(processed_images)


def get_dataloaders(base_dir, fs=64000, sample_length=8192, num_samples=100, batch_size=20, tune_size=20):
    """
    核心封装函数：统筹数据读取、预处理、标签构建与 DataLoader 封装

    参数:
        base_dir (str): 数据集根目录，例如 '../../数据集/BJTU/'
        fs (int): 采样频率
        sample_length (int): 样本切片长度
        num_samples (int): 每类提取的样本数
        batch_size (int): 批大小
        tune_size (int): 目标域微调所用的总样本数
    """
    print("🚀 开始加载原始数据并切分样本...")

    # 1. 定义相对路径并加载数据
    paths = {
        'normal': os.path.join(base_dir, '正常/data_leftaxlebox_M0_G0_LA0_RA0_20Hz_-10kN.csv'),
        'inner': os.path.join(base_dir, '内圈/data_leftaxlebox_M0_G0_LA1_RA0_20Hz_-10kN.csv'),
        'outer': os.path.join(base_dir, '外圈/data_leftaxlebox_M0_G0_LA2_RA0_20Hz_-10kN.csv'),
        'rolling': os.path.join(base_dir, '滚动体/data_leftaxlebox_M0_G0_LA3_RA0_20Hz_-10kN.csv'),
        'io': os.path.join(base_dir, '外圈加内圈/data_leftaxlebox_M0_G0_LA1+LA2_RA0_20Hz_-10kN.csv'),
        'or': os.path.join(base_dir, '外圈加滚动体/data_leftaxlebox_M0_G0_LA2+LA3_RA0_20Hz_-10kN.csv')
    }

    samples_dict = {
        key: create_samples(data_load(path), sample_length, num_samples)
        for key, path in paths.items()
    }

    # 2. 执行 CWT 及预处理
    print("⏳ 正在进行连续小波变换 (CWT) 及图像预处理，请稍候...")
    X_dict = {
        key: preprocess_pipeline(samples, fs)
        for key, samples in samples_dict.items()
    }

    # 3. 构建多标签体系
    print("🧮 正在构建特征张量与多标签...")
    y_normal = np.tile([0, 0, 0], (num_samples, 1))
    y_inner = np.tile([1, 0, 0], (num_samples, 1))
    y_outer = np.tile([0, 1, 0], (num_samples, 1))
    y_rolling = np.tile([0, 0, 1], (num_samples, 1))
    y_io = np.tile([1, 1, 0], (num_samples, 1))
    y_or = np.tile([0, 1, 1], (num_samples, 1))

    # 4. 组装源域与目标域
    X_source = np.concatenate((X_dict['normal'], X_dict['inner'], X_dict['outer'], X_dict['rolling']), axis=0)
    y_source = np.concatenate((y_normal, y_inner, y_outer, y_rolling), axis=0)

    X_target = np.concatenate((X_dict['io'], X_dict['or']), axis=0)
    y_target = np.concatenate((y_io, y_or), axis=0)

    # 5. 划分数据集 (Train/Test Split)
    X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(
        X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
    )

    X_tgt_tune, X_tgt_test, y_tgt_tune, y_tgt_test = train_test_split(
        X_target, y_target, train_size=tune_size, random_state=42, stratify=y_target
    )
    # ==========================================
    # 🌟 新增：构建“联合记忆”微调集 🌟
    # ==========================================
    # 从源域训练集中，按比例抽出 20 个单故障样本作为“旧知识记忆”
    X_src_memory, _, y_src_memory, _ = train_test_split(
        X_src_train, y_src_train, train_size=20, random_state=42, stratify=y_src_train
    )

    # 将 20个复合故障(新知识) 和 20个单故障(旧知识) 拼接在一起，总共 40 个样本
    X_joint_tune = np.concatenate((X_tgt_tune, X_src_memory), axis=0)
    y_joint_tune = np.concatenate((y_tgt_tune, y_src_memory), axis=0)
    # 6. 转换为 PyTorch Tensors 并构建 DataLoader
    # 2. 转换为 PyTorch DataLoader
    def to_loader(X, y, is_train=False):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return DataLoader(dataset, batch_size=20, shuffle=is_train)

    loader_src_train = to_loader(X_src_train, y_src_train, is_train=True)
    loader_src_test = to_loader(X_src_test, y_src_test, is_train=False)
    # 🌟 注意这里：使用混合后的 joint 数据集作为微调集
    loader_joint_tune = to_loader(X_joint_tune, y_joint_tune, is_train=True)
    loader_tgt_test = to_loader(X_tgt_test, y_tgt_test, is_train=False)



    print("✅ DataLoader 封装完成！")
    return loader_src_train, loader_src_test, loader_joint_tune, loader_tgt_test


# 模块化测试入口
if __name__ == "__main__":
    BASE_DIR = '../../数据集/BJTU/'
    # 为了快速测试，可以将 num_samples 设小一点
    src_train, src_test, tgt_tune, tgt_test = get_dataloaders(BASE_DIR, num_samples=10)
    print(f"源域训练集 Batch 数量: {len(src_train)}")