# 🚀 Comprehensive Intelligent Fault Diagnosis Toolkit
> 工业机械设备智能故障诊断与高级信号处理算法库

## 📖 项目简介 (Overview)
本项目是一个涵盖了从**传统经典信号处理算法**到**前沿深度学习模型**的综合性机械故障诊断算法库。
项目基于 Python 实现了针对旋转机械（滚动轴承、齿轮箱）的多源异构信号解耦、微弱故障特征提取以及自动化诊断流程。支持 BJTU（北京交通大学）和 CWRU（凯斯西储大学）等主流开源数据集的批量化处理与跨域验证。

## ✨ 核心特性 (Key Features)

* 🛠️ **经典高级信号处理 (Advanced Signal Processing):**
  * 实现了基于**快速谱峭度 (Fast Kurtogram)** 的最优共振频带自动搜索。
  * 集成**预白化 (Pre-whitening)** 与**包络解析 (Envelope Analysis)**，有效提取强噪声背景下的早期微弱故障周期性冲击特征。
  * 提供连续小波变换 (CWT) 时频特征提取模块，为 2D 图像网络提供高分辨率 Scalogram 输入。
* 🔬 **前沿论文复现与深度学习 (Paper Reproduction & Deep Learning):**
  * 包含针对复合故障的**单通道盲源分离 (Blind Source Separation, BSS)** 算法研究。
  * 结合 **VMD (变分模态分解)** 与自定义深度网络 (如 `mvd_net_final.pth`) 实现复杂耦合信号的精准剥离。
* 🏭 **工业级数据流水线 (Industrial Data Pipeline):**
  * 针对 BJTU 数据集设计了全自动化的分析与绘图脚本 (`_批量处理.ipynb`)。
  * 支持变转速 (20Hz/40Hz/60Hz)、变载荷 (-10kN/0/+10kN) 多工况下的批量特征图生成与对比，轻松应对海量高频工业数据。
* 📚 **完善的中文教程与文档 (Detailed Tutorials):**
  * 提供从理论到代码落地的《滚动轴承诊断教程.md》及各步骤详解，适合二次开发与学术研究参考。

## 📁 核心目录结构 (Repository Structure)

```text
├── 理论教程/
│   ├── 滚动轴承诊断教程.md             # 基础理论与流程指南
│   └── 滚动轴承诊断教程各步骤详解.md      # 算法核心公式与实现细节
├── 包络解析/ (Envelope Analysis & Kurtogram)
│   ├── fast_kurtogram.py           # 快速谱峭度底层实现
│   ├── 预白化+快速谱峭度+包络解析(BJTU)_批量处理.ipynb # 一键式流水线
│   └── 诊断结果图_批量输出/             # 自动生成的各工况分析图谱
├── 小波变换/ (Wavelet Transform)
│   └── 小波变换test.ipynb            # 时频特征提取探索
├── 论文复现/ (VMD & Blind Source Separation)
│   ├── main.py & model.py          # 深度学习模型定义与训练脚本
│   ├── mvd_net_final.pth           # 预训练模型权重
│   ├── vmd单通道盲源分离.ipynb         # VMD 信号分解核心逻辑
│   └── 实验结果对比图.ipynb            # 复现结果评估
└── 数据集/ (Datasets Integration)
    ├── BJTU/                       # 包含内圈、外圈、滚动体及复合故障数据
    └── cwru/                       # 包含 CWRU 基准测试数据与专属分析脚本
