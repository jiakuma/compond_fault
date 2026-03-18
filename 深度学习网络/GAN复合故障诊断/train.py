import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from model import GANFaultDiagnosis, ConditionalGANFaultDiagnosis, Generator, Discriminator

# 数据加载函数
def data_load(file_path):
    return pd.read_csv(file_path, usecols=['CH17']).values

# 超参数设置
WINDOW_SIZE = 1024
STEP_SIZE = 512
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 滑动窗口切分函数
def sliding_window(data, window_size=1024, step_size=512):
    """滑动窗口切分一维数据"""
    data = data.flatten()
    num_samples = (len(data) - window_size) // step_size + 1
    samples = np.zeros((num_samples, window_size))
    for i in range(num_samples):
        start = i * step_size
        samples[i] = data[start : start + window_size]
    return samples

# 数据预处理函数
def load_and_preprocess_data():
    print("开始加载并预处理数据...")
    
    # 加载数据
    data_normal_20hz = data_load('../../数据集/BJTU/正常/data_leftaxlebox_M0_G0_LA0_RA0_20Hz_-10kN.csv')
    data_inner_20hz = data_load('../../数据集/BJTU/内圈/data_leftaxlebox_M0_G0_LA1_RA0_20Hz_-10kN.csv')
    data_outer_20hz = data_load('../../数据集/BJTU/外圈/data_leftaxlebox_M0_G0_LA2_RA0_20Hz_-10kN.csv')
    data_rolling_20hz = data_load('../../数据集/BJTU/滚动体/data_leftaxlebox_M0_G0_LA3_RA0_20Hz_-10kN.csv')
    data_inner_outer_20hz = data_load('../../数据集/BJTU/外圈加内圈/data_leftaxlebox_M0_G0_LA1+LA2_RA0_20Hz_-10kN.csv')
    data_outer_rolling_20hz = data_load('../../数据集/BJTU/外圈加滚动体/data_leftaxlebox_M0_G0_LA2+LA3_RA0_20Hz_-10kN.csv')

    # 滑动窗口切分
    X_normal = sliding_window(data_normal_20hz, WINDOW_SIZE, STEP_SIZE)
    X_inner = sliding_window(data_inner_20hz, WINDOW_SIZE, STEP_SIZE)
    X_outer = sliding_window(data_outer_20hz, WINDOW_SIZE, STEP_SIZE)
    X_rolling = sliding_window(data_rolling_20hz, WINDOW_SIZE, STEP_SIZE)
    X_inner_outer = sliding_window(data_inner_outer_20hz, WINDOW_SIZE, STEP_SIZE)
    X_outer_rolling = sliding_window(data_outer_rolling_20hz, WINDOW_SIZE, STEP_SIZE)

    # 生成标签 (0: 正常, 1: 内圈, 2: 外圈, 3: 滚动体, 4: 内圈+外圈, 5: 外圈+滚动体)
    y_normal = np.zeros(len(X_normal))
    y_inner = np.ones(len(X_inner)) * 1
    y_outer = np.ones(len(X_outer)) * 2
    y_rolling = np.ones(len(X_rolling)) * 3
    y_inner_outer = np.ones(len(X_inner_outer)) * 4
    y_outer_rolling = np.ones(len(X_outer_rolling)) * 5

    # 合并数据
    X_all = np.concatenate((X_normal, X_inner, X_outer, X_rolling, X_inner_outer, X_outer_rolling), axis=0)
    y_all = np.concatenate((y_normal, y_inner, y_outer, y_rolling, y_inner_outer, y_outer_rolling), axis=0)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # 实例归一化
    def instance_normalize(data):
        # 沿着时间轴 (axis=1) 独立计算每一个样本的均值和标准差
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        # 加上 1e-8 防止除零错
        return (data - mean) / (std + 1e-8)

    X_train = instance_normalize(X_train)
    X_val = instance_normalize(X_val)

    # 转换形状适配 PyTorch 1D 卷积: (Samples, Channels, Sequence_Length)
    X_train = X_train.reshape(-1, 1, WINDOW_SIZE)
    X_val = X_val.reshape(-1, 1, WINDOW_SIZE)

    # 转换为 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")
    print(f"类别分布: 正常: {len(y_normal)}, 内圈: {len(y_inner)}, 外圈: {len(y_outer)}, 滚动体: {len(y_rolling)}, 内圈+外圈: {len(y_inner_outer)}, 外圈+滚动体: {len(y_outer_rolling)}")
    return train_loader, val_loader, X_val, y_val

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

# 绘制T-SNE图
def plot_tsne(features, labels, classes, model_name):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='viridis', s=50)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.title(f'{model_name} T-SNE 可视化')
    plt.savefig(f'{model_name}_tsne.png')
    plt.close()

# 训练GAN模型
def train_gan():
    train_loader, val_loader, X_val, y_val = load_and_preprocess_data()

    # 初始化GAN模型
    generator = Generator(latent_dim=LATENT_DIM, output_dim=1, sequence_length=WINDOW_SIZE).to(DEVICE)
    discriminator = Discriminator(input_dim=1, sequence_length=WINDOW_SIZE).to(DEVICE)
    gan_model = GANFaultDiagnosis(latent_dim=LATENT_DIM, input_dim=1, sequence_length=WINDOW_SIZE, num_classes=6).to(DEVICE)

    # 定义损失函数和优化器
    criterion_gan = torch.nn.BCELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_cls = optim.Adam(gan_model.parameters(), lr=LEARNING_RATE)

    print(f"\n模型已加载到: {DEVICE}")
    print("开始训练 GAN 模型...")

    for epoch in range(EPOCHS):
        # 训练阶段
        gan_model.train()
        generator.train()
        discriminator.train()
        train_loss_g = 0.0
        train_loss_d = 0.0
        train_loss_cls = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            batch_size = batch_x.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            
            # 真实样本
            real_output = discriminator(batch_x)
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            loss_real = criterion_gan(real_output, real_labels)
            
            # 生成样本
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_x = generator(z)
            fake_output = discriminator(fake_x.detach())
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
            loss_fake = criterion_gan(fake_output, fake_labels)
            
            # 判别器损失
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()
            train_loss_d += loss_d.item() * batch_size

            # 训练生成器
            optimizer_g.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_x = generator(z)
            fake_output = discriminator(fake_x)
            loss_g = criterion_gan(fake_output, real_labels)  # 生成器希望判别器将生成样本判为真实
            loss_g.backward()
            optimizer_g.step()
            train_loss_g += loss_g.item() * batch_size

            # 训练分类器
            optimizer_cls.zero_grad()
            outputs = gan_model(batch_x)
            loss_cls = criterion_cls(outputs, batch_y)
            loss_cls.backward()
            optimizer_cls.step()
            train_loss_cls += loss_cls.item() * batch_size
            
            # 计算准确率
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_size

        epoch_train_loss_g = train_loss_g / total_train
        epoch_train_loss_d = train_loss_d / total_train
        epoch_train_loss_cls = train_loss_cls / total_train
        epoch_train_acc = correct_train / total_train

        # 验证阶段
        gan_model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = gan_model(batch_x)
                loss = criterion_cls(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{EPOCHS}] | G Loss: {epoch_train_loss_g:.4f}, D Loss: {epoch_train_loss_d:.4f}, Cls Loss: {epoch_train_loss_cls:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # 保存模型
    torch.save(gan_model.state_dict(), "gan_fault_diagnosis.pth")
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("\n模型已保存")

    # 生成混淆矩阵和T-SNE图
    gan_model.eval()
    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = gan_model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            # 提取特征
            features = gan_model.feature_extractor(batch_x)
            features = features.view(batch_x.size(0), -1)
            all_features.extend(features.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.array(all_features)

    # 类别名称
    classes = ['正常', '内圈故障', '外圈故障', '滚动体故障', '内圈+外圈故障', '外圈+滚动体故障']

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, classes, "gan")
    print("混淆矩阵已保存为 gan_confusion_matrix.png")

    # 绘制T-SNE图
    plot_tsne(all_features, all_labels, classes, "gan")
    print("T-SNE图已保存为 gan_tsne.png")

# 训练条件GAN模型
def train_conditional_gan():
    train_loader, val_loader, X_val, y_val = load_and_preprocess_data()

    # 初始化条件GAN模型
    cgan_model = ConditionalGANFaultDiagnosis(latent_dim=LATENT_DIM, input_dim=1, sequence_length=WINDOW_SIZE, num_classes=6).to(DEVICE)

    # 定义损失函数和优化器
    criterion_gan = torch.nn.BCELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(cgan_model.generator.parameters(), lr=LEARNING_RATE)
    optimizer_d = optim.Adam(cgan_model.discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_cls = optim.Adam(cgan_model.classifier.parameters(), lr=LEARNING_RATE)

    print(f"\n模型已加载到: {DEVICE}")
    print("开始训练条件 GAN 模型...")

    for epoch in range(EPOCHS):
        # 训练阶段
        cgan_model.train()
        train_loss_g = 0.0
        train_loss_d = 0.0
        train_loss_cls = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            batch_size = batch_x.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            
            # 真实样本
            real_output = cgan_model.discriminate(batch_x, batch_y)
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            loss_real = criterion_gan(real_output, real_labels)
            
            # 生成样本
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_x = cgan_model.generate(z, batch_y)
            fake_output = cgan_model.discriminate(fake_x.detach(), batch_y)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
            loss_fake = criterion_gan(fake_output, fake_labels)
            
            # 判别器损失
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()
            train_loss_d += loss_d.item() * batch_size

            # 训练生成器
            optimizer_g.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_x = cgan_model.generate(z, batch_y)
            fake_output = cgan_model.discriminate(fake_x, batch_y)
            loss_g = criterion_gan(fake_output, real_labels)  # 生成器希望判别器将生成样本判为真实
            loss_g.backward()
            optimizer_g.step()
            train_loss_g += loss_g.item() * batch_size

            # 训练分类器
            optimizer_cls.zero_grad()
            outputs = cgan_model(batch_x)
            loss_cls = criterion_cls(outputs, batch_y)
            loss_cls.backward()
            optimizer_cls.step()
            train_loss_cls += loss_cls.item() * batch_size
            
            # 计算准确率
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_size

        epoch_train_loss_g = train_loss_g / total_train
        epoch_train_loss_d = train_loss_d / total_train
        epoch_train_loss_cls = train_loss_cls / total_train
        epoch_train_acc = correct_train / total_train

        # 验证阶段
        cgan_model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = cgan_model(batch_x)
                loss = criterion_cls(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{EPOCHS}] | G Loss: {epoch_train_loss_g:.4f}, D Loss: {epoch_train_loss_d:.4f}, Cls Loss: {epoch_train_loss_cls:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # 保存模型
    torch.save(cgan_model.state_dict(), "conditional_gan_fault_diagnosis.pth")
    print("\n模型已保存")

    # 生成混淆矩阵和T-SNE图
    cgan_model.eval()
    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = cgan_model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            # 提取特征
            features = cgan_model.classifier[0:8](batch_x)  # 提取分类器的前8层作为特征
            features = features.view(batch_x.size(0), -1)
            all_features.extend(features.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_features = np.array(all_features)

    # 类别名称
    classes = ['正常', '内圈故障', '外圈故障', '滚动体故障', '内圈+外圈故障', '外圈+滚动体故障']

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, classes, "conditional_gan")
    print("混淆矩阵已保存为 conditional_gan_confusion_matrix.png")

    # 绘制T-SNE图
    plot_tsne(all_features, all_labels, classes, "conditional_gan")
    print("T-SNE图已保存为 conditional_gan_tsne.png")

if __name__ == '__main__':
    # 训练GAN模型
    train_gan()
    # 训练条件GAN模型
    train_conditional_gan()
