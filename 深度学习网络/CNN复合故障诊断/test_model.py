print("开始测试模型导入和初始化...")
import torch
print(f"导入torch成功，版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

from model import CNNFaultDiagnosis, CNNFaultDiagnosisWithAttention
print("导入模型成功")

# 测试模型初始化
try:
    model = CNNFaultDiagnosis(num_classes=6)
    print("基础CNN模型初始化成功")
    
    model_attention = CNNFaultDiagnosisWithAttention(num_classes=6)
    print("带注意力机制的CNN模型初始化成功")
    
    # 测试模型前向传播
    test_input = torch.randn(1, 1, 1024)  # 批大小=1, 通道=1, 序列长度=1024
    output = model(test_input)
    print(f"基础CNN模型前向传播成功，输出形状: {output.shape}")
    
    output_attention = model_attention(test_input)
    print(f"带注意力机制的CNN模型前向传播成功，输出形状: {output_attention.shape}")
    
    print("模型测试成功！")
except Exception as e:
    print(f"模型测试失败: {e}")

print("测试完成！")
