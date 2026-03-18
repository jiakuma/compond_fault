print("开始测试Transformer模型...")
import torch
print(f"导入torch成功，版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

from model import TransformerFaultDiagnosis, WaveletTransformerFaultDiagnosis
print("导入模型成功")

# 测试模型初始化
try:
    model = TransformerFaultDiagnosis(num_classes=6)
    print("基础Transformer模型初始化成功")
    
    model_wavelet = WaveletTransformerFaultDiagnosis(num_classes=6)
    print("小波Transformer模型初始化成功")
    
    # 测试模型前向传播
    test_input = torch.randn(1, 1, 1024)  # 批大小=1, 通道=1, 序列长度=1024
    output = model(test_input)
    print(f"基础Transformer模型前向传播成功，输出形状: {output.shape}")
    
    output_wavelet = model_wavelet(test_input)
    print(f"小波Transformer模型前向传播成功，输出形状: {output_wavelet.shape}")
    
    print("模型测试成功！")
except Exception as e:
    print(f"模型测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成！")
