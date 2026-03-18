print("开始测试数据加载...")
import pandas as pd
print("导入pandas成功")

# 数据加载函数
def data_load(file_path):
    print(f"正在加载文件: {file_path}")
    try:
        data = pd.read_csv(file_path, usecols=['CH17']).values
        print(f"文件加载成功，数据形状: {data.shape}")
        return data
    except Exception as e:
        print(f"文件加载失败: {e}")
        return None

# 测试数据加载
data_normal_20hz = data_load('../../数据集/BJTU/正常/data_leftaxlebox_M0_G0_LA0_RA0_20Hz_-10kN.csv')
if data_normal_20hz is not None:
    print("正常数据加载成功！")
else:
    print("正常数据加载失败！")

print("测试完成！")
