import numpy as np

# 每个受试者的样本数
samples = [1626, 1760, 1794, 1440, 1760, 1760, 1308, 1760, 1760, 1760, 1760, 1600]

# 计算平均值
mean_samples = np.mean(samples)

print("平均样本数:", mean_samples)