import pandas as pd
import matplotlib.pyplot as plt

# 使用原始字符串读取TSV文件
df = pd.read_csv(r'G:\毕业论文文件\otdr_event_classification_training\2023-12-08_simulated_measurements\simulated_measurements_with_noise_and_events.tsv', delim_whitespace=True, header=None)

# 给数据列命名以便于引用
df.columns = ['距离_km', '列2', '列3', 'dB']

# 确保数据的数据类型正确
df['距离_km'] = pd.to_numeric(df['距离_km'], errors='coerce')  # 确保距离数据为数值类型
df['dB'] = pd.to_numeric(df['dB'], errors='coerce')  # 确保dB数据为数值类型

# 去除可能因为转换错误而产生的NaN值
df.dropna(subset=['距离_km', 'dB'], inplace=True)

# 绘图，调整尺寸和线宽
plt.figure(figsize=(30, 3))  # 调整图形尺寸，使横坐标看起来更长
plt.plot(df['距离_km'], df['dB'], marker='o', linewidth=0.1, markersize=3)  # 使用细线（线宽为1）绘制
plt.xlabel('距离 (km)')
plt.ylabel('dB')
plt.title('距离 vs dB')
plt.grid(True)
plt.show()
