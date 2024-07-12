import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_txt_files(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file_name)

            # 读取文件数据
            df = pd.read_csv(file_path, header=None, names=['Distance_km', 'Power_dB'], delimiter=',')

            # 绘图
            plt.figure(figsize=(10, 6))  # 创建图形对象，设置大小
            plt.plot(df['Distance_km'], df['Power_dB'], linestyle='-', linewidth=0.5)  # 绘制线图，只连接数据点
            plt.title(f'Distance vs Power - {file_name}')  # 设置图形标题
            plt.xlabel('Distance (km)')  # 设置x轴标签
            plt.ylabel('Power (dB)')  # 设置y轴标签
            plt.grid(True)  # 显示网格
            plt.show()  # 显示图形

# 指定包含.txt文件的文件夹路径
folder_path = r'G:\毕业论文文件\txt测试数据'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False
plot_txt_files(folder_path)
