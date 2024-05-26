import numpy as np
import matplotlib.pyplot as plt

def read_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # 跳过首行标题
    x, y = [], []
    for line in lines:
        parts = line.strip().split(',')  # 假设数据是用逗号分隔的
        x.append(float(parts[0]))
        y.append(float(parts[1]))
    return np.array(x), np.array(y)

def find_local_maxima(x, y):
    maxima_x = []
    maxima_y = []
    for i in range(1, len(y) - 1):
        if y[i-1] < y[i] > y[i+1]:
            maxima_x.append(x[i])
            maxima_y.append(y[i])
    return maxima_x, maxima_y

def plot_data_with_maxima(x, y, maxima_x, maxima_y):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Original Data')
    plt.scatter(maxima_x, maxima_y, color='red', s=50, label='Local Maxima')
    plt.title('Local Maxima in the Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用示例
filepath = r'G:\毕业论文文件\OTDR_Data\adjusted_data_new2\second_decomposition\IMFs\IMF_9.txt'  # 更换为您的文件路径
x, y = read_data(filepath)
maxima_x, maxima_y = find_local_maxima(x, y)
plot_data_with_maxima(x, y, maxima_x, maxima_y)
