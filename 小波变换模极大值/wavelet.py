import numpy as np
import matplotlib.pyplot as plt
import pywt

def read_signal(filename):
    # 从文件中读取数据
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1]

def perform_wtmm(signal, wavelet, levels):
    # 对信号执行小波变换模极大值分析
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    return coeffs

def plot_wtmm(coeffs, wavelet_name):
    # 绘制分解结果
    plt.figure(figsize=(10, 6))
    for i, coeff in enumerate(coeffs):
        plt.subplot(len(coeffs), 1, i + 1)
        plt.plot(coeff)
        plt.title(f'{wavelet_name} - Level {i}')
    plt.tight_layout()
    plt.show()

filename = r'G:\毕业论文文件\OTDR_Data\adjusted_data_new2\processed_adjusted_data_new2.txt'  # 修改为你的文件路径
x, signal = read_signal(filename)

wavelets = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'haar', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6',
                 'sym7', 'sym8', 'coif5', 'bior3.1', 'dmey']
levels = 5  # 小波分解层数

# 对每种小波基执行WTMM分析并绘图
for wavelet in wavelets:
    coeffs = perform_wtmm(signal, wavelet, levels)
    plot_wtmm(coeffs, wavelet)
