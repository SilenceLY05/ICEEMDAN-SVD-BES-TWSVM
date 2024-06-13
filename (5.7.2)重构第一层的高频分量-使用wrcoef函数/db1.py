import os
import numpy as np
import pywt
import matplotlib.pyplot as plt


def wrcoef(data, wavelet, coeff_type, level, coeffs):
    """Wrapper for reconstructing one level of coefficients."""
    coeffs_full = [None] * (level + 1)
    if coeff_type == 'd':  # details
        coeffs_full[-(level + 1)] = coeffs
    elif coeff_type == 'a':  # approximation
        coeffs_full[0] = coeffs
    result = pywt.waverec(coeffs_full, wavelet)
    return result[:len(data)]

def process_signal(file_path, wavelet='db1'):
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    # 4 层小波分解
    coeffs = pywt.wavedec(y, wavelet, level=4)
    cA4, cD4, cD3, cD2, cD1 = coeffs

    # 重构第一层高频小波系数信号
    OSignal1 = wrcoef(y, wavelet, 'd', 1, cD1)

    # 绘制原始信号
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label='Original Signal')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Original Signal from {os.path.basename(file_path)}')
    plt.show()

    # 绘制重构的高频信号
    plt.figure(figsize=(10, 4))
    plt.plot(x, OSignal1, label='Reconstructed D1', color='orange')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Reconstructed D1 from {os.path.basename(file_path)}')
    plt.show()

# 文件夹路径
folder_path = r'G:\毕业论文文件\OTDR_Data\新建文件夹2'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        process_signal(file_path)
