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


def process_signal(file_path, wavelet):
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    # 4 层小波分解
    coeffs = pywt.wavedec(y, wavelet, level=5)
    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 重构第一层高频小波系数信号
    OSignal1 = wrcoef(y, wavelet, 'd', 1, cD1)

    # 绘制原始信号
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label='Original Signal')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Original Signal from {os.path.basename(file_path)} ({wavelet})')
    plt.show()

    # 绘制重构的高频信号
    plt.figure(figsize=(10, 4))
    plt.plot(x, OSignal1, label=f'Reconstructed D1 ({wavelet})', color='orange')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Reconstructed D1 from {os.path.basename(file_path)} ({wavelet})')
    plt.show()

# 文件夹路径
folder_path = r'G:\毕业论文文件\OTDR_Data\新建文件夹2'

# 使用不同的小波基
wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'haar', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6',
                     'sym7', 'sym8', 'coif5', 'bior3.1', 'dmey']


# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        for wavelet in wavelets:
            process_signal(file_path, wavelet)
