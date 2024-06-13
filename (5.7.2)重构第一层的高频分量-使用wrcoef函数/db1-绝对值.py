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

def process_signal(file_path, wavelet='haar'):
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    # 4 层小波分解
    coeffs = pywt.wavedec(y, wavelet, level=4)
    cA4, cD4, cD3, cD2, cD1 = coeffs

    # 重构第一层高频小波系数信号
    OSignal1 = np.abs(wrcoef(y, wavelet, 'd', 1, cD1))

    # 找到最大值并确定阈值
    max_val = np.max(OSignal1)
    threshold = max_val / 10
    peak_position = np.argmax(OSignal1)  # 最大峰值的位置

    # 绘制原始信号
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label='Original Signal')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Original Signal from {os.path.basename(file_path)}')
    plt.show()

    # 绘制重构的高频信号，并标记超过阈值的区域
    plt.figure(figsize=(10, 4))
    plt.plot(x, OSignal1, label='Reconstructed Absolute D1', color='orange')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Reconstructed Absolute D1 from {os.path.basename(file_path)}')

    # 找出最大峰值之前的所有显著点
    significant_indices = np.where((OSignal1 > threshold) & (np.arange(len(OSignal1)) <= peak_position))[0]

    # 找出最大峰值之后直到首次低于阈值的点
    if peak_position < len(OSignal1) - 1:
        after_peak = OSignal1[peak_position + 1:]
        first_below_threshold = np.argmax(after_peak < threshold) if np.any(after_peak < threshold) else len(after_peak)
        significant_indices_after_peak = np.arange(peak_position + 1, peak_position + 1 + first_below_threshold)
        significant_indices = np.concatenate((significant_indices, significant_indices_after_peak))

    plt.bar(x[significant_indices], OSignal1[significant_indices], width=x[1] - x[0], color='blue', alpha=0.3)
    plt.legend()
    plt.show()

    # 分段打印显著点的横坐标范围
    if significant_indices.size > 0:
        # 找到所有连续区段并打印
        segments = np.split(significant_indices, np.where(np.diff(significant_indices) != 1)[0] + 1)
        for seg in segments:
            print(f"Significant segment from X = {x[seg[0]]} to X = {x[seg[-1]]}")
    else:
        print("No significant points found.")


# 文件夹路径
folder_path = r'G:\毕业论文文件\OTDR_Data\新建文件夹2'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        process_signal(file_path)
