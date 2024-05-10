import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd

def perform_svd(signal):
    """对整个信号执行SVD，并返回U, S, Vt。"""
    H = hankel(signal[:-1], signal[1:])
    U, S, Vt = svd(H, full_matrices=False)
    return U, S, Vt

def svd_reconstruct(U, S, Vt, last_value):
    """使用SVD的结果重构信号，选取差分谱中第二大的值作为阈值。"""
    diff_S = np.diff(S)  # 计算奇异值的差分
    sorted_indices = np.argsort(diff_S)  # 得到差分数组排序后的索引
    # 找到第二大差分值对应的索引，加1是因为差分减少了一个元素的长度
    threshold_idx = sorted_indices[-2] + 1
    S[threshold_idx:] = 0  # 将该索引之后的所有奇异值置零
    reconstructed_H = np.dot(U, np.dot(np.diag(S), Vt))
    # 补充原始信号的最后一个值以匹配原始信号的长度
    reconstructed_signal = np.append(np.mean(reconstructed_H, axis=0), last_value)
    return reconstructed_signal



def plot_singular_values(S):
    """绘制奇异值分布图。"""
    plt.figure(figsize=(10, 5))
    plt.plot(S, marker='o', linestyle='-', markersize=4, label='Singular Values')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.title('Singular Value Distribution')
    plt.show()


def plot_difference_spectrum(S):
    """计算奇异值差分谱并绘制。"""
    diff_spectra = np.diff(S)
    plt.figure(figsize=(10, 5))
    plt.plot(diff_spectra, marker='o', linestyle='-', markersize=4, label='Difference Spectrum')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Difference Spectrum')
    plt.legend()
    plt.title('Singular Value Difference Spectrum')
    plt.show()


def plot_original_signal(t, original_signal):
    """绘制原始信号。"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, original_signal, label='Original Signal')
    plt.legend()
    plt.title('Original Signal')
    plt.show()


def plot_reconstructed_signal(t, reconstructed_signal):
    """绘制重构信号。"""
    plt.figure(figsize=(10, 5))
    plt.plot(t, reconstructed_signal, label='Reconstructed Signal', linestyle='-', color='orange')
    plt.legend()
    plt.title('Reconstructed Signal')
    plt.show()


def process_and_plot_signal(signal):
    """处理信号并绘制相关图像。"""
    U, S, Vt = perform_svd(signal)
    # 传递原始信号的最后一个值给重构函数
    reconstructed_signal = svd_reconstruct(U, S, Vt, signal[-1])
    t = np.linspace(0, 1, len(signal))

    plot_singular_values(S)
    plot_difference_spectrum(S)
    plot_original_signal(t, signal)
    plot_reconstructed_signal(t, reconstructed_signal)


# 示例使用
if __name__ == "__main__":
    # 创建一个简单的信号
    t = np.linspace(0, 1, 600)
    original_signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, len(t))

    # 处理信号并绘制结果
    process_and_plot_signal(original_signal)
