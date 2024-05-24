__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # Assuming the data files are CSVs
import sys
from scipy.fftpack import fft
sys.path.append(r'..\ENTROPY')
from entropy import SampEn  # 导入 ENTROPY 类
sys.path.append(r'..\SVD')
from denoiser import Denoiser  # 导入 Denoiser 类
sys.path.append(r'..\signal_analysis')  # 导入信噪比和均方根误差类
from signal_analysis import calculate_snr, calculate_rmse


class ICEEMDAN:
    def __init__(self, input_folder, output_folder, entropy_threshold, Nstd, NR, MaxIter):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.entropy_threshold = entropy_threshold  # 熵值阈值
        self.eng = matlab.engine.start_matlab()  # Start MATLAB engine
        self.sampen_calculator = SampEn()  # 创建一个 ENTROPY 实例，如果不需要初始化参数，传入 None
        self.denoiser = Denoiser(mode="layman")  # 实例化 Denoiser 类
        self.Nstd = Nstd  # 从外部传入噪声标准差
        self.NR = NR  # 从外部传入实现的数量
        self.MaxIter = MaxIter  # 从外部传入最大迭代次数

    def plot_imfs(self, imfs, x_coord):
        """
        绘制所有 IMF 分量的图形，每个 IMF 分量在独立的图形窗口中显示。
        :param imfs: 一个包含所有 IMF 分量的数组
        """
        n_imfs = imfs.shape[0]
        for i in range(n_imfs):
            plt.figure(figsize=(10, 2))
            plt.plot(x_coord, imfs[i], 'r')
            plt.title(f'IMF {i + 1}')
            plt.xlabel("Distance")
            plt.ylabel("Amplitude")
            plt.show()

            # 绘制频谱
            plt.figure(figsize=(10, 2))
            n = len(imfs[i])
            T = x_coord[1] - x_coord[0] if len(x_coord) > 1 else 1  # 计算采样间隔
            yf = fft(imfs[i])
            xf = np.linspace(0.0, 1.0 / (2.0 * T), n // 2)
            plt.plot(xf, 2.0 / n * np.abs(yf[:n // 2]), 'b')
            plt.title(f'IMF {i + 1} - Frequency Spectrum')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.show()

    def iceemdan_and_denoise(self, original_signal, df, x_coord):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs, iters = self.eng.ceemdan(A, self.Nstd, self.NR, self.MaxIter, nargout=2)
        imfs = np.array(imfs)

        # 绘制计算出的所有 IMF 分量
        self.plot_imfs(imfs,x_coord)

        noise_flags = []  # 标记每个IMF是否被认为是噪声
        valid_imfs = []  # 用于存储被认为是有效的IMF

        for i, imf in enumerate(imfs):
            m = 2  # 示例值
            r = 0.2 * np.std(imf)  # 示例比例
            entropy = self.sampen_calculator.sampen(imf, m, r)
            print(f"IMF {i + 1}: Entropy = {entropy:.4f}")  # 输出每个IMF的排列熵值
            if entropy < self.entropy_threshold:
                # 如果样本熵低于阈值，则保留这个IMF
               # valid_imfs.append(imf)
               # noise_flags.append(False)
                print(f"IMF {i + 1} is considered clean and is kept.")
            else:
                # 如果样本熵高于阈值，则舍弃这个IMF
               # noise_flags.append(True)
                print(f"IMF {i + 1} is considered noisy and is discarded.")

        # 将所有保留的IMF相加，重构信号
        reconstructed_signal = np.sum(valid_imfs, axis=0) if valid_imfs else np.zeros_like(df)

        # 计算SNR和RMSE时使用 original_signal
        snr = calculate_snr(original_signal, reconstructed_signal)
        rmse = calculate_rmse(original_signal, reconstructed_signal)

        return reconstructed_signal, imfs, valid_imfs, noise_flags, snr, rmse

    def plot_signals(self, x_coord, noisy_signal, reconstructed_signal):
        """
        绘制原始信号和重构信号在同一张图上。
        :param original_signal: 原始信号数组。
        :param reconstructed_signal: 去噪后的信号数组。
        """
        plt.figure(figsize=(18, 4))
        plt.plot(x_coord, noisy_signal, 'b', label='原始信号', linewidth=0.5)  # 蓝色表示原始信号
        plt.plot(x_coord, reconstructed_signal, 'r', label='去噪信号', linewidth=0.5)  # 红色表示去噪后的信号
        plt.title('信号对比')
        plt.xlabel('距离/km')
        plt.ylabel('幅值/dBm')
        plt.legend()
        plt.show()

    def add_noise(self, signal, x_coord, noise_level_base=0.05, noise_level_after=0.5, threshold=26.3):
        """
        在原始信号中加入噪声。
        :param signal: 原始信号数组。
        :param x_coord: 对应的横坐标数组。
        :param noise_level_base: 横坐标小于阈值时的噪声水平，默认为0.05。
        :param noise_level_after: 横坐标大于等于阈值时的噪声水平，默认为0.1。
        :param threshold: 判断噪声水平改变的横坐标阈值，默认为30.8。
        :return: 加噪后的信号。
        """
        noise = np.zeros_like(signal)
        for i, x in enumerate(x_coord):
            if x >= threshold:
                noise_level = noise_level_after
            else:
                noise_level = noise_level_base
            noise[i] = np.random.normal(0, np.std(signal) * noise_level)
        return signal + noise

    def plot_original_and_noisy_signals(self, x_coord, original_signal, noisy_signal):
        """
        绘制原始信号和加噪后信号在同一张图上。
        :param original_signal: 原始信号数组。
        :param noisy_signal: 加噪后的信号数组。
        """
        plt.figure(figsize=(18, 4))
        plt.plot(x_coord, noisy_signal, 'r--', label='加噪信号', linewidth=0.5)  # 红色虚线表示加噪后的信号
        plt.plot(x_coord, original_signal, 'g', label='原始信号', linewidth=0.5)  # 绿色表示原始信号
        plt.title('原始信号与加噪后信号对比')
        plt.xlabel('距离/km')
        plt.ylabel('幅值/dB')
        plt.legend()
        plt.show()

    def process_folder(self):
        for file in os.listdir(self.input_folder):
            if file.endswith(".txt"):  # Check for .txt file extension
                input_file_path = os.path.join(self.input_folder, file)
                df_input = pd.read_csv(input_file_path)  # 默认分隔符为逗号
                x_coord = df_input.iloc[:, 0].values  # 假设第一列是横坐标
                original_signal = df_input.iloc[:, 1].values  # 假设第二列是纵坐标

                # 根据横坐标在原始信号中加入噪声
                noisy_signal = self.add_noise(original_signal, x_coord)
                # 在去噪之前绘制原始信号和加噪后信号的对比图
                self.plot_original_and_noisy_signals(x_coord, original_signal, noisy_signal)
                # 去噪并重构信号
                reconstructed_signal, imfs, denoised_imfs, noise_flags, snr, rmse = self.iceemdan_and_denoise(
                    original_signal, noisy_signal, x_coord)

                # 使用修改后的方法绘制原始信号和重构信号
                self.plot_signals(x_coord, noisy_signal, reconstructed_signal)

                # 创建一个新的 DataFrame 来保存重构的信号和距离信息
                combined_df = pd.DataFrame({
                    'Distance': x_coord,  # 假设第一列是距离信息
                    'Amplitude': reconstructed_signal
                })

                output_file_path = os.path.join(self.output_folder, f"processed_{file}")
                combined_df.to_csv(output_file_path, index=False, sep=';')
                print(f"Processed {file}. SNR: {snr:.2f} dB, RMSE: {rmse:.2f}")

# Example usage
if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    entropy_threshold = 0.3  # 这是一个示例值，您需要根据自己的需求来确定合适的阈值
    Nstd = 0.2  # 设置噪声标准差
    NR = 5   # 设置实现的数量
    MaxIter = 500  # 设置最大迭代次数

    # 确保在创建ICEEMDAN实例时传入所有必需的参数
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False
    iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold, Nstd, NR, MaxIter)
    iceemdan_processor.process_folder()
