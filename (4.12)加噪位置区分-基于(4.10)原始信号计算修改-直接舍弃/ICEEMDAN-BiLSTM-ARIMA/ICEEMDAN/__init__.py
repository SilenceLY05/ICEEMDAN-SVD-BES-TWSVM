__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # Assuming the data files are CSVs
import sys
sys.path.append(r'..\ENTROPY')
from entropy import SampEn  # 导入 ENTROPY 类
sys.path.append(r'..\SVD')
from denoiser import Denoiser  # 导入 Denoiser 类
sys.path.append(r'..\signal_analysis')  # 导入信噪比和均方根误差类
from signal_analysis import calculate_snr, calculate_rmse


class ICEEMDAN:
    def __init__(self, input_folder, output_folder, entropy_threshold):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.entropy_threshold = entropy_threshold  # 熵值阈值
        self.eng = matlab.engine.start_matlab()  # Start MATLAB engine
        self.sampen_calculator = SampEn()  # 创建一个 ENTROPY 实例，如果不需要初始化参数，传入 None
        self.denoiser = Denoiser(mode="layman")  # 实例化 Denoiser 类

    def plot_imfs(self, imfs):
        n_imfs = imfs.shape[0]
        for i in range(n_imfs):
            plt.figure(figsize=(10, 2))
            plt.plot(imfs[i], 'r')
            plt.title(f'IMF {i + 1}')
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.show()

    def iceemdan_and_denoise(self, original_signal, df):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        self.plot_imfs(imfs)

        valid_imfs = []  # 用于存储被认为是有效的IMF

        for i, imf in enumerate(imfs):
            m = 2
            r = 0.2 * np.std(imf)
            entropy = self.sampen_calculator.sampen(imf, m, r)
            if entropy < self.entropy_threshold:
                valid_imfs.append(imf)
                print(f"IMF {i + 1} is considered clean and is kept.")
            else:
                print(f"IMF {i + 1} is considered noisy due to high entropy and is discarded.")

        # 将所有被认为是有效的IMF相加，重构信号
        reconstructed_signal = np.sum(valid_imfs, axis=0) if valid_imfs else np.zeros_like(df)

        # 对重构信号进行去噪处理
        denoised_reconstructed_signal = self.denoiser.denoise(reconstructed_signal)

        # 计算SNR和RMSE时使用 original_signal
        snr = calculate_snr(original_signal, denoised_reconstructed_signal)
        rmse = calculate_rmse(original_signal, denoised_reconstructed_signal)

        return denoised_reconstructed_signal, snr, rmse

    def plot_signals(self, noisy_signal, reconstructed_signal):
        """
        绘制原始信号和重构信号在同一张图上。
        :param original_signal: 原始信号数组。
        :param reconstructed_signal: 去噪后的信号数组。
        """
        plt.figure(figsize=(18, 4))
        plt.plot(noisy_signal, 'b', label='原始信号', linewidth=0.5)  # 蓝色表示原始信号
        plt.plot(reconstructed_signal, 'r', label='去噪信号', linewidth=0.5)  # 红色表示去噪后的信号
        plt.title('信号对比')
        plt.xlabel('采样点')
        plt.ylabel('幅值/dBm')
        plt.legend()
        plt.show()

    def add_noise(self, signal, x_coord, noise_level_base=0.05, noise_level_after=0.1, threshold=30.8):
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

    def plot_original_and_noisy_signals(self, original_signal, noisy_signal):
        """
        绘制原始信号和加噪后信号在同一张图上。
        :param original_signal: 原始信号数组。
        :param noisy_signal: 加噪后的信号数组。
        """
        plt.figure(figsize=(18, 4))
        plt.plot(noisy_signal, 'r--', label='加噪信号', linewidth=0.5)  # 红色虚线表示加噪后的信号
        plt.plot(original_signal, 'g', label='原始信号', linewidth=0.5)  # 绿色表示原始信号
        plt.title('原始信号与加噪后信号对比')
        plt.xlabel('采样点')
        plt.ylabel('幅值')
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
                self.plot_original_and_noisy_signals(original_signal, noisy_signal)
                # 去噪并重构信号
                reconstructed_signal, snr, rmse = self.iceemdan_and_denoise(original_signal, noisy_signal)

                # 使用修改后的方法绘制原始信号和重构信号
                self.plot_signals(noisy_signal, reconstructed_signal)

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
    entropy_threshold = 0.1 # 这是一个示例值，您需要根据自己的需求来确定合适的阈值
    iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold)
    iceemdan_processor.process_folder()