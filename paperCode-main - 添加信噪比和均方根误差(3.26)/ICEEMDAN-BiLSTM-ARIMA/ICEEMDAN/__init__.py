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
        """
        绘制所有 IMF 分量的图形，每个 IMF 分量在独立的图形窗口中显示。
        :param imfs: 一个包含所有 IMF 分量的数组
        """
        n_imfs = imfs.shape[0]
        for i in range(n_imfs):
            plt.figure(figsize=(10, 2))
            plt.plot(imfs[i], 'r')
            plt.title(f'IMF {i + 1}')
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.show()

    def iceemdan_and_denoise(self, df):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)
        self.plot_imfs(imfs)  # 展示所有原始的 IMF 分量

        noise_flags = []  # 标记每个IMF是否被认为是噪声
        denoised_imfs = np.zeros_like(imfs)

        for i, imf in enumerate(imfs):
            m = 2  # 示例值
            r = 0.2 * np.std(imf)  # 示例比例
            entropy = self.sampen_calculator.sampen(imf, m, r)
            if entropy >= self.entropy_threshold:
                noise_flags.append(True)
                print(f"IMF {i + 1} is considered potential noise based on its sample entropy.")
                denoised_imfs[i] = self.denoiser.denoise(imf)
            else:
                noise_flags.append(False)
                denoised_imfs[i] = imf

        reconstructed_signal = np.sum(denoised_imfs, axis=0)

        # 计算SNR和RMSE
        snr = calculate_snr(df, reconstructed_signal)
        rmse = calculate_rmse(df, reconstructed_signal)

        return reconstructed_signal, imfs, denoised_imfs, noise_flags, snr, rmse

    def plot_signals(self, original_signal, reconstructed_signal):
        """
        绘制原始信号和重构信号在同一张图上。
        :param original_signal: 原始信号数组。
        :param reconstructed_signal: 去噪后的信号数组。
        """
        plt.figure(figsize=(12, 6))
        plt.plot(original_signal, 'b', label='原始信号')  # 蓝色表示原始信号
        plt.plot(reconstructed_signal, 'r', label='去噪信号')  # 红色表示去噪后的信号
        plt.title('信号对比')
        plt.xlabel('采样点')
        plt.ylabel('幅值/dBm')
        plt.legend()
        plt.show()

    def process_folder(self):
        for file in os.listdir(self.input_folder):
            if file.endswith(".txt"):  # Check for .txt file extension
                input_file_path = os.path.join(self.input_folder, file)
                df_input = pd.read_csv(input_file_path)  # Default delimiter is comma
                original_signal = df_input.iloc[:, 1].values  # 假设第二列是原始信号

                # 去噪并重构信号
                reconstructed_signal, imfs, denoised_imfs, noise_flags, snr, rmse = self.iceemdan_and_denoise(
                    original_signal)

                # 使用修改后的方法绘制原始信号和重构信号
                self.plot_signals(original_signal, reconstructed_signal)

                # 创建一个新的 DataFrame 来保存重构的信号和距离信息
                combined_df = pd.DataFrame({
                    'Distance': df_input.iloc[:, 0].values,  # 假设第一列是距离信息
                    'Amplitude': reconstructed_signal
                })

                output_file_path = os.path.join(self.output_folder, f"processed_{file}")
                combined_df.to_csv(output_file_path, index=False, sep=';')
                print(f"Processed {file}. SNR: {snr:.2f} dB, RMSE: {rmse:.2f}")

# Example usage
if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    entropy_threshold = 0.2 # 这是一个示例值，您需要根据自己的需求来确定合适的阈值
    iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold)
    iceemdan_processor.process_folder()