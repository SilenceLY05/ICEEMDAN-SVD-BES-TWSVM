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
         #Perform ICEEMDAN decomposition, return all IMFs, and indicate potential noise components based on sample entropy.
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)
         #在进行任何去噪处理之前，展示所有原始的 IMF 分量
        self.plot_imfs(imfs)

        noise_flags = []  # 标记每个IMF是否被认为是噪声
        denoised_imfs = np.zeros_like(imfs)  # 用于存储去噪后的 IMFs

        # 计算每个IMF的样本熵并标记噪声分量
        for i, imf in enumerate(imfs):
            # 需要确定计算样本熵所需的 m 和 r 参数值
            m = 2  # 示例值
            r = 0.2 * np.std(imf)  # 根据时间序列标准差的一定比例设置r，示例比例
            entropy = self.sampen_calculator.sampen(imf, m, r)
            if entropy >= self.entropy_threshold:
                # 样本熵低于阈值，认为是噪声
                noise_flags.append(True)
                print(f"IMF {i + 1} is considered potential noise based on its sample entropy.")
                # 将噪声IMF传递给SVD进行去噪处理
                denoised_imfs[i] = self.denoiser.denoise(imf)
            else:
                noise_flags.append(False)
                denoised_imfs[i] = imf  # 未被标记为噪声的 IMF 直接保留

        # 重构信号
        reconstructed_signal = np.sum(denoised_imfs, axis=0)
        return reconstructed_signal, imfs, denoised_imfs, noise_flags

    def plot_signal(self, title, signal):
        plt.figure(figsize=(10, 4))
        plt.plot(signal, 'r')
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()

    def process_folder(self):
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            df = pd.read_csv(file_path, delimiter=';')
            distance_info = df.iloc[:, 0].values  # 保留距离信息
            amplitude_info = df.iloc[:, 1].values
            reconstructed_signal, imfs, denoised_imfs, noise_flags = self.iceemdan_and_denoise(amplitude_info)

            # 绘制原始信号和重构信号
            self.plot_signal("Original Signal", amplitude_info)
            self.plot_signal("Reconstructed Signal", reconstructed_signal)

            # 创建一个新的 DataFrame 来保存重构的信号和距离信息
            combined_df = pd.DataFrame({
                'Distance': distance_info,
                'Amplitude': reconstructed_signal
            })

            # 定义输出文件的路径
            output_file_path = os.path.join(self.output_folder, f"processed_{file}")
            # 保存 DataFrame 到 CSV 文件
            combined_df.to_csv(output_file_path, index=False, sep=';')

# Example usage
if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    entropy_threshold = 0.1 # 这是一个示例值，您需要根据自己的需求来确定合适的阈值
    iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold)
    iceemdan_processor.process_folder()