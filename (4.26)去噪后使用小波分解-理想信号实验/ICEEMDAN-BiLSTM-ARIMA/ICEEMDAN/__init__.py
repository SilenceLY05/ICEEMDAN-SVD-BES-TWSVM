__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # Assuming the data files are CSVs
import sys
import pywt
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

    def save_imfs(self, imfs, x_coord, output_folder):
        imf_folder = os.path.join(output_folder, "IMFs")
        if not os.path.exists(imf_folder):
            os.makedirs(imf_folder)

        n_imfs = imfs.shape[0]
        for i in range(n_imfs):
            imf_data = pd.DataFrame({'X': x_coord, 'Y': imfs[i]})
            imf_data.to_csv(os.path.join(imf_folder, f"IMF_{i + 1}.txt"), index=False, sep=',')
            plt.figure(figsize=(10, 2))
            plt.plot(x_coord, imfs[i], 'r')
            plt.title(f'IMF {i + 1}')
            plt.xlabel("Distance")
            plt.ylabel("Amplitude")
            plt.savefig(os.path.join(imf_folder, f"IMF_{i + 1}.png"))
            plt.close()

    def iceemdan_and_denoise(self, original_signal, x_coord):
        dfList = original_signal.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        # 绘制计算出的所有 IMF 分量
        #self.plot_imfs(imfs,x_coord)

        noise_flags = []  # 标记每个IMF是否被认为是噪声
        valid_imfs = []  # 用于存储被认为是有效的IMF

        for i, imf in enumerate(imfs):
            m = 2  # 示例值
            r = 0.2 * np.std(imf)  # 示例比例
            entropy = self.sampen_calculator.sampen(imf, m, r)
            print(f"IMF {i + 1}: Entropy = {entropy:.4f}")  # 输出每个IMF的排列熵值
            if entropy >= self.entropy_threshold:
                print(f"IMF {i + 1} is considered noisy due to high entropy.")
                # 进行去噪处理
                denoised_imf = self.denoiser.denoise(imf)
                # 再次计算样本熵
                entropy_after_denoise = self.sampen_calculator.sampen(denoised_imf, m, r)
                if entropy_after_denoise < self.entropy_threshold:
                    # 如果去噪后的样本熵低于阈值，则保留去噪后的IMF
                    valid_imfs.append(denoised_imf)
                    noise_flags.append(False)
                    print(f"IMF {i + 1} is kept after denoising.")
                else:
                    # 如果去噪后的样本熵仍然高于阈值，则舍弃这个IMF
                    noise_flags.append(True)
                    print(f"IMF {i + 1} remains noisy after denoising and is discarded.")
            else:
                # 如果原始的样本熵就低于阈值，则保留这个IMF
                valid_imfs.append(imf)
                noise_flags.append(False)
                print(f"IMF {i + 1} is considered clean and is kept.")

        # 将所有保留的IMF相加，重构信号
        reconstructed_signal = np.sum(valid_imfs, axis=0) if valid_imfs else np.zeros_like(original_signal)


        # 计算SNR和RMSE时使用 original_signal
        snr = calculate_snr(original_signal, reconstructed_signal)
        rmse = calculate_rmse(original_signal, reconstructed_signal)

        return reconstructed_signal, imfs, valid_imfs, noise_flags, snr, rmse

    def plot_signals(self, x_coord, original_signal, reconstructed_signal, output_folder, file_name):
        plt.figure(figsize=(18, 4))
        plt.plot(x_coord, original_signal, 'b', label='原始信号', linewidth=0.5)
        plt.plot(x_coord, reconstructed_signal, 'r', label='去噪信号', linewidth=0.5)
        plt.title('信号对比')
        plt.xlabel('距离/km')
        plt.ylabel('幅值/dBm')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"{file_name}.png"))
        plt.close()

    def plot_original_signal(self, x_coord, original_signal, output_folder, file_name):
        plt.figure(figsize=(18, 4))
        plt.plot(x_coord, original_signal, 'b', label='原始信号', linewidth=0.5)
        plt.title('原始信号')
        plt.xlabel('距离/km')
        plt.ylabel('幅值/dB')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"{file_name}.png"))
        plt.close()

    def wavelet_decompose(self, signal, x_coord, output_folder, level=5, wavelets=['db5', 'haar', 'coif5', 'sym8']):
        """
        使用小波变换对信号进行分解，并保存和绘制分解结果。
        :param signal: 要分解的信号数组。
        :param x_coord: 信号的横坐标。
        :param output_folder: 图形和数据保存的文件夹路径。
        :param level: 小波分解的级别。
        :param wavelet: 使用的小波类型。
        """
        for wavelet in wavelets:
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            fig, axs = plt.subplots(nrows=len(coeffs), ncols=1, figsize=(12, 10))
            fig.suptitle(f'{wavelet} Wavelet Decomposition', fontsize=16)

            for i, coef in enumerate(coeffs):
                axs[i].plot(x_coord[:len(coef)], coef, label=f'Level {i}')
                axs[i].set_title(f'Detail Coefficients at Level {i}')
                axs[i].legend()

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_folder, f"{wavelet}_wavelet_decomposition.png"))
            plt.close()

            # 保存小波分解系数到文件
            for i, coef in enumerate(coeffs):
                coef_df = pd.DataFrame({'X': x_coord[:len(coef)], 'Y': coef})
                coef_df.to_csv(os.path.join(output_folder, f"{wavelet}_wavelet_level_{i}.txt"), index=False, sep=',')


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


    def process_folder(self):
        for file in os.listdir(self.input_folder):
            if file.endswith(".txt"):
                input_file_path = os.path.join(self.input_folder, file)
                df_input = pd.read_csv(input_file_path)
                x_coord = df_input.iloc[:, 0].values
                original_signal = df_input.iloc[:, 1].values

                # 在原始信号中加入噪声
                noisy_signal = self.add_noise(original_signal, x_coord)

                file_output_folder = os.path.join(self.output_folder, os.path.splitext(file)[0])
                if not os.path.exists(file_output_folder):
                    os.makedirs(file_output_folder)

                self.plot_original_signal(x_coord, noisy_signal, file_output_folder, 'noisy_signal')
                reconstructed_signal, imfs, denoised_imfs, noise_flags, snr, rmse = self.iceemdan_and_denoise(
                    noisy_signal, x_coord)

                self.plot_signals(x_coord, noisy_signal, reconstructed_signal, file_output_folder, 'signal_comparison')
                self.save_imfs(imfs, x_coord, file_output_folder)

                # 对去噪后的信号进行多种小波分解
                self.wavelet_decompose(reconstructed_signal, x_coord, file_output_folder)

                combined_df = pd.DataFrame({
                    'Distance': x_coord,
                    'Amplitude': reconstructed_signal
                })
                output_file_path = os.path.join(file_output_folder, "processed_data.txt")
                combined_df.to_csv(output_file_path, index=False, sep=',')
                print(f"Processed {file}. SNR: {snr:.2f} dB, RMSE: {rmse:.2f}")

# Example usage
if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    entropy_threshold = 0.1 # 这是一个示例值，您需要根据自己的需求来确定合适的阈值
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False
    iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold)
    iceemdan_processor.process_folder()