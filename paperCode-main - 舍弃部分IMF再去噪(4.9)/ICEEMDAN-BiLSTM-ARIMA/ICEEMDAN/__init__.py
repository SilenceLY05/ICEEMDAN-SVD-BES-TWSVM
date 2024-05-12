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
        self.entropy_threshold = entropy_threshold
        self.eng = matlab.engine.start_matlab()
        self.sampen_calculator = SampEn()
        self.denoiser = Denoiser(mode="layman")

    def iceemdan_and_denoise(self, df):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)
        imfs = np.array(imfs)

        valid_imfs = []
        discarded_imfs = []  # 用于存储被舍弃的IMF索引
        for i, imf in enumerate(imfs):
            entropy = self.sampen_calculator.sampen(imf, 2, 0.2 * np.std(imf))
            if entropy < self.entropy_threshold:
                valid_imfs.append(imf)
                print(f"IMF {i + 1} has been kept.")
            else:
                discarded_imfs.append(i + 1)
                print(f"IMF {i + 1} has been discarded due to high entropy.")

        if not valid_imfs:
            print("All IMFs were discarded, no signal to reconstruct.")
            snr, rmse = np.nan, np.nan
            reconstructed_signal = np.zeros_like(df)
        else:
            reconstructed_signal = np.sum(valid_imfs, axis=0)
            denoised_signal = self.denoiser.denoise(reconstructed_signal)
            snr = calculate_snr(df, denoised_signal)
            rmse = calculate_rmse(df, denoised_signal)

        return denoised_signal, imfs, valid_imfs, snr, rmse

    def plot_signals(self, original_signal, denoised_signal):
        plt.figure(figsize=(12, 6))
        plt.plot(original_signal, 'b', label='原始信号')
        plt.plot(denoised_signal, 'r', label='去噪后信号')
        plt.title('原始信号与去噪后信号对比')
        plt.xlabel('样本点')
        plt.ylabel('幅值')
        plt.legend()
        plt.show()

    def process_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for file in os.listdir(self.input_folder):
            if file.endswith(".txt"):
                input_file_path = os.path.join(self.input_folder, file)
                df_input = pd.read_csv(input_file_path, delimiter=',')
                original_signal = df_input.iloc[:, 1].values

                denoised_signal, imfs, valid_imfs, snr, rmse = self.iceemdan_and_denoise(original_signal)

                # 绘制原始信号和去噪后信号对比图
                self.plot_signals(original_signal, denoised_signal)

                output_file_name = f"processed_{file}"
                output_file_path = os.path.join(self.output_folder, output_file_name)
                np.savetxt(output_file_path, denoised_signal, delimiter=',')

                print(f"Processed {file}. SNR: {snr:.2f} dB, RMSE: {rmse:.2f}")


if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    entropy_threshold = 0.2
    iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold)
    iceemdan_processor.process_folder()
