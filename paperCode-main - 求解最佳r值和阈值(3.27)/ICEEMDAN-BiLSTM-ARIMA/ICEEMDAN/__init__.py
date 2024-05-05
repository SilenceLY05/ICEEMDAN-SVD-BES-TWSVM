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
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.eng = matlab.engine.start_matlab()  # Start MATLAB engine
        self.sampen_calculator = SampEn()  # 创建一个 ENTROPY 实例，如果不需要初始化参数，传入 None
        self.denoiser = Denoiser(mode="layman")  # 实例化 Denoiser 类


    def iceemdan_and_denoise(self, df, r, entropy_threshold):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        noise_flags = []
        denoised_imfs = np.zeros_like(imfs)

        for i, imf in enumerate(imfs):
            r_dynamic = r * np.std(imf)  # Dynamic r based on current imf
            entropy = self.sampen_calculator.sampen(imf, 2, r_dynamic)
            if entropy >= entropy_threshold:
                noise_flags.append(True)
                print(f"IMF {i + 1} is considered potential noise based on its sample entropy.")
                denoised_imfs[i] = self.denoiser.denoise(imf)
            else:
                noise_flags.append(False)
                denoised_imfs[i] = imf

        reconstructed_signal = np.sum(denoised_imfs, axis=0)
        snr = calculate_snr(df, reconstructed_signal)
        rmse = calculate_rmse(df, reconstructed_signal)

        return snr, rmse

    def process_and_evaluate(self):
        # 确保输出文件夹存在
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        output_file_path = os.path.join(self.output_folder, 'evaluation_results.csv')

        # 使用'w'模式初始化文件并写入表头
        with open(output_file_path, 'w') as file:
            file.write('r,entropy_threshold,SNR,RMSE\n')

        r = 0.1
        while r <= 0.25:
            entropy_threshold = 0.1
            while entropy_threshold <= 0.65:
                input_file_path = os.path.join(self.input_folder, os.listdir(self.input_folder)[0])  # 假设文件夹中只有一个文件
                df_input = pd.read_csv(input_file_path, delimiter=';')
                original_signal = df_input.iloc[:, 1].values

                try:
                    snr, rmse = self.iceemdan_and_denoise(original_signal, r, entropy_threshold)
                except ValueError as e:
                    print(f"Error processing file with r={r}, entropy_threshold={entropy_threshold}: {e}")
                    snr, rmse = np.nan, np.nan  # 使用NaN表示错误

                # 无论是否出现错误，都记录当前迭代的r值和entropy_threshold值，以及可能的snr和rmse
                with open(output_file_path, 'a') as file:
                    file.write(f'{r},{entropy_threshold},{snr},{rmse}\n')

                entropy_threshold += 0.05
            r += 0.01


# Example usage
if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    iceemdan_processor = ICEEMDAN(input_folder, output_folder)
    iceemdan_processor.process_and_evaluate()