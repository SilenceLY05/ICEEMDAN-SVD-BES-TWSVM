import matlab.engine
import numpy as np
import os
import pandas as pd
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
        self.sampen_calculator = SampEn()
        self.denoiser = Denoiser(mode="layman")

    def iceemdan_and_denoise(self, df, r, entropy_threshold):
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        noise_flags = []
        valid_imfs = []

        for i, imf in enumerate(imfs):
            r_dynamic = r * np.std(imf)  # Dynamic r based on current imf
            entropy = self.sampen_calculator.sampen(imf, 2, r_dynamic)
            if entropy < entropy_threshold:
                valid_imfs.append(imf)
            else:
                noise_flags.append(i)  # Keep track of noisy IMFs

        # Reconstruct the signal from the non-noisy IMFs
        if valid_imfs:
            reconstructed_signal = np.sum(valid_imfs, axis=0)
            # Denoise the reconstructed signal
            denoised_signal = self.denoiser.denoise(reconstructed_signal)
            snr = calculate_snr(df, denoised_signal)
            rmse = calculate_rmse(df, denoised_signal)
        else:
            print("All IMFs were discarded, no signal to reconstruct.")
            snr, rmse = np.nan, np.nan  # Set SNR and RMSE to NaN if no IMFs are valid

        return snr, rmse

    def process_and_evaluate(self):
        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        r_values = np.arange(0.1, 0.26, 0.01)
        entropy_threshold_values = np.arange(0.1, 0.66, 0.05)

        for r in r_values:
            for entropy_threshold in entropy_threshold_values:
                # Assume there is only one file and use it
                input_file_path = os.path.join(self.input_folder, os.listdir(self.input_folder)[0])
                df_input = pd.read_csv(input_file_path, delimiter=',')
                original_signal = df_input.iloc[:, 1].values

                try:
                    snr, rmse = self.iceemdan_and_denoise(original_signal, r, entropy_threshold)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    snr, rmse = np.nan, np.nan

                # Record the SNR and RMSE for the current iteration
                results_file_path = os.path.join(self.output_folder, 'evaluation_results.csv')
                with open(results_file_path, 'a') as file:
                    file.write(f"{r},{entropy_threshold},{snr},{rmse}\n")


if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
    iceemdan_processor = ICEEMDAN(input_folder, output_folder)
    iceemdan_processor.process_and_evaluate()
