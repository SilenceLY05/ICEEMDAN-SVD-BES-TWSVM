__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class ICEEMDAN:
    def __init__(self, input_folder, entropy_threshold):
        self.input_folder = input_folder
        self.entropy_threshold = entropy_threshold
        self.eng = matlab.engine.start_matlab()

    def plot_imfs(self, imfs):
        """
        Plot all IMF components, each in its own figure window.
        """
        n_imfs = imfs.shape[0]
        for i in range(n_imfs):
            plt.figure(figsize=(10, 2))
            plt.plot(imfs[i], 'r')
            plt.title(f'IMF {i + 1}')
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.show()

    def iceemdan_and_plot_imfs(self, signal):
        """
        Perform ICEEMDAN on the signal and plot the IMFs.
        """
        signal_list = signal.tolist()
        A = matlab.double(signal_list)
        imfs = self.eng.emd(A)  # MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        self.plot_imfs(imfs)  # Plot the calculated IMFs

    def plot_original_signal(self, signal):
        """
        绘制原始信号。
        """
        plt.figure(figsize=(18, 4))
        plt.plot(signal, 'b', linewidth=0.5)  # 蓝色线条表示原始信号
        plt.title('Original Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()

    def process_folder(self):
        """
        处理输入文件夹中的每个文件，执行ICEEMDAN，并绘制IMFs。
        """
        for file in os.listdir(self.input_folder):
            if file.endswith(".txt"):
                input_file_path = os.path.join(self.input_folder, file)
                df_input = pd.read_csv(input_file_path)
                signal = df_input.iloc[:, 1].values  # 假设第二列是信号

                # 绘制原始信号
                self.plot_original_signal(signal)

                # 使用ICEEMDAN处理信号并绘制IMFs
                self.iceemdan_and_plot_imfs(signal)

                print(f"Processed {file}")


if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    entropy_threshold = 0.1  # Example threshold, adjust based on your needs
    iceemdan_processor = ICEEMDAN(input_folder, entropy_threshold)
    iceemdan_processor.process_folder()
