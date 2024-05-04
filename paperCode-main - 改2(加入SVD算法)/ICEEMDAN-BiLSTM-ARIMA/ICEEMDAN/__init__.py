__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # Assuming the data files are CSVs
import sys
sys.path.append(r'G:\毕业论文文件\paperCode-main - 改1(样本熵)\ICEEMDAN-BiLSTM-ARIMA\ENTROPY')
from entropy import SampEn  # 导入 ENTROPY 类


class ICEEMDAN:
    def __init__(self, input_folder, output_folder, entropy_threshold):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.entropy_threshold = entropy_threshold  # 熵值阈值
        self.eng = matlab.engine.start_matlab()  # Start MATLAB engine
        self.sampen_calculator = SampEn()  # 创建一个 ENTROPY 实例，如果不需要初始化参数，传入 None

    def iceemdan(self, df):
         #Perform ICEEMDAN decomposition, return all IMFs, and indicate potential noise components based on sample entropy.
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        noise_flags = []  # 标记每个IMF是否被认为是噪声

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
            else:
                noise_flags.append(False)

        return imfs, noise_flags  # 返回IMFs及其噪声标记

    # 移除了 @staticmethod 装饰器
    def plot_imfs(self, imfs, data):
        """为每个IMF绘制单独的图形"""
        # 首先绘制原始信号
        plt.figure(figsize=(10, 4))
        plt.plot(data, 'r')
        plt.title("Original Signal")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()

        # 为每个IMF绘制图形
        for i, imf in enumerate(imfs, start=1):
            plt.figure(figsize=(10, 4))
            plt.plot(imf, 'g')
            plt.title(f"IMF {i}")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.show()

    def process_folder(self):
        """处理输入文件夹中的所有文件，并保存IMFs及其噪声标记。"""
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            df = pd.read_csv(file_path, delimiter=';')  # 指定分号为分隔符
            # 假设数据在第二列，需要提取这一列作为待处理的数据
            data = df.iloc[:, 1].values  # 如果需要的数据在其他列，请相应调整索引
            # 调用iceemdan方法并接收IMFs及其噪声标记
            imfs, noise_flags = self.iceemdan(data)

            # 例如，打印出被认为是噪声的IMF的索引
            for i, is_noise in enumerate(noise_flags):
                if is_noise:
                    print(f"IMF {i + 1} is considered potential noise.")

            # 在这里调用 plot_imfs 来绘制原始信号和IMFs
            self.plot_imfs(imfs, data)  # 确保已经移除了 plot_imfs 的 @staticmethod 装饰器

            output_file_path = os.path.join(self.output_folder, f"processed_{file}")
            np.savetxt(output_file_path, imfs, delimiter=",")  # 保存IMFs为CSV

# Example usage
input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
output_folder = r'G:\毕业论文文件\OTDR_Data\After treatment'
entropy_threshold = 0.1  # 这是一个示例值，您需要根据自己的需求来确定合适的阈值

iceemdan_processor = ICEEMDAN(input_folder, output_folder, entropy_threshold)
iceemdan_processor.process_folder()
