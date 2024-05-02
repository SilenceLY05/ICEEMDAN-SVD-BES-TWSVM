__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # Assuming the data files are CSVs
import sys
sys.path.append(r'G:\毕业论文文件\paperCode-main - 改2\ICEEMDAN-BiLSTM-ARIMA\ENTROPY')
from entropy import ENTROPY  # 导入 ENTROPY 类


class ICEEMDAN:
    def __init__(self, input_folder, output_folder, entropy_threshold):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.entropy_threshold = entropy_threshold  # 熵值阈值
        self.eng = matlab.engine.start_matlab()  # Start MATLAB engine
        self.entropy_calculator = ENTROPY(None)  # 创建一个 ENTROPY 实例，如果不需要初始化参数，传入 None

    def iceemdan(self, df):
        """Perform ICEEMDAN decomposition."""
        df = np.array(df)
        dfList = df.tolist()
        A = matlab.double(dfList)
        imfs = self.eng.iceemdan(A)  # Call MATLAB ICEEMDAN function
        imfs = np.array(imfs)

        # 计算每个IMF的排列熵并决定是否继续分解
        for i, imf in enumerate(imfs):
            entropy = self.entropy_calculator.permutation_entropy(imf, order=3, delay=1, normalize=True)
            if entropy < self.entropy_threshold:
                print(f"Stopping at IMF {i + 1} due to entropy threshold.")
                return imfs[:i + 1]  # 返回当前及之前所有的IMF
        return imfs

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
        """处理输入文件夹中的所有文件并保存结果。"""
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            df = pd.read_csv(file_path, delimiter=';')  # 指定分号为分隔符
            # 假设数据在第二列，需要提取这一列作为待处理的数据
            data = df.iloc[:, 1].values  # 如果需要的数据在其他列，请相应调整索引
            imfs = self.iceemdan(data)

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
