__all__ = ["ICEEMDAN"]

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd  # Assuming the data files are CSVs


class ICEEMDAN:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.eng = matlab.engine.start_matlab()  # Start MATLAB engine

    def plot_imfs(self, imfs, x_coord):
        """
        绘制所有 IMF 分量的图形，每个 IMF 分量在独立的图形窗口中显示。
        :param imfs: 一个包含所有 IMF 分量的数组
        """
        n_imfs = imfs.shape[0]
        for i in range(n_imfs):
            plt.figure(figsize=(10, 2))
            plt.plot(x_coord, imfs[i], 'r')
            plt.title(f'IMF {i + 1}')
            plt.xlabel("Distance")
            plt.ylabel("Amplitude")
            plt.show()

    def iceemdan_and_plot(self, x_coord, signal):
        """
        应用 ICEEMDAN 方法并绘制原始信号和结果。
        :param x_coord: 横坐标数组
        :param signal: 原始信号数组
        """
        # 将 Python 列表转换为 MATLAB 可接受的格式
        signal_list = signal.tolist()
        signal_matlab = matlab.double(signal_list)

        # 调用 MATLAB 中的 ICEEMDAN 函数处理信号
        imfs = self.eng.iceemdan(signal_matlab)
        imfs = np.array(imfs)  # 将结果转换回 Numpy 数组格式

        # 绘制原始信号
        plt.figure(figsize=(10, 2))
        plt.plot(x_coord, signal, 'b')
        plt.title("原始信号")
        plt.xlabel("距离")
        plt.ylabel("幅度")
        plt.show()

        # 绘制 IMF 分量
        self.plot_imfs(imfs, x_coord)

    def process_folder(self):
        """
        处理输入文件夹中的所有文件。
        """
        for file in os.listdir(self.input_folder):
            if file.endswith(".txt"):  # 确保处理的是 txt 文件
                file_path = os.path.join(self.input_folder, file)
                df = pd.read_csv(file_path, delimiter=';')  # 使用逗号作为分隔符读取文件
                x_coord = df.iloc[:, 0].values  # 第一列作为横坐标
                signal = df.iloc[:, 1].values  # 第二列作为信号数据

                # 应用 ICEEMDAN 方法并绘制结果
                self.iceemdan_and_plot(x_coord, signal)
                print(f"文件 {file} 已处理完成。")

# Example usage
if __name__ == "__main__":
    input_folder = r'G:\毕业论文文件\OTDR_Data\Before treatment'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False
    iceemdan_processor = ICEEMDAN(input_folder)
    iceemdan_processor.process_folder()