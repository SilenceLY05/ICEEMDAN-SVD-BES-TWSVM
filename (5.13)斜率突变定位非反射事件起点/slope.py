import os
import numpy as np
import pandas as pd
from scipy.stats import linregress


def read_latest_file(directory):
    # 获取文件夹中的所有文件，并按修改时间排序
    all_files = os.listdir(directory)
    full_paths = [os.path.join(directory, file) for file in all_files]
    latest_file = max(full_paths, key=os.path.getmtime)
    # 读取数据
    data = pd.read_csv(latest_file, header=None, names=['Position', 'Value'])
    return data


def sliding_window_analysis(data, window_size=10, step=5):
    slopes = []
    for start in range(0, len(data) - window_size + 1, step):
        # 获取窗口数据
        window_data = data.iloc[start:start + window_size]
        # 计算斜率
        slope, _, _, _, _ = linregress(window_data['Position'], window_data['Value'])
        slopes.append((start, slope))

    # 处理最后一个窗口
    if start + window_size < len(data):
        window_data = data.iloc[start + step:]
        slope, _, _, _, _ = linregress(window_data['Position'], window_data['Value'])
        slopes.append((start + step, slope))

    return slopes


def detect_slope_changes(data, slopes, threshold=2):
    for i in range(1, len(slopes)):
        previous_slope = slopes[i - 1][1]
        current_slope = slopes[i][1]
        previous_position = data.iloc[slopes[i - 1][0]]['Position']
        current_position = data.iloc[slopes[i][0]]['Position']

        # 检查前一个斜率是否为零
        if abs(previous_slope) == 0:
            #print(f"Previous slope is zero at position {previous_position}, cannot compute ratio.")
            continue

        if abs(current_slope) / abs(previous_slope) > threshold:
            print(f"Significant slope change detected at position: {current_position}")


# 主程序
directory_path = r'G:\毕业论文文件\OTDR_Data\新建文件夹2'  # 替换成你的文件夹路径
data = read_latest_file(directory_path)
slopes = sliding_window_analysis(data)
detect_slope_changes(data, slopes)
