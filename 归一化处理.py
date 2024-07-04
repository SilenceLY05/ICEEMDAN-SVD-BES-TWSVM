import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_csv_files(folder_path):
    # 遍历文件夹中的所有.csv文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)

            # 对“Max Position”、“Event Start”、“Event End”和“Event Position”列进行相对位置归一化
            if 'Max Position' in data.columns:
                data['Max Position'] = (data['Max Position'] - data['Start Position']) / (data['End Position'] - data['Start Position'])

            if 'Event Start' in data.columns:
                data['Event Start'] = (data['Event Start'] - data['Start Position']) / (data['End Position'] - data['Start Position'])

            if 'Event End' in data.columns:
                data['Event End'] = (data['Event End'] - data['Start Position']) / (data['End Position'] - data['Start Position'])

            if 'Event Position' in data.columns:
                data['Event Position'] = (data['Event Position'] - data['Start Position']) / (data['End Position'] - data['Start Position'])

            # 初始化MinMaxScaler
            scaler = MinMaxScaler()

            # 获取需要归一化的列名（排除已处理列和标签列）
            feature_columns = data.columns.drop(['Max Position', 'Event Start', 'Event End', 'Event Position', 'Label'])

            # 对其他特征列进行最小-最大值归一化
            data[feature_columns] = scaler.fit_transform(data[feature_columns])

            # 保存归一化后的数据到新的CSV文件
            normalized_file_path = os.path.join(folder_path, "normalized_" + filename)
            data.to_csv(normalized_file_path, index=False)

            print(f"Processed and saved normalized data for {filename}")

# 使用示例
folder_path = r'G:\OTDR_Data\特征值文件 - 副本'  # 替换为你的文件夹路径
normalize_csv_files(folder_path)
