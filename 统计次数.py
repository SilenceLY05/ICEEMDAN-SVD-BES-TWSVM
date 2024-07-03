import os
import pandas as pd


def process_csv_files(directory):
    # 初始化计数器
    counts = {i: 0 for i in range(1, 6)}

    # 遍历目录中的所有CSV文件
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            # 忽略第一列，统计倒数第二列和最后一列的数值
            df = df.iloc[:, 1:]
            df = df.dropna(subset=[df.columns[-2]])  # 移除倒数第二列为空的行
            df = df.drop_duplicates(subset=[df.columns[-2]])  # 倒数第二列相同的值只保留一个

            # 统计最后一列中1-5的出现次数
            for value in df.iloc[:, -1]:
                if value in counts:
                    counts[value] += 1

    return counts


# 使用示例
directory_path = r'G:\OTDR_Data\特征值文件'  # 替换为你的CSV文件所在的目录
counts = process_csv_files(directory_path)
print(counts)
