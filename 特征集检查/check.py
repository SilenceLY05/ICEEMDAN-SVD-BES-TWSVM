import os
import pandas as pd


def check_feature_files(directory):
    # 遍历指定目录中的所有csv文件
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            # 读取csv文件
            df = pd.read_csv(file_path)

            # 检查最后一列数据中是否有空白
            if df.iloc[:, -1].isnull().any():
                print(f"{filename} 中最后一列存在空白")

            # 检查最后一列数据中是否有数值大于6
            if (df.iloc[:, -1] > 6).any():
                print(f"{filename} 中最后一列存在大于6的数值")

            # 检查倒数第四列到倒数第二列是否有数字后面带逗号的情况
            for col in df.columns[-4:-1]:
                if df[col].astype(str).str.contains(r'\d,').any():
                    print(f"{filename} 中倒数第四列到倒数第二列存在数字后面带逗号的情况")
                    break

            # 检查倒数第二列第一行数值是否是0
            if df.iloc[0, -2] != 0:
                print(f"{filename} 中倒数第二列的第一行不是0")

            # 检查最后一列数据中是否有空白的数据
            if df.iloc[:, -1].eq('').any():
                print(f"{filename} 中最后一列存在空白的数据")

            # 检查最后一列中在最后一个数字5出现后是否后续的全都是6
            last_col = df.iloc[:, -1]
            if 5 in last_col.values:
                index_of_last_5 = last_col[last_col == 5].index[-1]
                if not (last_col.iloc[index_of_last_5 + 1:] == 6).all():
                    print(f"{filename} 中最后一列在最后一个数字5出现后并未全部是6")

            # 检查最后一列中的数字为0或者6时，对应行数的倒数第四列到倒数第三列是否都是空白
            for idx, value in last_col.iteritems():
                if value in [0, 6]:
                    if not (pd.isna(df.iloc[idx, -4]) and pd.isna(df.iloc[idx, -3])):
                        print(f"{filename} 中第 {idx + 1} 行，当最后一列为0或6时，倒数第四列到倒数第三列不为空")
                        break

            # 检查最后一列数据如果出现的是1到5中间的任何一个数值，对应行的倒数第四列到倒数第二列是否是空白
            for idx, value in last_col.iteritems():
                if value in [1, 2, 3, 4, 5]:
                    if pd.isna(df.iloc[idx, -4]) and pd.isna(df.iloc[idx, -3]) and pd.isna(
                            df.iloc[idx, -2]):
                        print(f"{filename} 中第 {idx + 1} 行，当最后一列为1到5时，倒数第四列到倒数第二列为空")
                        break

            # 检查倒数第四列和倒数第三列在一行中没有同时出现数据
            for idx in range(len(df)):
                if pd.isna(df.iloc[idx, -4]) != pd.isna(df.iloc[idx, -3]):
                    print(f"{filename} 中第 {idx + 1} 行，倒数第四列和倒数第三列没有同时出现数据")
                    break

# 使用示例
directory_path = r"G:\OTDR_Data\特征值文件"
check_feature_files(directory_path)
