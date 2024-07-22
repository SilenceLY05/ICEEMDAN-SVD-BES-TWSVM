import os

def process_file(file_path):
    # 创建一个新的文件路径用于保存处理后的数据
    new_file_path = os.path.splitext(file_path)[0] + '_processed.txt'

    with open(file_path, 'r') as file, open(new_file_path, 'w') as new_file:
        for line in file:
            data = line.strip().split(',')
            # 确保每行有两列数据
            if len(data) == 2:
                try:
                    # 尝试将第一列转换为浮点数，并进行四舍五入保留整数
                    col1 = int(round(float(data[0])))
                    # 尝试将第二列转换为浮点数，并进行四舍五入保留小数点后3位
                    col2 = round(float(data[1]), 3)
                    # 使用110减去第二列的数值
                    col2 = round(110 - col2, 3)
                    # 将处理后的数据写入新文件
                    new_file.write(f"{col1},{col2:.3f}\n")
                except ValueError:
                    # 输出无法解析为数字的行
                    print(f"Skipping invalid line (unable to parse numbers): {line.strip()}")
            else:
                # 输出不包含两列数据的行
                print(f"Skipping invalid line (incorrect number of columns): {line.strip()}")



# 文件列表
file_list = ['IBES-TWSVM.txt', 'BES-TWSVM.txt', 'PSO-TWSVM.txt']

for file_path in file_list:
    process_file(file_path)

print("处理完成")
