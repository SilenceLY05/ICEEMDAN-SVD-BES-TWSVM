import os

def filter_rows_by_threshold(file_path, threshold, output_file):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                # 将第一列的值转换为浮点数
                value = float(parts[0])
                # 检查第一列的值是否大于阈值
                if value <= threshold:
                    output_file.write(line)
            except ValueError:
                # 如果转换失败，忽略这一行
                continue

def process_folder(folder_path, threshold):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(folder_path, f'filtered_{filename}')
            with open(output_file_path, 'w') as output_file:
                filter_rows_by_threshold(input_file_path, threshold, output_file)
                print(f'Processed {filename} -> {output_file_path}')

# 设置文件夹路径和阈值
folder_path = r'C:\Users\86130\Desktop\厦门备纤测试记录\测试记录\JINZHANG'  # 请替换成你的文件夹路径
threshold = 80# 请设置你需要的阈值
process_folder(folder_path, threshold)
