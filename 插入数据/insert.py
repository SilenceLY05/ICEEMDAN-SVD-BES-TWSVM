import os
import numpy as np

def insert_interpolated_points(data, interval, tolerance=0.05):
    interpolated_points = []
    interval_with_tolerance = interval * (1 + tolerance)

    for i in range(len(data) - 1):
        x1, y1 = data[i]
        x2, y2 = data[i + 1]
        interpolated_points.append([x1, y1])

        dist = x2 - x1
        if dist > interval_with_tolerance:
            num_points = int(dist / interval) - 1
            slope = (y2 - y1) / dist

            for j in range(1, num_points + 1):
                new_x = x1 + interval * j
                new_y = y1 + slope * (new_x - x1)
                interpolated_points.append([round(new_x, 7), round(new_y, 3)])

    interpolated_points.append(data[-1])
    return np.array(interpolated_points)

def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # 检查是否为空行
                try:
                    float_line = list(map(float, stripped_line.split(',')))
                    data.append(float_line)
                except ValueError:
                    continue
    return np.array(data)

def write_data_to_txt(data, output_file_path):
    with open(output_file_path, 'w') as file:
        for x, y in data:
            file.write(f"{x:.7f},{y:.3f}\n")

def determine_interval(max_x):
    if max_x < 16:
        return 0.000825
    elif max_x < 30:
        return 0.00126
    elif max_x < 50:
        return 0.00261
    else:
        return 0.00502

def process_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            data = read_data_from_txt(input_file_path)
            if data.size > 0:
                max_x = np.max(data[:, 0])
                interval = determine_interval(max_x)
                interpolated_data = insert_interpolated_points(data, interval)
                write_data_to_txt(interpolated_data, output_file_path)

# 指定输入和输出文件夹路径
input_folder = r'C:\Users\86130\Desktop\新建文件夹'  # 更改为你的输入文件夹路径
output_folder = r'C:\Users\86130\Desktop\新建文件夹 (3)'  # 更改为你的输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理文件夹中的所有TXT文件
process_folder(input_folder, output_folder)
