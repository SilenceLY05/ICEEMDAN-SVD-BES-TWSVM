import os

def read_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 跳过空行
                parts = line.split(',')
                iteration = parts[0].split(' ')[-1].strip()
                c_value = parts[1].split(': ')[-1].strip()
                gamma = parts[2].split(': ')[-1].strip()
                accuracy_str = parts[3].split(': ')[-1].strip().replace('%', '')
                try:
                    accuracy = float(accuracy_str)
                except ValueError:
                    print(f"Could not convert accuracy value to float: '{accuracy_str}' in line: '{line}'")
                    continue
                data.append((iteration, c_value, gamma, accuracy))
    return data

def main(file_paths):
    all_data = []
    for file_path in file_paths:
        all_data.extend(read_file(file_path))

    # 按照Accuracy从大到小排序
    sorted_data = sorted(all_data, key=lambda x: x[-1], reverse=True)

    # 输出排序后的结果
    for entry in sorted_data:
        print(f"Iteration {entry[0]}, C: {entry[1]}, Gamma: {entry[2]}, Accuracy: {entry[3]}%")


# 读取三个txt文件路径
file_paths = ['lssvm.txt']
main(file_paths)
