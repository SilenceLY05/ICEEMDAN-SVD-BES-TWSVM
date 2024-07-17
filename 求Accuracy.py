import re
from collections import defaultdict

# 初始化数据结构
accuracy_data = defaultdict(list)

# 读取txt文件
with open('path/to/your/file.txt', 'r') as file:
    for line in file:
        # 使用正则表达式提取C值, Gamma值和Accuracy
        match = re.search(r'C: ([\d.]+), Gamma: ([\de.-]+), Accuracy: ([\d.]+)%', line)
        if match:
            C = match.group(1)
            Gamma = match.group(2)
            Accuracy = float(match.group(3))
            # 将Accuracy加入到对应的C值和Gamma值组中
            accuracy_data[(C, Gamma)].append(Accuracy)

# 计算每组的平均值，并找到最大平均值
max_avg_accuracy = 0
best_params = None
for params, accuracies in accuracy_data.items():
    avg_accuracy = sum(accuracies) / len(accuracies)
    if avg_accuracy > max_avg_accuracy:
        max_avg_accuracy = avg_accuracy
        best_params = params

print(f"Best parameters: C={best_params[0]}, Gamma={best_params[1]}, Average Accuracy={max_avg_accuracy:.2f}%")
