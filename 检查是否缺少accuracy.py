import os
import re


def check_iterations(files):
    pattern = r"Iteration (\d+)/(\d+), C: ([\d.e-]+), Gamma: ([\d.e-]+), Accuracy: ([\d.e-]+)%"
    for file in files:
        if not os.path.isfile(file):
            print(f"File '{file}' not found.")
            continue

        with open(file, 'r') as f:
            data = f.readlines()

        iterations_dict = {}
        for line in data:
            match = re.match(pattern, line.strip())
            if match:
                c_value = float(match.group(3))
                gamma_value = float(match.group(4))

                key = (c_value, gamma_value)
                if key not in iterations_dict:
                    iterations_dict[key] = 0
                iterations_dict[key] += 1

        for key, count in iterations_dict.items():
            if count != 5:
                print(f"File '{file}' has {count} entries for C: {key[0]}, Gamma: {key[1]} (Expected 5).")


# 文件列表
files = ["lssvm.txt", "SVM.txt", "twsvm.txt"]
check_iterations(files)
