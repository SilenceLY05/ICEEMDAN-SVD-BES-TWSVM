import os
import shutil


def adjust_data(source_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在就创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(destination_folder, filename)
            error_found = False

            try:
                with open(source_file_path, 'r') as file:
                    lines = file.readlines()

                first_line = lines[0].strip().split(',')
                initial_value = float(first_line[0]) if first_line[0] else 0
                if initial_value < 0:
                    adjustment = abs(initial_value)
                else:
                    adjustment = 0

                adjusted_lines = []
                for line in lines:
                    parts = line.strip().split(',')
                    if parts[0]:
                        try:
                            new_value = float(parts[0]) + adjustment
                            # 格式化第一列的数据，保留7位小数
                            adjusted_line = f"{new_value:.7f},{parts[1]}\n"
                            adjusted_lines.append(adjusted_line)
                        except ValueError:
                            print(f"Warning: Line with incorrect data in file {filename}, skipped.")
                    else:
                        print(f"Warning: Empty first column found in file {filename}, line skipped.")

                # Write adjusted data to a new file
                with open(destination_file_path, 'w') as file:
                    file.writelines(adjusted_lines)

            except Exception as e:
                error_found = True
                print(f"Error processing file {filename}: {e}")

            if not error_found:
                print(f"File {filename} processed successfully.")
            else:
                print(f"Skipped file {filename} due to errors.")


# 使用函数
source_folder = r'C:\Users\86130\Desktop\新建文件夹'
destination_folder = r'C:\Users\86130\Desktop\新建文件夹'
adjust_data(source_folder, destination_folder)
