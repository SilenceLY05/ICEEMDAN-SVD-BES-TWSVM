import csv
import os

def tsv_to_txt(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tsv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.tsv', '.txt'))
            
            with open(input_path, mode='r', newline='', encoding='utf-8') as file_in:
                # 读取TSV文件
                reader = csv.reader(file_in, delimiter='\t')
                with open(output_path, mode='w', newline='', encoding='utf-8') as file_out:
                    writer = csv.writer(file_out, delimiter=',')
                    
                    # 遍历文件中的每一行，只保存第一列和最后一列
                    for row in reader:
                        if row:  # 确保行不为空
                            # 写入第一列和最后一列到输出文件
                            writer.writerow([row[0], row[-1]])

# 使用示例
input_folder = r'G:\毕业论文文件\测试数据'  # 输入文件夹路径
output_folder = r'G:\毕业论文文件\txt测试数据'  # 输出文件夹路径
tsv_to_txt(input_folder, output_folder)
