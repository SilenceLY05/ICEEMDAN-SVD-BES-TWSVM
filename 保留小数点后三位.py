import os

def process_files(directory):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    for file in files:
        file_path = os.path.join(directory, file)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 处理每一行的第二列数据
        new_lines = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    # 保留第二列小数点后三位
                    parts[1] = "{:.3f}".format(float(parts[1]))
                    new_line = ','.join(parts)
                    new_lines.append(new_line)
                except ValueError:
                    # 如果无法转换为浮点数，保留原数据
                    new_lines.append(line.strip())
            else:
                new_lines.append(line.strip())
        
        # 写入修改后的内容到原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')

# 使用示例，指定文件夹路径
directory = r'C:\Users\86130\Desktop\新建文件夹'
process_files(directory)
