import os

def change_delimiter(directory, old_delimiter=';', new_delimiter=','):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 替换分隔符
            if old_delimiter in content:
                updated_content = content.replace(old_delimiter, new_delimiter)
                # 写回文件
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(updated_content)
                print(f"Updated delimiter in {filename}")

# 调用函数
directory = r'G:\毕业论文文件\txt测试数据2'  # 你的文件夹路径
change_delimiter(directory)
