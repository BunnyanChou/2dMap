import os

# 1. 指定文件夹路径和输出txt文件路径
folder_path = "/home/heda/zyy/dataset/phantom3_village-kfs/"  # 替换为你的文件夹路径
output_txt = os.path.join(folder_path, 'rgb.txt')  # 你想输出的txt文件名

# 2. 获取文件夹中的所有文件名并排序
file_names = os.listdir(os.path.join(folder_path, 'rgb'))
sorted_file_names = sorted(file_names)  # 使用自然排序

# 3. 过滤出图片文件
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG')  # 需要的图片格式
image_files = [f for f in sorted_file_names if f.lower().endswith(image_extensions)]

# 4. 将文件名写入txt文件
with open(output_txt, 'w') as file:
    for idx, image in enumerate(image_files):  # 从1开始枚举
        file.write(f"{idx} rgb/{image}\n")