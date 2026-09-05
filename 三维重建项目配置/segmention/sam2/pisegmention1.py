# import os

# # 主目录
# main_directory = '/extdatashare/dir_wangam00/data_orange/Tomato_kumquat/picture_z'

# # 分割脚本的名称
# segmentation_script = '/extdatashare/dir_wangam00/segmention/sam2/object_seg_box.py'

# # 遍历主目录下的所有次级目录
# for sub_dir in os.listdir(main_directory):
#     sub_dir_path = os.path.join(main_directory, sub_dir)
#     if os.path.isdir(sub_dir_path):
#         # 检查是否存在 images 文件夹
#         images_folder = os.path.join(sub_dir_path, 'images')
#         if os.path.exists(images_folder) and os.path.isdir(images_folder):
#             # 创建 object_mask 文件夹
#             object_mask_folder = os.path.join(sub_dir_path, 'object_mask')
#             os.makedirs(object_mask_folder, exist_ok=True)

#             # 构建命令
#             command = f'python {segmentation_script} --video_dir {images_folder} --output_dir {object_mask_folder}'

#             # 执行命令
#             print(f'正在处理 {sub_dir}...')
#             os.system(command)
#             print(f'{sub_dir} 处理完成。')

# print('所有图片处理完成。')

import os
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='处理图片并生成掩码')
parser.add_argument('--pic_dir', type=str, required=True, help='主目录的路径')

# 解析命令行参数
args = parser.parse_args()

# 获取主目录
main_directory = args.pic_dir

# 分割脚本的名称
segmentation_script = '/extdatashare/dir_wangam00/segmention/sam2/object_seg_box.py'

# 遍历主目录下的所有次级目录
for sub_dir in os.listdir(main_directory):
    sub_dir_path = os.path.join(main_directory, sub_dir)
    if os.path.isdir(sub_dir_path):
        # 检查是否存在 images 文件夹
        images_folder = os.path.join(sub_dir_path, 'images')
        if os.path.exists(images_folder) and os.path.isdir(images_folder):
            # 创建 object_mask 文件夹
            object_mask_folder = os.path.join(sub_dir_path, 'object_mask')
            os.makedirs(object_mask_folder, exist_ok=True)

            # 构建命令
            command = f'python {segmentation_script} --video_dir {images_folder} --output_dir {object_mask_folder}'

            # 执行命令
            print(f'正在处理 {sub_dir}...')
            os.system(command)
            print(f'{sub_dir} 处理完成。')

print('所有图片处理完成。')