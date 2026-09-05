# import cv2
# import os

# # 输入视频文件夹路径
# input_folder = '/extdatashare/dir_wangam00/yumi2/video/'  # 替换为包含视频文件的文件夹
# # 输出帧的根目录
# output_root_folder = '/extdatashare/dir_wangam00/yumi2/picture/'  # 存放抽帧图像的根目录

# # 创建输出根目录
# os.makedirs(output_root_folder, exist_ok=True)

# # 遍历输入文件夹中的所有文件
# for video_file in os.listdir(input_folder):
#     if video_file.endswith(('.mp4', '.avi', '.mov')):  # 检查视频文件类型
#         video_path = os.path.join(input_folder, video_file)
        
#         # 创建以视频文件名命名的输出文件夹
#         video_name = os.path.splitext(video_file)[0]  # 获取视频文件名（不带扩展名）
#         output_folder = os.path.join(output_root_folder, video_name + '_frames')
#         os.makedirs(output_folder, exist_ok=True)

#         # 打开视频文件
#         cap = cv2.VideoCapture(video_path)

#         # 初始化帧计数器
#         frame_count = 0
#         saved_frame_count = 0  # 用于记录保存的帧数

#         while True:
#             # 读取下一帧
#             ret, frame = cap.read()
            
#             # 如果没有读取到帧，退出循环
#             if not ret:
#                 break

#             # 每隔5帧保存一帧
#             if frame_count % 15 == 0:
#                 # 使用格式化字符串保存帧文件名，命名从0001开始
#                 frame_filename = os.path.join(output_folder, f'{saved_frame_count + 1:04}.jpg')
#                 cv2.imwrite(frame_filename, frame)
#                 saved_frame_count += 1  # 增加保存的帧计数

#             # 增加总帧计数器
#             frame_count += 1

#         # 释放视频捕获对象
#         cap.release()

#         print(f'视频 "{video_file}" 抽取完成，共保存 {saved_frame_count} 帧图像到 "{output_folder}" 目录。')

import cv2
import os
import argparse

def extract_frames(input_folder, output_root_folder):
    # 创建输出根目录
    os.makedirs(output_root_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for video_file in os.listdir(input_folder):
        # 检查视频文件类型
        if video_file.lower().endswith(("mp4", ".avi", ".mov", ".mkv", ".flv")):
            video_path = os.path.join(input_folder, video_file)
            
            # 创建以视频文件名命名的输出文件夹
            video_name = os.path.splitext(video_file)[0]  # 获取视频文件名（不带扩展名）
            output_folder = os.path.join(output_root_folder, f"{video_name}_frames")
            os.makedirs(output_folder, exist_ok=True)

            # 创建 images 子目录
            images_folder = os.path.join(output_folder, 'images')
            os.makedirs(images_folder, exist_ok=True)

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)

            # 检查视频是否成功打开
            if not cap.isOpened():
                print(f"警告：无法打开视频文件 {video_file}")
                continue

            # 初始化帧计数器
            frame_count = 0
            saved_frame_count = 0  # 用于记录保存的帧数

            while True:
                # 读取下一帧
                ret, frame = cap.read()
                
                # 如果没有读取到帧，退出循环
                if not ret:
                    break

                # 每隔30帧保存一帧（可根据需要修改这个数值）
                if frame_count % 30 == 0:
                    # 使用格式化字符串保存帧文件名，命名从0001开始
                    frame_filename = os.path.join(images_folder, f"{saved_frame_count + 1:04}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_frame_count += 1  # 增加保存的帧计数

                # 增加总帧计数器
                frame_count += 1

            # 释放视频捕获对象
            cap.release()

            print(f'视频 "{video_file}" 抽取完成，共保存 {saved_frame_count} 帧图像到 "{images_folder}" 目录。')

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='从视频文件中抽取帧并保存为图片')
    
    # 添加命令行参数
    parser.add_argument('--input', required=True, help='包含视频文件的输入文件夹路径')
    parser.add_argument('--output', required=True, help='保存抽取帧图像的输出根目录路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用抽帧函数
    extract_frames(args.input, args.output)
