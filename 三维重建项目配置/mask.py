# #单个视频文件夹
# import os
# import cv2
# from tqdm import tqdm

# def process_images(images_dir, mask_dir, output_dir):
#     # 获取图片文件列表
#     image_files = sorted(os.listdir(images_dir))
#     mask_files = sorted(os.listdir(mask_dir))

#     # 确保图片文件和掩码文件数量一致
#     if len(image_files) != len(mask_files):
#         print("原图片和掩码图片数量不一致，请检查！")
#         return

#     # 处理每一对图像
#     for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
#         # 读取原图片和掩码图片
#         image_path = os.path.join(images_dir, image_file)
#         mask_path = os.path.join(mask_dir, mask_file)
#         try:
#             image = cv2.imread(image_path)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#             if image is None or mask is None:
#                 print(f"无法读取图像: {image_path} 或 {mask_path}")
#                 continue

#             # 掩码处理，去除背景
#             result = cv2.bitwise_and(image, image, mask=mask)

#             # 保存处理后的图片
#             output_path = os.path.join(output_dir, image_file)
#             cv2.imwrite(output_path, result)
#         except Exception as e:
#             print(f"处理 {image_path} 时出错: {e}")

#     print("处理完成，处理后的图片已保存到", output_dir)

# # 主目录
# main_dir = "/extdatashare/dir_wangam00/wheat4/plant/qu_bac_vgg/done/WeChat_20250513111253_frames"

# # 原图片目录
# images_dir = os.path.join(main_dir, "images")

# # 掩码图片目录
# mask_dir = os.path.join(main_dir, "object_mask")

# # 新的次级目录
# output_dir = os.path.join(main_dir, "images_qb")

# # 创建新的次级目录
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 处理图像
# process_images(images_dir, mask_dir, output_dir)

# # 重命名目录
# new_images_real_dir = os.path.join(main_dir, "images_real")
# try:
#     os.rename(images_dir, new_images_real_dir)
#     print(f"已将 {images_dir} 重命名为 {new_images_real_dir}")
# except FileExistsError:
#     print(f"重命名失败，目标目录 {new_images_real_dir} 已存在。")
# except FileNotFoundError:
#     print(f"重命名失败，源目录 {images_dir} 不存在。")
# except Exception as e:
#     print(f"重命名 {images_dir} 时出现未知错误: {e}")

# new_images_dir = os.path.join(main_dir, "images")
# try:
#     os.rename(output_dir, new_images_dir)
#     print(f"已将 {output_dir} 重命名为 {new_images_dir}")
# except FileExistsError:
#     print(f"重命名失败，目标目录 {new_images_dir} 已存在。")
# except FileNotFoundError:
#     print(f"重命名失败，源目录 {output_dir} 不存在。")
# except Exception as e:
#     print(f"重命名 {output_dir} 时出现未知错误: {e}")


#多个视频的文件夹掩码
import os
import cv2
from tqdm import tqdm
import argparse

def process_images(images_dir, mask_dir, output_dir):
    """处理单个视频文件夹中的图像和掩码"""
    # 获取图片文件列表并排序
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # 确保图片文件和掩码文件数量一致
    if len(image_files) != len(mask_files):
        print(f"警告：{images_dir} 中的原图片和掩码图片数量不一致，跳过该目录！")
        return

    # 处理每一对图像
    for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc=f"处理 {images_dir}"):
        # 读取原图片和掩码图片
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"无法读取图像: {image_path} 或 {mask_path}")
                continue

            # 掩码处理，去除背景
            result = cv2.bitwise_and(image, image, mask=mask)

            # 保存处理后的图片
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, result)
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")

    print(f"处理完成，处理后的图片已保存到 {output_dir}")


def batch_process_videos(main_dir):
    """批量处理顶层目录中的所有视频文件夹"""
    # 遍历顶层目录中的所有子目录
    for video_folder in os.listdir(main_dir):
        video_folder_path = os.path.join(main_dir, video_folder)
        if os.path.isdir(video_folder_path):  # 确保是目录
            print(f"\n正在处理视频文件夹: {video_folder}")

            # 定义输入和输出目录
            images_dir = os.path.join(video_folder_path, "images")
            mask_dir = os.path.join(video_folder_path, "object_mask")
            output_dir = os.path.join(video_folder_path, "images_qb")  # 输出目录位于视频文件夹内部

            # 检查输入目录是否存在
            if not os.path.exists(images_dir) or not os.path.exists(mask_dir):
                print(f"跳过 {video_folder}：缺少 images 或 object_mask 目录")
                continue

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 处理图像
            process_images(images_dir, mask_dir, output_dir)

            # 重命名目录
            new_images_real_dir = os.path.join(video_folder_path, "images_real")
            new_images_dir = os.path.join(video_folder_path, "images")

            try:
                os.rename(images_dir, new_images_real_dir)
                print(f"已将 {images_dir} 重命名为 {new_images_real_dir}")
            except Exception as e:
                print(f"重命名 {images_dir} 时出现错误: {e}")

            try:
                os.rename(output_dir, new_images_dir)
                print(f"已将 {output_dir} 重命名为 {new_images_dir}")
            except Exception as e:
                print(f"重命名 {output_dir} 时出现错误: {e}")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='批量处理图像与掩码，生成去除背景的图像')
    
    # 添加主目录参数
    parser.add_argument('--main_dir', required=True, 
                        help='包含多个视频文件夹的顶层目录路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行批量处理
    batch_process_videos(args.main_dir)

