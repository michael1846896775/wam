import os
import shutil
import argparse

def batch_move_and_rename_ply_files(source_dir):
    """
    批量移动并重命名PLY点云文件，自动在主目录下创建ply文件夹作为目标路径。

    :param source_dir: 主目录路径
    """
    # 自动构建目标路径（主目录下的ply文件夹）
    target_dir = os.path.join(source_dir, "ply")
    
    # 确保目标路径存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标路径: {target_dir}")

    # 遍历主目录下的所有文件夹
    for video_folder in os.listdir(source_dir):
        video_folder_path = os.path.join(source_dir, video_folder)
        
        # 确保是文件夹且不是目标ply文件夹（避免处理自身）
        if os.path.isdir(video_folder_path) and video_folder != "ply":
            # 构建PLY文件的完整路径
            ply_folder_path = os.path.join(video_folder_path, "output/point_cloud/iteration_7000")
            
            # 检查路径是否存在
            if os.path.exists(ply_folder_path):
                # 遍历PLY文件夹中的所有文件
                for file_name in os.listdir(ply_folder_path):
                    if file_name.endswith(".ply"):  # 确保是PLY文件
                        # 构建源文件路径和目标文件路径
                        source_file_path = os.path.join(ply_folder_path, file_name)
                        target_file_name = f"{video_folder}.ply"  # 以视频文件夹的名字重命名
                        target_file_path = os.path.join(target_dir, target_file_name)
                        
                        # 移动文件（如果目标文件已存在，会直接覆盖）
                        shutil.move(source_file_path, target_file_path)
                        print(f"移动文件: {source_file_path} -> {target_file_path}")
            else:
                print(f"路径不存在，跳过文件夹: {ply_folder_path}")
        else:
            # 跳过非文件夹项和目标ply文件夹
            if not os.path.isdir(video_folder_path):
                print(f"跳过非文件夹项: {video_folder_path}")

    print("批量移动和重命名完成！")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='批量移动并重命名PLY点云文件')
    # 添加主目录路径参数（必选）
    parser.add_argument('--source_dir', type=str, help='主目录路径（例如：/extdatashare/dir_lzouyi00/wheat3_jiaoji）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数执行操作
    batch_move_and_rename_ply_files(args.source_dir)