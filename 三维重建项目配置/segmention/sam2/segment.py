import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# 创建解析器
parser = argparse.ArgumentParser(description="Batch segmentation tool for SAM2 with mask processing")

# 添加命令行参数
parser.add_argument(
    '--annotation_file', 
    type=str, 
    default='./all_annotations.txt', 
    help="File containing all annotation data"
)

# 解析命令行参数
args = parser.parse_args()

# 获取传入的参数
annotation_file = args.annotation_file

print(f"Annotation file: {annotation_file}")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("\nSupport for MPS devices is preliminary.")

colors = [np.array([0, 128, 255], dtype=np.uint8),
         np.array([255, 128, 0], dtype=np.uint8),
         np.array([0, 128, 128], dtype=np.uint8),
         np.array([55, 128, 50], dtype=np.uint8)]

def add_mask2(image, mask, color_id):
    int_mask = mask.astype(np.uint8)
    int_mask_3d = np.dstack((int_mask, int_mask, int_mask))

    mask_color = colors[color_id]
    mask = np.full_like(image, mask_color)

    mask[int_mask == 0] = 0

    alpha = 0
    beta = 1
    gamma = 0

    res = cv2.addWeighted(image, alpha, mask, beta, gamma, dtype=cv2.CV_8U)

    black_areas = int_mask_3d == 0
    res[black_areas] = image[black_areas]
    return res

def load_all_annotations(annotation_file):
    """
    加载所有视频的标注数据
    """
    try:
        with open(annotation_file, 'r') as f:
            all_annotations = json.load(f)
        return all_annotations.get('videos', [])
    except Exception as e:
        print(f"Error reading annotation file {annotation_file}: {e}")
        return []

def process_images_with_mask(images_dir, mask_dir, output_dir):
    """
    处理单个视频文件夹中的图像和掩码，去除背景
    """
    # 获取图片文件列表并排序
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])

    # 确保图片文件和掩码文件数量一致
    if len(image_files) != len(mask_files):
        print(f"警告：{images_dir} 中的原图片和掩码图片数量不一致，跳过该目录！")
        return False

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每一对图像
    for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc=f"处理 {os.path.basename(images_dir)}"):
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

    print(f"掩码处理完成，处理后的图片已保存到 {output_dir}")
    return True

def rename_directories(video_dir):
    """
    重命名目录：images -> images_real, images_qb -> images
    """
    images_dir = os.path.join(video_dir, "images")
    output_dir = os.path.join(video_dir, "images_qb")
    new_images_real_dir = os.path.join(video_dir, "images_real")
    new_images_dir = os.path.join(video_dir, "images")

    try:
        if os.path.exists(images_dir):
            os.rename(images_dir, new_images_real_dir)
            print(f"已将 {images_dir} 重命名为 {new_images_real_dir}")
        else:
            print(f"源目录 {images_dir} 不存在，跳过重命名")
    except Exception as e:
        print(f"重命名 {images_dir} 时出现错误: {e}")
        return False

    try:
        if os.path.exists(output_dir):
            os.rename(output_dir, new_images_dir)
            print(f"已将 {output_dir} 重命名为 {new_images_dir}")
        else:
            print(f"源目录 {output_dir} 不存在，跳过重命名")
    except Exception as e:
        print(f"重命名 {output_dir} 时出现错误: {e}")
        return False

    return True

def process_single_video(annotation_data):
    """
    处理单个视频的分割和掩码处理
    """
    video_dir = annotation_data['video_dir']
    
    # 提取注解信息
    points = {int(k): np.array(v, dtype=np.float32) if v else None for k, v in annotation_data['points'].items()}
    labels = {int(k): np.array(v, dtype=np.int32) if v else None for k, v in annotation_data['labels'].items()}
    boxes = {int(k): np.array(v, dtype=np.int32) if v else None for k, v in annotation_data['boxes'].items()}
    ann_obj_id = annotation_data['ann_obj_id']
    frame_idx = annotation_data['frame_idx']

    print(f"Processing video: {os.path.basename(video_dir)}")
    print(f"Loaded annotations for object IDs: {ann_obj_id}")

    # 创建输出目录 - object_mask与images同级
    mask_dir = os.path.join(video_dir, "object_mask")
    os.makedirs(mask_dir, exist_ok=True)
    print(f"Mask output directory: {mask_dir}")

    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    try:
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

    # 查找图像文件
    images_dir = os.path.join(video_dir, "images")
    frame_names = sorted(
        [p for p in os.listdir(images_dir) if p.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png"))],
        key=lambda p: int(os.path.splitext(p)[0]) if p.split('.')[0].isdigit() else p,
    )

    if not frame_names:
        print(f"No image files found in {images_dir}")
        return False

    # 初始化推理状态
    try:
        inference_state = predictor.init_state(video_path=images_dir)
        predictor.reset_state(inference_state)
    except Exception as e:
        print(f"Error initializing inference state: {e}")
        return False

    # 处理每个对象ID
    for obj_id in ann_obj_id:
        if boxes.get(obj_id) is not None and len(boxes[obj_id]) > 0:
            boxes_obj = boxes[obj_id][-1]
        else:
            boxes_obj = None
            
        points_obj = points.get(obj_id)
        labels_obj = labels.get(obj_id)
        
        print(f"Processing object ID {obj_id}: points={points_obj is not None}, box={boxes_obj is not None}")
        
        try:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_obj,
                labels=labels_obj,
                box=boxes_obj,
            )
        except Exception as e:
            print(f"Error processing object {obj_id}: {e}")
            continue

    # 运行传播并收集结果
    video_segments = {}
    try:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            print(f"Frame {out_frame_idx}: objects {out_obj_ids}")
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    except Exception as e:
        print(f"Error during propagation: {e}")
        return False

    # 保存掩码
    for out_frame_idx in range(len(frame_names)):
        if out_frame_idx not in video_segments:
            continue
            
        color_num = 0
        img_path = os.path.join(images_dir, frame_names[out_frame_idx])
        now_img = cv2.imread(img_path)
        if now_img is None:
            print(f"Error reading image: {img_path}")
            continue
            
        result_img = np.full_like(now_img, 0)
        
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            result_img = add_mask2(result_img, np.squeeze(out_mask), color_num)
            color_num += 1
            
        mask_out_name = os.path.join(mask_dir, f"{str(out_frame_idx).zfill(4)}.jpg")
        print(f"Saving mask: {mask_out_name}")
        cv2.imwrite(mask_out_name, result_img)

    print(f"All masks saved to {mask_dir}")

    # 进行掩码处理（去除背景）
    print(f"\n开始掩码处理...")
    output_dir = os.path.join(video_dir, "images_qb")
    mask_success = process_images_with_mask(images_dir, mask_dir, output_dir)
    
    if mask_success:
        # 重命名目录
        print(f"\n开始重命名目录...")
        rename_success = rename_directories(video_dir)
        if rename_success:
            print(f"所有处理完成: {os.path.basename(video_dir)}")
        else:
            print(f"目录重命名失败: {os.path.basename(video_dir)}")
            return False
    else:
        print(f"掩码处理失败: {os.path.basename(video_dir)}")
        return False

    return True

def main():
    # 加载所有注解数据
    all_annotations = load_all_annotations(annotation_file)
    
    if not all_annotations:
        print(f"No annotation data found in {annotation_file}")
        return
    
    print(f"Found {len(all_annotations)} video annotations to process")
    
    for i, annotation_data in enumerate(all_annotations, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(all_annotations)}: {os.path.basename(annotation_data['video_dir'])}")
        print(f"{'='*60}")
        
        # 处理当前视频（包括分割和掩码处理）
        success = process_single_video(annotation_data)
        
        if success:
            print(f"Completed processing: {os.path.basename(annotation_data['video_dir'])}")
        else:
            print(f"Failed to process: {os.path.basename(annotation_data['video_dir'])}")
        
        print(f"Progress: {i}/{len(all_annotations)} completed")
    
    print(f"\nBatch segmentation and mask processing completed! Processed {len(all_annotations)} videos.")

if __name__ == "__main__":
    main()